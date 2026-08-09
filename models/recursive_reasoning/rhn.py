from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import itertools
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import (rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding,
                           CastedParameter, CastedLinear, DynamicSwiGLU, DynamicAttention)
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class RHN_ACTV1InnerCarry:
    z: torch.Tensor
    inner_steps: torch.Tensor

@dataclass
class RHN_ACTV1Carry:
    inner_carry: RHN_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]

class RHN_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

    hypernet_hidden_size: int
    hypernet_rank: int
    layer_emb_dim: int
    hypernet_relative_scale: float
    kron_dims: int
    kron_dims_mult: bool
    perceiver_heads: int
    hypernet_l2_lambda: float = 1e-4

    hypernet_attn: bool
    hypernet_attn_type: str
    hypernet_rmsnorm: bool
    hypernet_rmsaffine: bool

    # Size of the parallel residual streams for mHC-lite
    mhc_window_size: int = 4


class MHCLiteMixer(nn.Module):
    """Generates a Doubly Stochastic Matrix to safely mix parallel residual streams."""
    def __init__(self, num_streams: int):
        super().__init__()
        self.num_streams = num_streams

        # Generate all permutations for the given number of streams
        perms = list(itertools.permutations(range(num_streams)))
        self.num_perms = len(perms)

        # Create the permutation matrices
        # Shape: (num_perms, num_streams, num_streams)
        perm_matrices = torch.zeros(self.num_perms, num_streams, num_streams)
        for i, p in enumerate(perms):
            for j, val in enumerate(p):
                perm_matrices[i, j, val] = 1.0

        # Register as a buffer so it automatically moves to the correct device
        self.register_buffer("perm_matrices", perm_matrices)

        # Learnable logits for the convex combination
        self.logits = nn.Parameter(torch.zeros(self.num_perms))

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        # streams shape: [Batch, Streams, Seq, Dim]

        # 1. Softmax to get valid Birkhoff-von Neumann convex coefficients
        weights = F.softmax(self.logits, dim=0)

        # 2. Combine permutation matrices: W_mix = sum(w_i * P_i)
        # w_mix shape: [Streams, Streams]
        w_mix = torch.tensordot(weights, self.perm_matrices.to(weights.dtype), dims=([0], [0]))

        # 3. Mix the streams
        # 'ij' is the mixing matrix [Out_Stream, In_Stream]
        # 'bjkl' is the batch data [Batch, In_Stream, Seq, Dim]
        # Result 'bikl' is the newly mixed streams [Batch, Out_Stream, Seq, Dim]
        mixed_streams = torch.einsum('ij, bjkl -> bikl', w_mix.to(streams.dtype), streams)

        return mixed_streams


class RHN_ACTV1Block(nn.Module):
    def __init__(self, config: RHN_ACTV1Config, attn: bool = True, attn_type: str = "self",
                 attn_params: dict = None, rmsnorm: bool = True) -> None:
        super().__init__()

        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.attn = attn
        self.attn_type = attn_type
        self.rmsnorm = rmsnorm

        if self.attn:
            self.pre_attn_norm = nn.RMSNorm(attn_params["input_size"], eps=self.config.rms_norm_eps, elementwise_affine=self.config.hypernet_rmsaffine).to(dtype=self.forward_dtype)

            if self.attn_type == "mlp_t":
                self.mlp_t = SwiGLU(hidden_size=attn_params["seq_len"], expansion=config.expansion)
            elif self.attn_type == "self":
                self.self_attn = Attention(
                    hidden_size=attn_params["input_size"],
                    kdim=attn_params["input_size"],
                    vdim=attn_params["input_size"],
                    head_dim=attn_params["input_size"] // attn_params["heads"],
                    num_heads=attn_params["heads"],
                    num_key_value_heads=attn_params["heads"],
                    causal=False,
                )
            elif self.attn_type == "perceiver":
                self.kv_norm = nn.RMSNorm(attn_params["kv_size"],
                                          eps=self.config.rms_norm_eps,
                                          elementwise_affine=self.config.hypernet_rmsaffine).to(dtype=self.forward_dtype)
                self.perceiver_attn = nn.MultiheadAttention(
                    embed_dim=attn_params["input_size"],
                    kdim=attn_params["kv_size"],
                    vdim=attn_params["kv_size"],
                    num_heads=attn_params["heads"],
                    batch_first=True,
                ).to(dtype=self.forward_dtype)

        if self.rmsnorm:
            self.pre_mlp_norm = nn.RMSNorm(attn_params["input_size"], eps=self.config.rms_norm_eps, elementwise_affine=self.config.hypernet_rmsaffine).to(dtype=self.forward_dtype)

        self.mlp = SwiGLU(hidden_size=attn_params["input_size"], expansion=config.expansion)

    def forward(self, hidden_states: torch.Tensor, kv: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if self.attn:
            normed = self.pre_attn_norm(hidden_states)
            if self.attn_type == "mlp_t":
                normed = normed.transpose(1, 2)
                attn_out = self.mlp_t(normed).transpose(1, 2)
            elif self.attn_type == "self":
                attn_out = self.self_attn(cos_sin=None, query=normed, key=normed, value=normed)
            elif self.attn_type == "perceiver":
                if normed.dim() == 2:
                    normed = normed.unsqueeze(0)
                if normed.dim() == 3 and normed.size(0) == 1 and kv is not None:
                    batch_size = kv.shape[0]
                    normed = normed.expand(batch_size, -1, -1)
                normed_kv = self.kv_norm(kv) if kv is not None else None
                attn_out, _ = self.perceiver_attn(query=normed, key=normed_kv, value=normed_kv, need_weights=False)

            hidden_states = hidden_states + attn_out

        if self.rmsnorm:
            normed = self.pre_mlp_norm(hidden_states)
            out = self.mlp(normed)
            hidden_states = hidden_states + out
        else:
            out = self.mlp(hidden_states)
            hidden_states = hidden_states + out

        return hidden_states


class RHN_ACTV1Block_Dynamic(nn.Module):
    def __init__(self, config: RHN_ACTV1Config, attn: bool = True, attn_type: str = "self") -> None:
        super().__init__()

        self.config = config
        self.attn = attn
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.pre_attn_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, elementwise_affine=True).to(dtype=self.forward_dtype)
        self.pre_mlp_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, elementwise_affine=True).to(dtype=self.forward_dtype)

        if self.attn:
            if self.config.mlp_t:
                self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
                self.mlp_t = DynamicSwiGLU(hidden_size=self.config.seq_len + self.puzzle_emb_len, expansion=config.expansion)
            else:
                self.self_attn = DynamicAttention(hidden_size=config.hidden_size, head_dim=config.hidden_size // config.num_heads, num_heads=config.num_heads, num_key_value_heads=config.num_heads, causal=False)

        self.mlp = DynamicSwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)

    def set_dynamic_adapter(self, dynamic_weights: Dict[str, torch.Tensor], layer_idx: int):
        if self.attn:
            if self.config.mlp_t:
                gate_up = dynamic_weights[f"L_level.{layer_idx}.mlp_t.gate_up_proj.weight"]
                down = dynamic_weights[f"L_level.{layer_idx}.mlp_t.down_proj.weight"]
                self.mlp_t.set_dynamic_adapter(gate_up[0], gate_up[1], down[0], down[1])
            else:
                qkv = dynamic_weights[f"L_level.{layer_idx}.self_attn.qkv_proj.weight"]
                o = dynamic_weights[f"L_level.{layer_idx}.self_attn.o_proj.weight"]
                self.self_attn.set_dynamic_adapter(qkv[0], qkv[1], o[0], o[1])

        mlp_gate_up = dynamic_weights[f"L_level.{layer_idx}.mlp.gate_up_proj.weight"]
        mlp_down = dynamic_weights[f"L_level.{layer_idx}.mlp.down_proj.weight"]
        self.mlp.set_dynamic_adapter(mlp_gate_up[0], mlp_gate_up[1], mlp_down[0], mlp_down[1])

    def clear_dynamic_adapter(self):
        self.mlp.clear_dynamic_adapter()
        if self.attn:
            if self.config.mlp_t:
                self.mlp_t.clear_dynamic_adapter()
            else:
                self.self_attn.clear_dynamic_adapter()

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.attn:
            residual = hidden_states
            normed = self.pre_attn_norm(hidden_states)
            if self.config.mlp_t:
                normed = normed.transpose(1, 2)
                attn_out = self.mlp_t(normed).transpose(1, 2)
            else:
                attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=normed)

            hidden_states = residual + attn_out

        residual = hidden_states
        normed = self.pre_mlp_norm(hidden_states)
        mlp_out = self.mlp(normed)
        hidden_states = residual + mlp_out

        return hidden_states


class RHN_Hypernetwork(nn.Module):
    def __init__(self, config: RHN_ACTV1Config, layer_specs) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.layer_specs = layer_specs
        self.config_per_layer = {}
        for name, shape in self.layer_specs:
            self.config_per_layer[name] = {
                "shape": shape,
                "type": "vector" if self._is_vector_like(shape) else "matrix",
            }

        self.embed_scale = math.sqrt(self.config.hypernet_hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.input_size = self.config.hidden_size * self.config.L_layers
        self.num_layers = len(self.layer_specs)

        if self.config.kron_dims_mult:
            self.num_queries = self.num_layers * self.config.kron_dims
        else:
            self.num_queries = self.config.kron_dims

        self.input_queries = nn.Parameter(
            trunc_normal_init_(
                torch.empty((1, self.num_queries, self.config.hypernet_hidden_size), dtype=self.forward_dtype),
                std=1.0 / math.sqrt(self.config.hypernet_hidden_size),
            )
        )

        self.hypernet_base = nn.ModuleList()
        self.hypernet_base.append(
            RHN_ACTV1Block(self.config, attn=True, attn_type="perceiver", attn_params={
                "input_size": self.config.hypernet_hidden_size,
                "kv_size": self.config.hidden_size,
                "heads": self.config.perceiver_heads,
                "seq_len": self.num_queries
            })
        )
        for _i in range(self.config.H_layers):
            self.hypernet_base.append(
                RHN_ACTV1Block(self.config,
                               rmsnorm=self.config.hypernet_rmsnorm,
                               attn=self.config.hypernet_attn,
                               attn_type="self",
                               attn_params={
                                   "input_size": self.config.hypernet_hidden_size,
                                   "kv_size": self.config.hypernet_hidden_size,
                                   "heads": self.config.perceiver_heads,
                                   "seq_len": self.num_queries
                               })
            )

        self.output_dim = self._output_dim(layer_specs)
        self.output_head = CastedLinear(self.config.hypernet_hidden_size, self.output_dim, bias=False)

    def forward(self, activations: torch.Tensor, **seq_info) -> Tuple[dict, torch.Tensor, dict]:
        batch_size = activations.shape[0]

        hidden_states = None
        for i, layer in enumerate(self.hypernet_base):
            if i == 0:
                hidden_states = layer(hidden_states=self.input_queries, kv=activations, **seq_info)
            else:
                hidden_states = layer(hidden_states=hidden_states, kv=None, **seq_info)

        outputs = self.output_head(hidden_states)
        output_head_l2 = outputs.flatten(1).norm(dim=1).mean()

        outputs = self._expand_output(outputs)
        expansion_l2 = outputs.flatten(1).norm(dim=1).mean()

        step_l2 = outputs.view(batch_size, -1).pow(2).sum(dim=1)

        outputs_by_layer = {}
        gen_norm_sq = torch.zeros(batch_size, device=outputs.device)

        for i, (layer_name, layer_info) in enumerate(self.config_per_layer.items()):
            shape = layer_info["shape"]
            layer_params = outputs[:, i, :]

            output_index = 0
            size_a = shape[0] * self.config.hypernet_rank
            outputs_a = layer_params[:, output_index: output_index + size_a]
            gen_norm_sq += outputs_a.pow(2).sum(dim=1)
            outputs_a = outputs_a.view(batch_size, shape[0], self.config.hypernet_rank)
            output_index += size_a

            if layer_info["type"] == "matrix":
                size_b = shape[1] * self.config.hypernet_rank
                outputs_b = layer_params[:, output_index: output_index + size_b]
                gen_norm_sq += outputs_b.pow(2).sum(dim=1)
                outputs_b = outputs_b.view(batch_size, self.config.hypernet_rank, shape[1])
                outputs_by_layer[layer_name] = (outputs_a, outputs_b)
            else:
                outputs_by_layer[layer_name] = outputs_a

        gen_norm_l2 = gen_norm_sq.sqrt().mean()

        hyper_metrics = {
            "output_head_l2": output_head_l2.detach(),
            "expansion_l2": expansion_l2.detach(),
            "gen_norm_l2": gen_norm_l2.detach()
        }

        return outputs_by_layer, step_l2, hyper_metrics

    def _is_vector_like(self, shape: list) -> bool:
        if len(shape) < 2: return True
        num_large_dims = 0
        for dim in shape:
            if dim >= 1: num_large_dims += 1
        return num_large_dims < 2

    def _output_dim(self, layer_specs: dict) -> int:
        max_params = 0
        for name, shape in layer_specs:
            if self._is_vector_like(shape):
                params = shape[0] * self.config.hypernet_rank
            else:
                params = (shape[0] + shape[1]) * self.config.hypernet_rank
            if params > max_params:
                max_params = params

        self.kron_dim = int(math.ceil(max_params ** 0.25))
        elements_per_matrix = self.kron_dim ** 2
        total_elements_needed = self.num_layers * 2 * elements_per_matrix
        output_dim = int(math.ceil(total_elements_needed / self.num_queries))
        return output_dim

    def _expand_output(self, outputs: torch.Tensor) -> torch.Tensor:
        batch_size = outputs.shape[0]
        outputs = outputs.flatten(start_dim=1)

        needed_elements_per_matrix = self.kron_dim ** 2
        needed_elements_per_layer = 2 * needed_elements_per_matrix
        total_needed = self.num_layers * needed_elements_per_layer

        outputs = outputs[:, :total_needed]

        # Reshape to isolate each factor matrix
        outputs = outputs.reshape(batch_size, self.num_layers, 2, needed_elements_per_matrix)

        outputs_a = outputs[:, :, 0, :]
        outputs_b = outputs[:, :, 1, :]

        # Normalize the ENTIRE factor matrix globally to preserve 2D internal geometry
        outputs_a = rms_norm(outputs_a, variance_epsilon=self.config.rms_norm_eps)
        outputs_b = rms_norm(outputs_b, variance_epsilon=self.config.rms_norm_eps)

        outputs_a = outputs_a.reshape(batch_size, self.num_layers, self.kron_dim, self.kron_dim)
        outputs_b = outputs_b.reshape(batch_size, self.num_layers, self.kron_dim, self.kron_dim)

        expanded_outputs = torch.einsum('blij,blkm->blikjm', outputs_a, outputs_b)
        outputs = expanded_outputs.flatten(start_dim=2, end_dim=-1)

        return outputs


class RHN_ACTV1_Inner(nn.Module):
    def __init__(self, config: RHN_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hypernet_hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        self.L_level = torch.nn.ModuleList([RHN_ACTV1Block_Dynamic(self.config, attn=True) for _i in range(self.config.L_layers)])

        self.layer_specs = []
        for name, param in self.named_parameters():
            name_tag = name.split(".")[0]
            if name_tag != "L_level":
                continue
            if "norm" in name.lower() or "scale" in name.lower():
                continue
            self.layer_specs.append((name, param.shape))

        self.hypernet = RHN_Hypernetwork(self.config, self.layer_specs)

        # Parallel Stream Temporal Iteration Mixer (mHC-lite)
        self.mhc_mixer = MHCLiteMixer(num_streams=self.config.mhc_window_size)

        # Unified Initial State (Now dynamically tracked as K streams)
        self.Z_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.mhc_window_size, self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        self.mhc_pre = nn.Parameter(torch.zeros(self.config.mhc_window_size, dtype=self.forward_dtype))
        self.mhc_pre.data[0] = 1.0  # Initialize to extract stream 0

        self.mhc_post = nn.Parameter(torch.zeros(self.config.mhc_window_size, dtype=self.forward_dtype))
        self.mhc_post.data[0] = 1.0  # Initialize to inject back to stream 0

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

        # Macro and Readout Normalization
        self.macro_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, elementwise_affine=True).to(dtype=self.forward_dtype)
        self.readout_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, elementwise_affine=True).to(dtype=self.forward_dtype)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return RHN_ACTV1InnerCarry(
            z=torch.empty(batch_size, self.config.mhc_window_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            inner_steps=torch.zeros(batch_size, dtype=torch.int32, device="cuda" if torch.cuda.is_available() else "cpu"),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: RHN_ACTV1InnerCarry):
        reset_mask = reset_flag.view(-1, 1, 1, 1)
        return RHN_ACTV1InnerCarry(
            z=torch.where(reset_mask, self.Z_init.view(1, self.config.mhc_window_size, 1, self.config.hidden_size), carry.z),
            inner_steps=torch.where(reset_flag, 0, carry.inner_steps)
        )

    def forward(self, carry: RHN_ACTV1InnerCarry, batch: Dict[str, torch.Tensor], log_deep_metrics: bool = False, **kwargs) -> Tuple[RHN_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, dict]:
        seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)
        inner_steps = carry.inner_steps

        if inner_steps[0] == 0:
            input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
            # Push the kickstart embeddings symmetrically into all parallel streams
            z_macro = carry.z + input_embeddings.unsqueeze(1)
        else:
            z_macro = carry.z

        # ----------------------------------------------------
        # 1. Macro Norm & Hypernetwork Path
        # ----------------------------------------------------
        # Standard RMSNorm naturally handles trailing dimensions regardless of [B, S, L, D] shapes
        z_normed = self.macro_norm(z_macro)

        total_metrics = {
            "telemetry/act_sparsity": torch.tensor(0.0, device=z_macro.device),
            "telemetry/act_saturation": torch.tensor(0.0, device=z_macro.device),
            "telemetry/state_drift": torch.tensor(0.0, device=z_macro.device),
            "telemetry/gen_l2_norm": torch.tensor(0.0, device=z_macro.device)
        }

        if log_deep_metrics:
            total_metrics["telemetry/gen_svd_ratio"] = torch.tensor(0.0, device=z_macro.device)
            total_metrics["telemetry/gen_base_l2_ratio"] = torch.tensor(0.0, device=z_macro.device)
            total_metrics["telemetry/output_head_l2"] = torch.tensor(0.0, device=z_macro.device)
            total_metrics["telemetry/expansion_l2"] = torch.tensor(0.0, device=z_macro.device)
            total_metrics["telemetry/gen_norm_l2"] = torch.tensor(0.0, device=z_macro.device)

        metric_calls = 0
        total_l2 = torch.zeros(z_macro.shape[0], device=z_macro.device, dtype=z_macro.dtype)

        def track_metrics(prev_state, new_state, step_metrics):
            nonlocal metric_calls
            total_metrics["telemetry/act_sparsity"] += step_metrics["sparsity"]
            total_metrics["telemetry/act_saturation"] += step_metrics["saturation"]
            total_metrics["telemetry/gen_l2_norm"] += step_metrics["gen_norm"]
            total_metrics["telemetry/state_drift"] += F.cosine_similarity(prev_state, new_state, dim=-1).mean()

            # Low-Frequency (Every 100 Steps)
            if log_deep_metrics:
                total_metrics["telemetry/gen_svd_ratio"] += step_metrics["svd_ratio"]
                total_metrics["telemetry/gen_base_l2_ratio"] += step_metrics["gen_base_l2_ratio"]
                total_metrics["telemetry/output_head_l2"] += step_metrics["output_head_l2"]
                total_metrics["telemetry/expansion_l2"] += step_metrics["expansion_l2"]
                total_metrics["telemetry/gen_norm_l2"] += step_metrics["gen_norm_l2"]
            metric_calls += 1

        # ----------------------------------------------------
        # 2. Micro-Cycles (Temporal mHC-lite constraints)
        # ----------------------------------------------------
        z_local = z_normed

        # H_cycles-1 without grad (Truncated BPTT)
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_local, prev_z_0, new_z_0, step_l2, step_metrics = self._dynamic_forward(
                        z_local=z_local,
                        log_deep_metrics=log_deep_metrics,
                        **seq_info
                    )

                    total_l2 += step_l2
                    track_metrics(prev_z_0, new_z_0, step_metrics)
                    inner_steps += 1

        # Final gradient-tracked cycle
        for _L_step in range(self.config.L_cycles):
            z_local, prev_z_0, new_z_0, step_l2, step_metrics = self._dynamic_forward(
                z_local=z_local,
                log_deep_metrics=log_deep_metrics,
                **seq_info
            )

            total_l2 += step_l2
            track_metrics(prev_z_0, new_z_0, step_metrics)
            inner_steps += 1

        # ----------------------------------------------------
        # 3. The Hierarchical Ratchet (Additive Macro Step)
        # ----------------------------------------------------
        z_macro = z_macro + z_local

        if metric_calls > 0:
            for k in total_metrics:
                total_metrics[k] /= metric_calls

        avg_l2 = total_l2 / metric_calls if metric_calls > 0 else total_l2

        # ----------------------------------------------------
        # 4. Readout
        # ----------------------------------------------------
        new_carry = RHN_ACTV1InnerCarry(z=z_macro.detach(), inner_steps=inner_steps)

        # Linear readouts read purely from Stream 0 context memory
        z_readout = self.readout_norm(z_macro[:, 0])
        output = self.lm_head(z_readout)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_readout[:, 0].detach()).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), avg_l2, total_metrics

    def _dynamic_forward(self, z_local: torch.Tensor, log_deep_metrics: bool = False, **seq_info) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # 1. mHC Res-Mapping: Safely mix parallel streams for this iteration
        z_local_mixed = self.mhc_mixer(z_local)

        # 2. mHC Pre-Mapping: Extract the active stream 0
        z_in = torch.einsum('k, bksd -> bsd', self.mhc_pre.to(self.forward_dtype), z_local_mixed)

        dynamic_weights, step_l2, hyper_metrics = self.hypernet(z_in, **seq_info)

        step_metrics = {}
        with torch.no_grad():
            step_metrics["sparsity"] = (z_in.abs() < 1e-3).float().mean()
            step_metrics["saturation"] = (z_in.abs() > 5.0).float().mean()

            if log_deep_metrics:
                step_metrics.update(hyper_metrics)

            gen_norm = 0.0
            svd_ratio = 0.0
            gen_base_l2_ratio = 0.0
            count = 0

            for k, v in dynamic_weights.items():
                base_param = self.get_parameter(k)
                if isinstance(v, tuple) and len(v) == 2:
                    A, B = v
                    gen_norm += (A[0].norm() + B[0].norm())

                    if log_deep_metrics:
                        delta_W = torch.matmul(A[0], B[0]).float()
                        S = torch.linalg.svdvals(delta_W)
                        svd_ratio += (S[0] / (S.sum() + 1e-6))
                        gen_base_l2_ratio += delta_W.norm() / (base_param.norm() + 1e-8)
                    count += 1
                else:
                    A = v
                    gen_norm += A[0].norm()

                    if log_deep_metrics:
                        delta_W = A[0].float()
                        gen_base_l2_ratio += delta_W.norm() / (base_param.norm() + 1e-8)
                    count += 1

            step_metrics["gen_norm"] = (gen_norm / count) if count > 0 else torch.tensor(0.0, device=z_local.device)

            if log_deep_metrics:
                step_metrics["svd_ratio"] = (svd_ratio / count) if count > 0 else torch.tensor(0.0, device=z_local.device)
                step_metrics["gen_base_l2_ratio"] = (gen_base_l2_ratio / count) if count > 0 else torch.tensor(0.0, device=z_local.device)

        # 3. Standard Spatial Forward Pass (No mHC operations inside this loop)
        h_dyn = z_in
        for i, layer in enumerate(self.L_level):
            layer.set_dynamic_adapter(dynamic_weights, layer_idx=i)
            h_dyn = layer(hidden_states=h_dyn, **seq_info)

        # Delta over the entire base model spatial depth for this iteration
        y_t = h_dyn - z_in

        # 4. mHC Post-Mapping: Inject the resulting delta back into the mixed manifold streams
        z_local_out = z_local_mixed + torch.einsum('k, bsd -> bksd', self.mhc_post.to(self.forward_dtype), y_t)

        return z_local_out, z_in, h_dyn, step_l2, step_metrics


class RHN_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = RHN_ACTV1Config(**config_dict)
        self.inner = RHN_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return RHN_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: RHN_ACTV1Carry, batch: Dict[str, torch.Tensor], log_deep_metrics: bool = False) -> Tuple[
        RHN_ACTV1Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v
                            in carry.current_data.items()}

        new_inner_carry, logits, (q_halt_logits, q_continue_logits), hypernet_l2, deep_metrics = self.inner(
            new_inner_carry, new_current_data, log_deep_metrics)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "hypernet_l2": hypernet_l2,
        }
        outputs.update(deep_metrics)

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return RHN_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
