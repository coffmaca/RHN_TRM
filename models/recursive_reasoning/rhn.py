from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
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
    z_H: torch.Tensor
    z_L: torch.Tensor
    prev_activations: Optional[torch.Tensor] = None


@dataclass
class RHN_ACTV1Carry:
    inner_carry: RHN_ACTV1InnerCarry

    inference_carry: Optional[Dict[str, torch.Tensor]]
    inference_active_indices: Optional[torch.Tensor]

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
    hypernet_hidden_depth: int
    hypernet_rank: int
    layer_emb_dim: int
    hypernet_relative_scale: float
    kron_dims: int
    kron_dims_mult: bool
    perceiver_heads: int
    hypernet_relative_scale: int
    hypernet_l2_lambda: float = 1e-4
    hypernet_kl_lambda: float = 1e-4

class RHN_ACTV1Block(nn.Module):
    def __init__(self, config: RHN_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L # TODO - Confirm reasoning for these values
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            att_out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + att_out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            att_out = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
            hidden_states = rms_norm(hidden_states + att_out, variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class RHN_ACTV1Block_Dynamic(nn.Module):
    def __init__(self, config: RHN_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(
                        self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = DynamicSwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = DynamicAttention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = DynamicSwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def set_dynamic_adapter(self, attn_1, attn_2, up, down):
        A_up, B_up = up
        A_down, B_down = down
        self.mlp.set_dynamic_adapter(A_up, B_up, A_down, B_down)

        A_attn_1, B_attn_1 = attn_1
        A_attn_2, B_attn_2 = attn_2

        if self.config.mlp_t:
            self.mlp_t.set_dynamic_adapter(A_attn_1, B_attn_1, A_attn_2, B_attn_2)
        else:
            self.self_attn.set_dynamic_adapter(A_attn_1, B_attn_1, A_attn_2, B_attn_2)


    def clear_dynamic_adapter(self):
        self.mlp.clear_dynamic_adapter()
        if self.config.mlp_t:
            self.mlp_t.clear_dynamic_adapter()
        else:
            self.self_attn.clear_dynamic_adapter()

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            att_out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + att_out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            # Self Attention
            att_out = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
            hidden_states = rms_norm(hidden_states + att_out, variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
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

        self.input_size = self.config.hidden_size
        self.num_layers = len(self.layer_specs)

        if self.config.kron_dims_mult:
            self.num_queries = self.num_layers * self.config.kron_dims
        else:
            self.num_queries = self.config.kron_dims

        self.input_queries = nn.Parameter(
            trunc_normal_init_(
                torch.empty((1, self.num_queries, self.input_size), dtype=self.forward_dtype),
                std=1.0 / math.sqrt(self.input_size),
            )
        )

        # TODO - Consider alternative initialization to 0's.  Classes below have built-in LeCun Normal initialization.
        module_list = nn.ModuleList(
            [CastedLinear(self.input_size,
                          self.config.hypernet_hidden_size,
                          bias=False)] + \
            [nn.SiLU()]
        )
        for _ in range(self.config.hypernet_hidden_depth):
            module_list.append(SwiGLU(self.config.hypernet_hidden_size, self.config.expansion))
            module_list.append(torch.nn.RMSNorm(self.config.hypernet_hidden_size,
                                                eps=self.config.rms_norm_eps,
                                                dtype=self.forward_dtype))

        self.hypernet_base = nn.Sequential(*module_list)

        self.output_dim = self._output_dim(layer_specs)
        self.output_head = CastedLinear(self.config.hypernet_hidden_size,
                                         self.output_dim,
                                         bias=False)

    def forward(self, activations: torch.Tensor) -> Tuple[dict, torch.Tensor]:
        batch_size, seq_len, _ = activations.shape

        inputs = self._attention(activations)

        inputs = rms_norm(inputs, variance_epsilon=self.config.rms_norm_eps)

        outputs = self.hypernet_base(inputs)
        outputs = self.output_head(outputs)
        outputs = rms_norm(outputs.flatten(start_dim=1), variance_epsilon=self.config.rms_norm_eps).view(outputs.shape)
        outputs_list = self._expand_output(outputs)

        flat_expanded = torch.cat(outputs_list, dim=1)
        step_l2 = flat_expanded.pow(2).sum(dim=1)

        outputs_by_layer = {}

        for i, (layer_name, layer_info) in enumerate(self.config_per_layer.items()):
            shape = layer_info["shape"]

            # Retrieve the specific expanded tensor for this layer
            layer_params = outputs_list[i]

            output_index = 0

            size_a = shape[0] * self.config.hypernet_rank
            outputs_a = layer_params[:, output_index: output_index + size_a]
            # outputs_a = rms_norm(outputs_a, variance_epsilon=self.config.rms_norm_eps)
            outputs_a = outputs_a.view(batch_size, shape[0], self.config.hypernet_rank)
            output_index += size_a

            if layer_info["type"] == "matrix":
                size_b = shape[1] * self.config.hypernet_rank
                outputs_b = layer_params[:, output_index: output_index + size_b]
                # outputs_b = rms_norm(outputs_b, variance_epsilon=self.config.rms_norm_eps)
                outputs_b = outputs_b.view(batch_size, self.config.hypernet_rank, shape[1])

                output_index += size_b

                outputs_by_layer[layer_name] = (outputs_a, outputs_b)
            else:
                outputs_by_layer[layer_name] = outputs_a

        return outputs_by_layer, step_l2

    def _is_vector_like(self, shape:list) -> bool:
        if len(shape) < 2:
            return True

        num_large_dims = 0
        for dim in shape:
            if dim >= 1:
                num_large_dims += 1

        if num_large_dims >= 2:
            return False
        else:
            return True

    def get_low_rank_factors(self, base_param_total_low_rank: int) -> list:
        if base_param_total_low_rank < 4:
            return [2, 2]

        def get_factors_if_valid(d):
            factors = []
            n = d
            while n % 2 == 0:
                factors.append(2)
                n //= 2
            while n % 3 == 0:
                factors.append(3)
                n //= 3
            if n > 1:
                if 3 < n < 10:
                    factors.append(n)
                else:
                    return None
            return factors

        target_dim = math.ceil(math.sqrt(base_param_total_low_rank))

        while True:
            factors = get_factors_if_valid(target_dim)
            if factors is not None:
                return factors
            target_dim += 1

    def _output_dim(self, layer_specs) -> int:
        self.layer_kron_factors = {}
        total_elements_needed = 0

        for name, shape in layer_specs:
            if self._is_vector_like(shape):
                params = shape[0] * self.config.hypernet_rank
            else:
                params = (shape[0] + shape[1]) * self.config.hypernet_rank

            # Fetch and store factors tailored specifically to this layer
            factors = self.get_low_rank_factors(params)
            self.layer_kron_factors[name] = factors

            # Accumulate the elements needed dynamically
            layer_elements = sum(f ** 2 for f in factors)
            total_elements_needed += layer_elements

        output_dim = int(math.ceil(total_elements_needed / self.num_queries))

        return output_dim

    def _attention(self, inputs) -> torch.Tensor:
        B, S, D = inputs.shape
        H = self.config.perceiver_heads
        Q = self.num_queries
        head_dim = D // H

        q = self.input_queries.view(1, Q, H, head_dim).transpose(1, 2)
        k = inputs.view(B, S, H, head_dim).transpose(1, 2)
        v = inputs.view(B, S, H, head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = F.softmax(attn_logits, dim=-1)
        pooled_inputs = torch.matmul(attn_weights, v)
        pooled_inputs = pooled_inputs.transpose(1, 2).contiguous().view(B, Q, D)

        return pooled_inputs

    def _expand_output(self, outputs: torch.Tensor) -> list:
        batch_size = outputs.shape[0]
        outputs = outputs.flatten(start_dim=1)

        expanded_per_layer = []
        current_idx = 0

        for name, _ in self.layer_specs:
            factors = self.layer_kron_factors[name]
            expanded = None

            for f in factors:
                elements = f ** 2

                factor_tensor = outputs[:, current_idx : current_idx + elements]
                factor_tensor = rms_norm(factor_tensor, variance_epsilon=self.config.rms_norm_eps)
                factor_tensor = factor_tensor.view(batch_size, f, f)
                current_idx += elements

                if expanded is None:
                    expanded = factor_tensor
                else:
                    expanded = torch.einsum('bij,bkl->bikjl', expanded, factor_tensor)
                    H1, H2 = expanded.shape[1], expanded.shape[2]
                    W1, W2 = expanded.shape[3], expanded.shape[4]
                    expanded = expanded.reshape(batch_size, H1 * H2, W1 * W2)

            expanded_per_layer.append(expanded.flatten(start_dim=1))

        return expanded_per_layer


# class RHN_ACTV1ReasoningModule(nn.Module):
#     def __init__(self, layers: List[RHN_ACTV1Block_Dynamic]):
#         super().__init__()
#         self.layers = torch.nn.ModuleList(layers)
#
#     def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
#         hidden_states = hidden_states + input_injection
#         for layer in self.layers:
#             hidden_states = layer(hidden_states=hidden_states, **kwargs)
#         return hidden_states


class RHN_ACTV1_Inner(nn.Module):
    def __init__(self, config: RHN_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

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
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = torch.nn.ModuleList([RHN_ACTV1Block_Dynamic(self.config) for _i in range(self.config.L_layers)])

        # Hypernetwork
        self.layer_specs = []
        valid_proj_names = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]

        for name, param in self.named_parameters():
            if not name.startswith("L_level."):
                continue
            if not name.endswith(".weight"):
                continue
            if any(proj in name for proj in valid_proj_names):
                self.layer_specs.append((name, param.shape))

        self.hypernet = RHN_Hypernetwork(self.config, self.layer_specs)

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

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
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            prev_activations=None
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: RHN_ACTV1InnerCarry):
        new_prev = carry.prev_activations
        if new_prev is not None:
            new_prev = torch.where(reset_flag.view(-1, 1, 1), torch.zeros_like(new_prev), new_prev)

        return RHN_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
            prev_activations=new_prev
        )

    def forward(self, carry: RHN_ACTV1InnerCarry, batch: Dict[str, torch.Tensor],
                log_deep_metrics: bool = False, **kwargs) -> Tuple[
        RHN_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, dict
    ]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        z_H, z_L = carry.z_H, carry.z_L
        prev_activations = carry.prev_activations

        total_metrics = {
            "telemetry/act_sparsity": torch.tensor(0.0, device=z_H.device),
            "telemetry/act_saturation": torch.tensor(0.0, device=z_H.device),
            "telemetry/gen_l2_norm": torch.tensor(0.0, device=z_H.device),
            "telemetry/state_drift": torch.tensor(0.0, device=z_H.device)
        }

        if log_deep_metrics:
            total_metrics["telemetry/gen_svd_ratio"] = torch.tensor(0.0, device=z_H.device)
            total_metrics["telemetry/gen_base_l2_ratio"] = torch.tensor(0.0, device=z_H.device)

        metric_calls = 0

        def track_metrics(prev_state, new_state, step_metrics):
            nonlocal metric_calls
            total_metrics["telemetry/act_sparsity"] += step_metrics["sparsity"]
            total_metrics["telemetry/act_saturation"] += step_metrics["saturation"]
            total_metrics["telemetry/gen_l2_norm"] += step_metrics["gen_norm"]
            total_metrics["telemetry/state_drift"] += F.cosine_similarity(prev_state, new_state, dim=-1).mean()
            if log_deep_metrics:
                total_metrics["telemetry/gen_svd_ratio"] += step_metrics["svd_ratio"]
                total_metrics["telemetry/gen_base_l2_ratio"] += step_metrics["gen_base_l2_ratio"]
            metric_calls += 1

        total_l2 = torch.zeros(z_L.shape[0], device=z_L.device, dtype=z_L.dtype)

        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    prev_z_L = z_L
                    z_L, prev_activations, _, step_m = self._dynamic_forward(
                        z_L=z_L, z_H=z_H, prev_activations=prev_activations,
                        input_embeddings=input_embeddings, log_deep_metrics=log_deep_metrics, **seq_info
                    )
                    track_metrics(prev_z_L, z_L, step_m)
                prev_z_H = z_H
                z_H, prev_activations, _, step_m = self._dynamic_forward(
                    z_L=z_L, z_H=z_H, prev_activations=prev_activations,
                    input_embeddings=None, log_deep_metrics=log_deep_metrics, **seq_info
                )
                track_metrics(prev_z_H, z_H, step_m)

        for _L_step in range(self.config.L_cycles):
            prev_z_L = z_L
            z_L, prev_activations, step_l2, step_m = self._dynamic_forward(
                z_L=z_L, z_H=z_H, prev_activations=prev_activations,
                input_embeddings=input_embeddings, log_deep_metrics=log_deep_metrics, **seq_info
            )
            track_metrics(prev_z_L, z_L, step_m)

        prev_z_H = z_H
        z_H, prev_activations, step_l2, step_m = self._dynamic_forward(
            z_L=z_L, z_H=z_H, prev_activations=prev_activations,
            input_embeddings=None, log_deep_metrics=log_deep_metrics, **seq_info
        )

        total_l2 += step_l2
        avg_l2 = total_l2 / (self.config.L_cycles + 1)
        track_metrics(prev_z_H, z_H, step_m)

        if metric_calls > 0:
            for k in total_metrics:
                total_metrics[k] /= metric_calls

        prev_activations_detached = prev_activations.detach() if prev_activations is not None else None
        new_carry = RHN_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach(), prev_activations=prev_activations_detached)

        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), avg_l2, total_metrics

    def _dynamic_forward(self, z_L, z_H, prev_activations, input_embeddings=None, log_deep_metrics=False, **seq_info) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict
    ]:
        h_input = z_L + z_H + input_embeddings if input_embeddings is not None else z_L + z_H

        if prev_activations is None:
            activations_list = []
            h_temp = h_input
            for layer in self.L_level:
                layer.clear_dynamic_adapter()
                h_temp = layer(hidden_states=h_temp, **seq_info)
                activations_list.append(h_temp.detach())

            # Concatenate on sequence dimension (dim=1)
            prev_activations = torch.cat(activations_list, dim=1)

        dynamic_weights, step_l2 = self.hypernet(prev_activations)

        step_metrics = {}
        with torch.no_grad():
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

            step_metrics["gen_norm"] = (gen_norm / count) if count > 0 else torch.tensor(0.0, device=h_input.device)
            if log_deep_metrics:
                step_metrics["svd_ratio"] = (svd_ratio / count) if count > 0 else torch.tensor(0.0, device=h_input.device)
                step_metrics["gen_base_l2_ratio"] = (gen_base_l2_ratio / count) if count > 0 else torch.tensor(0.0, device=h_input.device)

        for i, layer in enumerate(self.L_level):
            layer_weights = [dynamic_weights[layer_name] for layer_name in dynamic_weights if f"L_level.{i}" in layer_name]
            layer.set_dynamic_adapter(*layer_weights)

        h_out = h_input
        new_activations_list = []
        for layer in self.L_level:
            h_out = layer(hidden_states=h_out, **seq_info)
            new_activations_list.append(h_out.detach())

        # 5. Extract current activations for the next iteration (Sequence dimension)
        new_activations = torch.cat(new_activations_list, dim=1)

        with torch.no_grad():
            step_metrics["sparsity"] = (h_out.abs() < 1e-3).float().mean()
            step_metrics["saturation"] = (h_out.abs() > 5.0).float().mean()

        return h_out, new_activations, step_l2, step_metrics



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
            inference_carry=None,
            inference_active_indices=None,
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: RHN_ACTV1Carry, batch: Dict[str, torch.Tensor], log_deep_metrics: bool = False) -> Tuple[RHN_ACTV1Carry, Dict[str, torch.Tensor]]:

        # If (i) training or (ii) first pass of inference (i.e., when all samples are default halted)
        if self.training or carry.steps.sum() == 0:
            # Update data, carry (removing halted sequences)
            new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

            new_steps = torch.where(carry.halted, 0, carry.steps)

            new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}
        else:
            new_inner_carry = carry.inner_carry
            new_steps = carry.steps[carry.inference_carry["active"]]
            new_current_data = carry.current_data

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), hypernet_l2, deep_metrics = self.inner(new_inner_carry, new_current_data, log_deep_metrics)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "hypernet_l2": hypernet_l2,
        }

        outputs.update(deep_metrics)

        # Initialize inference carries
        if not self.training:
            new_inference_carry = carry.inference_carry if carry.inference_carry is not None else {
                "halted": torch.empty_like(carry.halted).fill_(False),
                "active": torch.empty_like(carry.halted).fill_(True),
                "logits": torch.empty_like(logits),
                "steps": torch.empty_like(new_steps),
                "q_halt_logits": torch.empty_like(q_halt_logits),
                "q_continue_logits": torch.empty_like(q_continue_logits),
                "hypernet_l2": torch.empty_like(hypernet_l2)
            }
            new_inference_active_indices = (carry.inference_active_indices if carry.inference_active_indices is not None
                                            else torch.arange(logits.shape[0], device=logits.device))
        else:
            new_inference_carry = None
            new_inference_active_indices = None

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # If ACT is enabled
            if self.config.halt_max_steps > 1:

                # Dynamic Halt signal (Active for both Train and Eval)
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Training-only logic: Exploration and Target Q computation
                if self.training:
                    # Exploration
                    min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                    halted = halted & (new_steps >= min_halt_steps)

                    # Compute target Q correctly nested under training
                    if not self.config.no_ACT_continue:
                        _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                        outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

                # Freeze halted samples in separate tensor during inference
                if not self.training:
                    new_halted_indices = new_inference_active_indices[halted]
                    active = ~halted

                    # Save halted sample data to inference_carry and restore prior saved to outputs
                    new_inference_carry["halted"][new_halted_indices] = halted[halted]
                    new_inference_carry["active"][new_halted_indices] = ~halted[halted]
                    new_inference_carry["logits"][new_halted_indices] = logits[halted]
                    new_inference_carry["steps"][new_halted_indices] = new_steps[halted]
                    new_inference_carry["q_halt_logits"][new_halted_indices] = q_halt_logits[halted]
                    new_inference_carry["q_continue_logits"][new_halted_indices] = q_continue_logits[halted]
                    new_inference_carry["hypernet_l2"][new_halted_indices] = hypernet_l2[halted]

                    output_logits = new_inference_carry["logits"]
                    output_logits[new_inference_carry["active"]] = logits[active]
                    outputs["logits"] = output_logits

                    output_q_halt_logits = new_inference_carry["q_halt_logits"]
                    output_q_halt_logits[new_inference_carry["active"]] = q_halt_logits[active]
                    outputs["q_halt_logits"] = output_q_halt_logits

                    output_q_continue_logits = new_inference_carry["q_continue_logits"]
                    output_q_continue_logits[new_inference_carry["active"]] = q_continue_logits[active]
                    outputs["q_continue_logits"] = output_q_continue_logits

                    output_hypernet_l2 = new_inference_carry["hypernet_l2"]
                    output_hypernet_l2[new_inference_carry["active"]] = hypernet_l2[active]
                    outputs["hypernet_l2"] = output_hypernet_l2

                    # Filter halted samples from data
                    new_inner_carry.z_H = new_inner_carry.z_H[active]
                    new_inner_carry.z_L = new_inner_carry.z_L[active]
                    if new_inner_carry.prev_activations is not None:
                        new_inner_carry.prev_activations = new_inner_carry.prev_activations[active]

                    new_current_data["inputs"] = new_current_data["inputs"][active]
                    # new_current_data["labels"] = new_current_data["labels"][active] # Skip labels - Need full batch to test full batch accuracy
                    new_current_data["puzzle_identifiers"] = new_current_data["puzzle_identifiers"][active]

                    new_inference_active_indices = new_inference_active_indices[active]
                    halted = new_inference_carry["halted"]
                    new_steps_updated = new_inference_carry["steps"]
                    new_steps_updated[new_inference_carry["active"]] = new_steps[active]
                    new_steps = new_steps_updated

        return RHN_ACTV1Carry(new_inner_carry, new_inference_carry, new_inference_active_indices, new_steps, halted, new_current_data), outputs
