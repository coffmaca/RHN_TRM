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
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear, DynamicSwiGLU, DynamicAttention
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class RHN_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


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
    hypernet_hidden_depth: int
    hypernet_rank: int
    layer_emb_dim: int

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
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
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
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                                     variance_epsilon=self.norm_eps)
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
        self.total_chunks = 0
        chunks_per_layer = []
        alt_dims = []
        for layer_spec in layer_specs:
            num_chunks, alt_dim = self._calculate_chunks(layer_spec)
            chunks_per_layer.append(num_chunks)
            alt_dims += alt_dim
            self.total_chunks += num_chunks

        alt_dims = list(set(alt_dims))

        self.embed_scale = math.sqrt(self.config.hypernet_hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.layer_embeddings = CastedEmbedding(self.total_chunks, self.config.layer_emb_dim, init_std=embed_init_std,
                                            cast_to=self.forward_dtype)

        self.config_per_layer = []
        self.input_size = self.config.hidden_size * self.config.L_layers
        self.chunk_ids = {"vector": [], "matrix": []}
        for dim in alt_dims:
            self.chunk_ids[f"alt_dim_{dim}"] = []
        chunk_idx = 0
        for i, (name, shape) in enumerate(self.layer_specs): # TODO - Consider combining logic below.  "Type" is just structure.  "Vector" also currently assumes dim size, which is wrong.
            rows, cols = shape

            # Vector-like parameters generated directly, without low rank matrices.
            if self._is_vector_like(shape):
                size = 1
                for dim in shape:
                    size *= dim

                ids = torch.arange(chunks_per_layer[i]) + chunk_idx
                chunk_idx += chunks_per_layer[i]

                self.config_per_layer.append({
                    "name": name,
                    "type": "vector",
                    "shape": shape,
                    "size": size,
                    "chunks": chunks_per_layer[i],
                    "type_a": "vector",
                    "shape_a": shape,
                    "chunks_a": chunks_per_layer[i],
                    "size_a": None,
                    "shape_b": None,
                    "size_b": None
                })

                self.chunk_ids["vector"] += ids.tolist()

            # Matrix-like parameters generated via low rank matrices
            else:
                chunks_a = int(rows / self.config.hypernet_hidden_size) if rows % self.config.hypernet_hidden_size == 0 else 1
                chunks_b = int(cols / self.config.hypernet_hidden_size) if cols % self.config.hypernet_hidden_size == 0 else 1

                alt_dim = rows if chunks_a == 1 else cols

                ids = torch.arange(chunks_per_layer[i]) + chunk_idx
                chunk_idx += chunks_per_layer[i]
                ids_a = ids[:chunks_a]
                ids_b = ids[chunks_a:]

                self.config_per_layer.append({
                    "name": name,
                    "type": "matrix",
                    "shape": shape,
                    "size": None,
                    "chunks": chunks_per_layer[i],
                    "type_a": "matrix" if chunks_a != 1 else f"alt_dim_{alt_dim}",
                    "shape_a": (rows, self.config.hypernet_rank),
                    "chunks_a": chunks_a,
                    "type_b": "matrix" if chunks_b != 1 else f"alt_dim_{alt_dim}",
                    "shape_b": (self.config.hypernet_rank, cols),
                    "chunks_b": chunks_b,
                })

                id_label_a = "matrix" if chunks_a != 1 else f"alt_dim_{alt_dim}"
                id_label_b = "matrix" if chunks_b != 1 else f"alt_dim_{alt_dim}"

                self.chunk_ids[id_label_a] += ids_a.tolist()
                self.chunk_ids[id_label_b] += ids_b.tolist()

        # TODO - Consider alternative initialization to 0's.  Classes below have built-in LeCun Normal initialization.
        module_list = nn.ModuleList(
            [CastedLinear(self.input_size + self.config.layer_emb_dim,
                          self.config.hypernet_hidden_size,
                          bias=False)] + \
            [nn.SiLU()] + \
            [SwiGLU(self.config.hypernet_hidden_size,
                    self.config.expansion) for _ in range(self.config.hypernet_hidden_depth)]
        )
        self.hypernet_base = nn.Sequential(*module_list)

        # Generate parameters in chunks proportional to hidden dim, as all main network parameter dims multiples of hidden dim.
        self.output_heads = nn.ModuleDict({
            "matrix": CastedLinear(self.config.hypernet_hidden_size,
                                   self.config.hypernet_rank * self.config.hypernet_hidden_size, # Generates LR matrices
                                   bias=False),
            "vector": CastedLinear(self.config.hypernet_hidden_size,
                                   self.config.hypernet_hidden_size,  # Directly generates parameters
                                   bias=False)
        })
        self.output_heads.update(
            nn.ModuleDict({
                f"alt_dim_{dim}" : CastedLinear(self.config.hypernet_hidden_size,
                                                self.config.hypernet_rank * dim, # Generates LR matrices
                                                bias=False)
                          for dim in alt_dims
            })
        )

    def forward(self, activations: torch.Tensor) -> dict:
        batch_size, seq_len, _ = activations.shape
        device = activations.device

        activations_expanded = activations.unsqueeze(2).expand(-1, -1, self.total_chunks,
                                                               -1)  # TODO - Consider expanding and adding embeddings only after trunk

        ids = torch.arange(self.total_chunks, device=device)
        embeddings = self.layer_embeddings(ids)
        embeddings = embeddings.view(1, 1, self.total_chunks, -1).expand(batch_size, seq_len, -1, -1)

        inputs = torch.cat([activations_expanded, embeddings], dim=3)

        outputs_raw = self.hypernet_base(inputs)

        outputs_grouped = {}
        outputs = {}

        for layer_type in self.chunk_ids:
            indices = torch.tensor(self.chunk_ids[layer_type], device=device, dtype=torch.int)
            features = outputs_raw.index_select(2, indices)
            outputs_grouped[layer_type] = {
                "data" : self.output_heads[layer_type](features),
                "index" : 0
            }

        for layer in self.config_per_layer:
            type_a = layer["type_a"]
            chunks_a = layer["chunks_a"]
            index_a = outputs_grouped[type_a]["index"]
            outputs_a = outputs_grouped[type_a]["data"][:, :, index_a:index_a + chunks_a]
            outputs_a = outputs_a.view(batch_size, seq_len, *layer["shape_a"])

            outputs_grouped[type_a]["index"] += chunks_a
            if outputs_grouped[type_a]["index"] == outputs_grouped[type_a]["data"].shape[2]:
                outputs_grouped[type_a]["index"] = 0

            if layer["type"] == "matrix":
                type_b = layer["type_b"]
                chunks_b = layer["chunks_b"]
                index_b = outputs_grouped[type_b]["index"]
                outputs_b = outputs_grouped[type_b]["data"][:, :, index_b:index_b + chunks_b]
                outputs_b = outputs_b.view(batch_size, seq_len, *layer["shape_b"])

                outputs_grouped[type_b]["index"] += chunks_b
                if outputs_grouped[type_b]["index"] == outputs_grouped[type_b]["data"].shape[2]:
                    outputs_grouped[type_b]["index"] = 0

            if layer["type"] == "vector":
                outputs[layer["name"]] = outputs_a
            else:
                outputs[layer["name"]] = (outputs_a, outputs_b)

        return outputs

    def _calculate_chunks(self, layer_spec):
        name, shape = layer_spec
        is_vector_like = self._is_vector_like(shape)
        alt_dim = []

        num_chunks = 0
        if is_vector_like:
            for dim in shape:
                if dim % self.config.hypernet_hidden_size == 0:
                    num_chunks += dim / self.config.hypernet_hidden_size
                elif dim != 1:
                    alt_dim.append(dim)
                    num_chunks += 1
        else:
            dim_sum = 0
            for dim in shape:
                if dim % self.config.hypernet_hidden_size == 0:
                    dim_sum += dim
                else:
                    alt_dim.append(dim)
                    num_chunks += 1
            num_chunks += dim_sum / self.config.hypernet_hidden_size

        return int(num_chunks), alt_dim

    def _is_vector_like(self, shape):
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
        for name, param in self.named_parameters():
            name_tag = name.split(".")[0]
            if name_tag != "L_level":
                continue
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
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: RHN_ACTV1InnerCarry):
        return RHN_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: RHN_ACTV1InnerCarry, batch: Dict[str, torch.Tensor], **kwargs) -> Tuple[RHN_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        hidden_states = hidden_states_prior = z_L + z_H
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    hidden_states = hidden_states + input_embeddings
                    hidden_states, hidden_states_prior = self._dynamic_forward(hidden_states=hidden_states,
                                                                               hidden_states_prior=hidden_states_prior,
                                                                               input_embeddings=input_embeddings,
                                                                               **seq_info)
                z_L = hidden_states
                hidden_states = hidden_states_prior = z_H + z_L
                hidden_states, hidden_states_prior = self._dynamic_forward(hidden_states=hidden_states,
                                                                           hidden_states_prior=hidden_states_prior,
                                                                           input_embeddings=None,
                                                                           **seq_info)
                z_H = hidden_states
        # 1 with grad

        # torch.cuda.memory._record_memory_history(
        #     max_entries=100000
        # )

        for _L_step in range(self.config.L_cycles):
            hidden_states = z_L + z_H + input_embeddings
            hidden_states, hidden_states_prior = self._dynamic_forward(hidden_states=hidden_states,
                                                                       hidden_states_prior=hidden_states_prior,
                                                                       input_embeddings=input_embeddings,
                                                                       **seq_info)

        # try:
        #     torch.cuda.memory._dump_snapshot(f"mem_prof1.pickle")
        # except Exception as e:
        #     print(f"Failed to capture memory snapshot {e}")
        #
        # # Stop recording memory snapshot history.
        # torch.cuda.memory._record_memory_history(enabled=None)

        z_L = hidden_states
        hidden_states = z_H + z_L
        hidden_states, hidden_states_prior = self._dynamic_forward(hidden_states=hidden_states,
                                                                   hidden_states_prior=hidden_states_prior,
                                                                   input_embeddings=None,
                                                                   **seq_info)
        z_H = hidden_states

        # LM Outputs
        new_carry = RHN_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])

    def _dynamic_forward(self, hidden_states, hidden_states_prior, input_embeddings=None, **seq_info):
        hidden_states = hidden_states + input_embeddings if input_embeddings is not None else hidden_states
        activations = torch.tensor([], dtype=hidden_states.dtype, device=hidden_states.device)
        # Base model output
        for layer in self.L_level:
            layer.clear_dynamic_adapter()
            hidden_states = layer(hidden_states=hidden_states, **seq_info)
            activations = torch.cat((activations, hidden_states.detach()),
                                    dim=2)  # TODO - Determine whether detaching is preferable here.
        base_out = hidden_states.clone()

        # Dynamic weight output
        hidden_states = hidden_states_prior + input_embeddings if input_embeddings is not None else hidden_states_prior
        dynamic_weights = self.hypernet(activations)
        for i, layer in enumerate(self.L_level):
            layer_weights = [dynamic_weights[layer_name] for layer_name in dynamic_weights if
                             f"L_level.{i}" in layer_name]
            layer.set_dynamic_adapter(*layer_weights)
            hidden_states = layer(hidden_states=hidden_states, **seq_info)
        dynamic_out = hidden_states
        hidden_states = base_out + dynamic_out
        hidden_states_prior = hidden_states.clone()

        return hidden_states, hidden_states_prior



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
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: RHN_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[RHN_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

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
