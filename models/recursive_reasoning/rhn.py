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
    cumulative_h_steps: torch.Tensor
    cumulative_l_h_steps: torch.Tensor


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
    puzzle_emb_len: int = 17 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

    hypernet_hidden_size: int
    hypernet_hidden_depth: int
    hypernet_rank: int
    layer_emb_dim: int
    hypernet_relative_scale: int

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
        self.config_per_layer = {}
        for name, shape in self.layer_specs:
            self.config_per_layer[name] = {
                "shape": shape,
                "type": "vector" if self._is_vector_like(shape) else "matrix",
            }

        self.embed_scale = math.sqrt(self.config.hypernet_hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # self.token_sum_query = CastedParameter((1, 1, self.config.hidden_size * self.config.L_layers), init_std=embed_init_std,
        #                                         cast_to=self.forward_dtype)
        self.token_sum_query = nn.Parameter(
            trunc_normal_init_(
                torch.empty((1, 1, self.config.hidden_size * self.config.L_layers), dtype=self.forward_dtype),
                std=embed_init_std
            )
        )

        self.input_size = self.config.hidden_size * self.config.L_layers

        # TODO - Consider alternative initialization to 0's.  Classes below have built-in LeCun Normal initialization.
        module_list = nn.ModuleList(
            [CastedLinear(self.input_size,
                          self.config.hypernet_hidden_size,
                          bias=False)] + \
            [nn.SiLU()] + \
            [SwiGLU(self.config.hypernet_hidden_size,
                    self.config.expansion) for _ in range(self.config.hypernet_hidden_depth)]
        )
        self.hypernet_base = nn.Sequential(*module_list)

        self.output_head = CastedLinear(self.config.hypernet_hidden_size,
                                         self._output_dim(layer_specs),
                                         bias=False)

    def forward(self, activations: torch.Tensor) -> dict:
        batch_size, seq_len, _ = activations.shape

        inputs = self._attention(activations)

        outputs = self.hypernet_base(inputs)
        outputs = self.output_head(outputs)
        outputs = self._expand_output(outputs)

        outputs_by_layer = {}
        output_index = 0
        for layer in self.config_per_layer:
            shape = self.config_per_layer[layer]["shape"]

            outputs_a = outputs[:, output_index : output_index + (shape[0] * self.config.hypernet_rank)]
            outputs_a = outputs_a.view(batch_size, shape[0], self.config.hypernet_rank)
            output_index += shape[0] * self.config.hypernet_rank

            if self.config_per_layer[layer]["type"] == "matrix":
                outputs_b = outputs[:, output_index : output_index + (shape[1] * self.config.hypernet_rank)]
                outputs_b = outputs_b.view(batch_size, self.config.hypernet_rank, shape[1])
                output_index += shape[1] * self.config.hypernet_rank

            if self.config_per_layer[layer]["type"] == "vector":
                outputs_by_layer[layer] = outputs_a
            else:
                outputs_by_layer[layer] = (outputs_a, outputs_b)

        return outputs_by_layer

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

    # TODO - Make this dynamic / tunable for larger base models, to ensure appropriate hypernetwork scaling.
    def _output_dim(self, layer_specs:dict) -> int:
        base_param_dim_sum = 0
        base_param_total = 0
        for layer in layer_specs:
            rows, cols = layer[1]
            base_param_dim_sum += rows + cols
            base_param_total += rows * cols

        base_param_total_low_rank = base_param_dim_sum * self.config.hypernet_rank

        self.output_dim = int(-(-base_param_total_low_rank**(1/4)//1)) # Square root twice (i.e., 1/4th root) and round up

        vals_to_generate = self.output_dim**2 * 2

        return vals_to_generate

    def _attention(self, inputs) -> torch.Tensor:
        token_sum_query = self.token_sum_query.transpose(1, 2)
        attn_logits = torch.matmul(inputs, token_sum_query)
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled_inputs = torch.matmul(inputs.transpose(1, 2), attn_weights)

        return pooled_inputs.squeeze(2)

    def _expand_output(self, outputs) -> torch.Tensor:
        used_outputs_a = outputs[...,:self.output_dim**2]
        used_outputs_a = used_outputs_a.unsqueeze(-1).view(-1, self.output_dim, self.output_dim)
        used_outputs_b = outputs[...,self.output_dim**2 : self.output_dim**2 * 2]
        used_outputs_b = used_outputs_b.unsqueeze(-1).view(-1, self.output_dim, self.output_dim)
        expanded_outputs = torch.einsum('bij,bkl->bikjl', used_outputs_a, used_outputs_b)
        outputs = expanded_outputs.flatten(start_dim=1,end_dim=-1)

        return outputs


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

        self.q_head_outer = CastedLinear(self.config.hidden_size, 2, bias=True)
        self.q_head_inner = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        self.outer_act_token = nn.Parameter(
            trunc_normal_init_(torch.empty(1, 1, self.config.hidden_size, dtype=self.forward_dtype), std=embed_init_std)
        )
        self.inner_act_token = nn.Parameter(
            trunc_normal_init_(torch.empty(1, 1, self.config.hidden_size, dtype=self.forward_dtype), std=embed_init_std)
        )

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        # Reasoning Layers
        self.L_level = nn.ModuleList([RHN_ACTV1Block_Dynamic(self.config) for _i in range(self.config.L_layers)])

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
            self.q_head_outer.weight.zero_()
            self.q_head_outer.bias.fill_(-5)
            self.q_head_inner.weight.zero_()
            self.q_head_inner.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))
        batch_size = input.shape[0]

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            puzzle_embedding = puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size)

            outer_token_exp = self.outer_act_token.expand(batch_size, -1, -1)
            inner_token_exp = self.inner_act_token.expand(batch_size, -1, -1)

            puzzle_embedding = torch.cat((
                outer_token_exp,
                inner_token_exp,
                puzzle_embedding[:, 2:, :]
            ), dim=1)

            embedding = torch.cat((puzzle_embedding, embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        device = self.H_init.device
        return RHN_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size,
                            dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size,
                            dtype=self.forward_dtype, device=device),
            cumulative_h_steps=torch.zeros(batch_size, dtype=torch.float32, device=device),
            cumulative_l_h_steps=torch.zeros(batch_size, dtype=torch.float32, device=device)
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: RHN_ACTV1InnerCarry):
        return RHN_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
            cumulative_h_steps=torch.where(reset_flag, 0.0, carry.cumulative_h_steps),
            cumulative_l_h_steps=torch.where(reset_flag, 0.0, carry.cumulative_l_h_steps)
        )

    # TODO - Consider updating dynamic forward approach to work more like typical LoRA
    # TODO - (i.e., by simply adding the dynamic weights to the base weights before running the forward pass).
    # TODO - This way, you only need to execute one forward pass instead of 2.
    def _dynamic_forward(self, z_L, z_H, dynamic_weights, input_embeddings=None, return_activations=False,
                                 **seq_info):
        # Base Pass
        h_base = z_L + z_H + input_embeddings if input_embeddings is not None else z_L + z_H
        activations = torch.tensor([], dtype=z_H.dtype, device=z_H.device) if return_activations else None
        for layer in self.L_level:
            layer.clear_dynamic_adapter()
            h_base = layer(hidden_states=h_base, **seq_info)
            if return_activations:
                activations = torch.cat((activations, h_base.detach()), dim=2)

        # Dynamic Pass
        h_dyn = z_L + z_H + input_embeddings if input_embeddings is not None else z_L + z_H
        for i, layer in enumerate(self.L_level):
            layer_weights = [dynamic_weights[name] for name in dynamic_weights if f"L_level.{i}" in name]
            layer.set_dynamic_adapter(*layer_weights)
            h_dyn = layer(hidden_states=h_dyn, **seq_info)

        return h_base + h_dyn, activations

    def forward(self, carry: RHN_ACTV1InnerCarry, batch: Dict[str, torch.Tensor], **kwargs) -> Tuple[
        RHN_ACTV1InnerCarry,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        batch_size = carry.z_H.shape[0]
        device = carry.z_H.device

        # Initialize completed sample tracking tensors
        final_z_H = torch.empty_like(carry.z_H)
        final_z_L = torch.empty_like(carry.z_L)
        final_q_outer = torch.zeros(batch_size, 2, dtype=torch.float32, device=device)

        active_idx = torch.arange(batch_size, device=device)
        h_steps = torch.zeros(batch_size, dtype=torch.float32, device=device)

        chunk_halted = torch.zeros(batch_size, dtype=torch.bool, device=device)

        max_total_inner_steps = self.config.H_cycles * self.config.L_cycles
        global_q_inner = torch.zeros((batch_size, max_total_inner_steps), dtype=torch.float32, device=device)
        global_inner_masks = torch.zeros((batch_size, max_total_inner_steps), dtype=torch.bool, device=device)
        inner_step_counter = 0

        z_H = carry.z_H.detach()
        z_L = carry.z_L.detach()

        activations = torch.tensor([], dtype=z_H.dtype, device=device)
        for layer in self.L_level:
            layer.clear_dynamic_adapter()
            temp_z_L = layer(hidden_states=z_L + z_H + input_embeddings, **seq_info)
            activations = torch.cat((activations, temp_z_L.detach()), dim=2)

        for h in range(self.config.H_cycles):
            if active_idx.numel() == 0:
                break

            # Given dynamic halting, using detach() implements truncated BPTT on a per-sample basis,
            # retaining the gradient tracking for only the last H_cycle iteration
            z_H = z_H.detach()
            z_L = z_L.detach()

            h_steps[active_idx] += 1.0

            dynamic_weights = self.hypernet(activations)  # TODO - Consider passing current L_level output instead of activations

            inner_halted = torch.zeros(active_idx.numel(), dtype=torch.bool, device=device)

            for l in range(self.config.L_cycles):
                active_mask = ~inner_halted
                if not active_mask.any():
                    break

                new_z_L, _ = self._dynamic_forward(
                    z_L=z_L,
                    z_H=z_H,
                    dynamic_weights=dynamic_weights,
                    input_embeddings=input_embeddings,
                    **seq_info
                )

                q_inner_logits = self.q_head_inner(new_z_L[:, 1]).to(torch.float32)

                if self.training:
                    # Probabilistic exploration using Bernoulli distribution
                    halt_probs = torch.sigmoid(q_inner_logits[..., 0])
                    new_inner_halt = torch.bernoulli(halt_probs).to(torch.bool)
                else:
                    new_inner_halt = q_inner_logits[..., 0] > 0

                global_q_inner[active_idx, inner_step_counter] = q_inner_logits[..., 0]
                global_inner_masks[active_idx, inner_step_counter] = active_mask
                inner_step_counter += 1

                active_mask_exp = active_mask.view(-1, 1, 1)
                z_L = torch.where(active_mask_exp, new_z_L, z_L)

                inner_halted = inner_halted | (new_inner_halt & active_mask)

            # Final L_level forward pass to update z_H and return new activations
            new_z_H, activations = self._dynamic_forward(
                z_L=z_L,
                z_H=z_H,
                dynamic_weights=dynamic_weights,
                input_embeddings=None,
                return_activations=True,
                **seq_info
            )

            q_outer = self.q_head_outer(new_z_H[:, 0]).to(torch.float32)
            final_q_outer[active_idx] = q_outer

            if self.training:
                # Probabilistic exploration using Bernoulli distribution
                halt_probs = torch.sigmoid(q_outer[..., 0])
                new_outer_halt = torch.bernoulli(halt_probs).to(torch.bool)
            else:
                new_outer_halt = q_outer[..., 0] > 0

            halting_global_idx = active_idx[new_outer_halt]
            chunk_halted[halting_global_idx] = True
            final_z_H[halting_global_idx] = new_z_H[new_outer_halt]
            final_z_L[halting_global_idx] = new_z_L[new_outer_halt]

            keep_mask = ~new_outer_halt

            z_L = new_z_L[keep_mask]
            z_H = new_z_H[keep_mask]
            input_embeddings = input_embeddings[keep_mask] if input_embeddings is not None else None

            activations = activations[keep_mask]

            active_idx = active_idx[keep_mask]

        # Fallback for never-halted samples
        if active_idx.numel() > 0:
            final_z_H[active_idx] = z_H
            final_z_L[active_idx] = z_L

        if inner_step_counter > 0:
            stacked_q_inner = global_q_inner[:, :inner_step_counter]
            stacked_inner_masks = global_inner_masks[:, :inner_step_counter]
            current_l_h_steps = stacked_inner_masks.to(torch.float32).sum(dim=1)
        else:
            stacked_q_inner = torch.empty((batch_size, 0), dtype=torch.float32, device=device)
            stacked_inner_masks = torch.empty((batch_size, 0), dtype=torch.bool, device=device)
            current_l_h_steps = torch.zeros(batch_size, dtype=torch.float32, device=device)

        new_cumulative_h_steps = carry.cumulative_h_steps + h_steps
        new_cumulative_l_h_steps = carry.cumulative_l_h_steps + current_l_h_steps

        new_carry = RHN_ACTV1InnerCarry(
            z_H=final_z_H.detach(),
            z_L=final_z_L.detach(),
            cumulative_h_steps=new_cumulative_h_steps.detach(),
            cumulative_l_h_steps=new_cumulative_l_h_steps.detach()
        )

        output = self.lm_head(final_z_H)[:, self.puzzle_emb_len:]

        return new_carry, output, (final_q_outer[..., 0], final_q_outer[..., 1], stacked_q_inner, stacked_inner_masks,
                                   h_steps, chunk_halted)



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
        device = batch["inputs"].device

        return RHN_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.float32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: RHN_ACTV1Carry, batch: Dict[str, torch.Tensor], **kwargs) -> Tuple[
        RHN_ACTV1Carry,
        Dict[str, torch.Tensor]
    ]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0.0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        new_inner_carry, logits, inner_outputs = self.inner(new_inner_carry, new_current_data, **kwargs)

        q_halt_logits, q_continue_logits, q_inner_logits, q_inner_masks, h_steps, chunk_halted = inner_outputs

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "q_inner_logits": q_inner_logits,
            "q_inner_masks": q_inner_masks,
            "h_steps": h_steps
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step if self.training else chunk_halted | is_last_step # TODO - Implement ACT for inference

        final_carry = RHN_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data)

        return final_carry, outputs
