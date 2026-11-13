from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, LinearSwish, SwiGLU, MoESwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
)
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
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

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

    # Mixture of Experts (MoE) Configuration
    use_moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_expert_expansion: Optional[float] = None
    moe_num_shared_experts: int = 0
    moe_shared_expert_expansion: Optional[float] = None
    moe_gate_type: str = "softmax"
    moe_norm_topk_prob: bool = True
    moe_router_temperature: float = 1.0
    moe_jitter_noise: float = 0.0
    moe_dropout: float = 0.0
    moe_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 0.001
    moe_apply_to_mlp_t: bool = False
    moe_lock_gate_weights: bool = True
    moe_route_on_H_step: bool = False

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config

        if self.config.mlp_t:
            self.puzzle_emb_len = (
                -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
                if self.config.puzzle_emb_len == 0
                else self.config.puzzle_emb_len
            )
            seq_dim = self.config.seq_len + self.puzzle_emb_len
            if self.config.use_moe and self.config.moe_apply_to_mlp_t:
                self.mlp_t = MoESwiGLU(
                    hidden_size=seq_dim,
                    expansion=config.expansion,
                    num_experts=config.moe_num_experts,
                    top_k=config.moe_top_k,
                    expert_expansion=config.moe_expert_expansion,
                    num_shared_experts=config.moe_num_shared_experts,
                    shared_expert_expansion=config.moe_shared_expert_expansion,
                    gate_type=config.moe_gate_type,
                    norm_topk_prob=config.moe_norm_topk_prob,
                    router_temperature=config.moe_router_temperature,
                    jitter_noise=config.moe_jitter_noise,
                    moe_dropout=config.moe_dropout,
                    lock_gate_weights=config.moe_lock_gate_weights,
                )
            else:
                self.mlp_t = SwiGLU(
                    hidden_size=seq_dim,
                    expansion=config.expansion,
                )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )

        if self.config.use_moe:
            self.mlp = MoESwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
                num_experts=config.moe_num_experts,
                top_k=config.moe_top_k,
                expert_expansion=config.moe_expert_expansion,
                num_shared_experts=config.moe_num_shared_experts,
                shared_expert_expansion=config.moe_shared_expert_expansion,
                gate_type=config.moe_gate_type,
                norm_topk_prob=config.moe_norm_topk_prob,
                router_temperature=config.moe_router_temperature,
                jitter_noise=config.moe_jitter_noise,
                moe_dropout=config.moe_dropout,
                lock_gate_weights=config.moe_lock_gate_weights,
            )
        else:
            self.mlp = SwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
            )
        self.norm_eps = config.rms_norm_eps

    def forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        update_moe_routing: bool = True,
    ) -> torch.Tensor:
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            if isinstance(self.mlp_t, MoESwiGLU):
                out = self.mlp_t(hidden_states, update_routing=update_moe_routing)
            else:
                out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            # Self Attention
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )

        # Fully Connected (MoE or Dense SwiGLU)
        if isinstance(self.mlp, MoESwiGLU):
            out = self.mlp(hidden_states, update_routing=update_moe_routing)
        else:
            out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def clear_moe_routing_cache(self) -> None:
        for layer in self.layers:
            if isinstance(layer.mlp, MoESwiGLU):
                layer.mlp.clear_routing_cache()
            if hasattr(layer, "mlp_t") and isinstance(layer.mlp_t, MoESwiGLU):
                layer.mlp_t.clear_routing_cache()

    def reset_moe_buffers(self) -> None:
        for layer in self.layers:
            if isinstance(layer.mlp, MoESwiGLU):
                layer.mlp.reset_moe_Buffer()
            if hasattr(layer, "mlp_t") and isinstance(layer.mlp_t, MoESwiGLU):
                layer.mlp_t.reset_moe_Buffer()

    def collect_moe_metrics(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        balance_losses, z_losses, max_loads, switch_rates = [], [], [], []
        for layer in self.layers:
            for submod_name in ("mlp", "mlp_t"):
                submod = getattr(layer, submod_name, None)
                if isinstance(submod, MoESwiGLU):
                    b_loss, z_loss, m_load, s_rate = submod.pop_moe_metrics(device)
                    balance_losses.append(b_loss)
                    z_losses.append(z_loss)
                    max_loads.append(m_load)
                    switch_rates.append(s_rate)
        if not balance_losses:
            zero = torch.tensor(0.0, dtype=torch.float32, device=device)
            return zero, zero, zero, zero
        return (
            torch.stack(balance_losses).mean(),
            torch.stack(z_losses).mean(),
            torch.stack(max_loads).mean(),
            torch.stack(switch_rates).mean(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        update_moe_routing: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                update_moe_routing=update_moe_routing,
                **kwargs,
            )
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
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
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

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
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def _batch_cos_sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Computes per-sample cosine similarity averaged across the sequence dimension [B]."""
        return F.cosine_similarity(a.detach().float(), b.detach().float(), dim=-1).mean(dim=-1)

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[
        TinyRecursiveReasoningModel_ACTV1InnerCarry,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L
        init_z_H = z_H.detach()
        init_z_L = z_L.detach()

        # Telemetry accumulators (shape: [B])
        B = z_H.shape[0]
        device = z_H.device
        zero_b = torch.zeros(B, dtype=torch.float32, device=device)

        zL_lag1_sum, zL_lag1_count = zero_b.clone(), 0
        zL_lag2_sum, zL_lag2_count = zero_b.clone(), 0
        zL_first_L_sum, zL_first_L_count = zero_b.clone(), 0
        zL_last_L_sum, zL_last_L_count = zero_b.clone(), 0
        zL_H_window_sum, zL_H_window_count = zero_b.clone(), 0
        zL_delta_cos_sum, zL_delta_cos_count = zero_b.clone(), 0

        zH_step_sum, zH_step_count = zero_b.clone(), 0
        zH_delta_cos_sum, zH_delta_cos_count = zero_b.clone(), 0

        prev_z_L = z_L.detach()
        prev2_z_L: Optional[torch.Tensor] = None
        prev_delta_z_L: Optional[torch.Tensor] = None

        prev_z_H = z_H.detach()
        prev_delta_z_H: Optional[torch.Tensor] = None

        def record_zL_step(curr_z_L: torch.Tensor, l_idx: int):
            nonlocal prev_z_L, prev2_z_L, prev_delta_z_L
            nonlocal zL_lag1_sum, zL_lag1_count, zL_lag2_sum, zL_lag2_count
            nonlocal zL_first_L_sum, zL_first_L_count, zL_last_L_sum, zL_last_L_count
            nonlocal zL_delta_cos_sum, zL_delta_cos_count

            with torch.no_grad():
                curr_det = curr_z_L.detach()
                cos_lag1 = self._batch_cos_sim(curr_det, prev_z_L)
                zL_lag1_sum = zL_lag1_sum + cos_lag1
                zL_lag1_count += 1

                if l_idx == 0:
                    zL_first_L_sum = zL_first_L_sum + cos_lag1
                    zL_first_L_count += 1
                if l_idx == self.config.L_cycles - 1:
                    zL_last_L_sum = zL_last_L_sum + cos_lag1
                    zL_last_L_count += 1

                if prev2_z_L is not None:
                    zL_lag2_sum = zL_lag2_sum + self._batch_cos_sim(curr_det, prev2_z_L)
                    zL_lag2_count += 1

                delta_z_L = curr_det.float() - prev_z_L.float()
                if prev_delta_z_L is not None:
                    zL_delta_cos_sum = zL_delta_cos_sum + self._batch_cos_sim(delta_z_L, prev_delta_z_L)
                    zL_delta_cos_count += 1

                prev_delta_z_L = delta_z_L
                prev2_z_L = prev_z_L
                prev_z_L = curr_det

        def record_zH_step(curr_z_H: torch.Tensor):
            nonlocal prev_z_H, prev_delta_z_H
            nonlocal zH_step_sum, zH_step_count, zH_delta_cos_sum, zH_delta_cos_count

            with torch.no_grad():
                curr_det = curr_z_H.detach()
                zH_step_sum = zH_step_sum + self._batch_cos_sim(curr_det, prev_z_H)
                zH_step_count += 1

                delta_z_H = curr_det.float() - prev_z_H.float()
                if prev_delta_z_H is not None:
                    zH_delta_cos_sum = zH_delta_cos_sum + self._batch_cos_sim(delta_z_H, prev_delta_z_H)
                    zH_delta_cos_count += 1

                prev_delta_z_H = delta_z_H
                prev_z_H = curr_det

        if self.config.use_moe:
            self.L_level.reset_moe_buffers()

        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                cycle_start_z_L = z_L.detach()
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(
                        z_L,
                        z_H + input_embeddings,
                        update_moe_routing=(_L_step == 0),
                        **seq_info,
                    )
                    record_zL_step(z_L, _L_step)

                zL_H_window_sum = zL_H_window_sum + self._batch_cos_sim(z_L, cycle_start_z_L)
                zL_H_window_count += 1

                z_H = self.L_level(
                    z_H,
                    z_L,
                    update_moe_routing=self.config.moe_route_on_H_step,
                    **seq_info,
                )
                record_zH_step(z_H)

        # 1 H_cycle with grad
        cycle_start_z_L = z_L.detach()
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(
                z_L,
                z_H + input_embeddings,
                update_moe_routing=(_L_step == 0),
                **seq_info,
            )
            record_zL_step(z_L, _L_step)

        with torch.no_grad():
            zL_H_window_sum = zL_H_window_sum + self._batch_cos_sim(z_L, cycle_start_z_L)
            zL_H_window_count += 1

        z_H = self.L_level(
            z_H,
            z_L,
            update_moe_routing=self.config.moe_route_on_H_step,
            **seq_info,
        )
        record_zH_step(z_H)

        # Release cached routing tensors so no autograd references persist on the module
        if self.config.use_moe:
            self.L_level.clear_moe_routing_cache()

        with torch.no_grad():
            telemetry_metrics = {
                "telemetry/zL_cos_sim_lag1": zL_lag1_sum / max(zL_lag1_count, 1),
                "telemetry/zL_cos_sim_lag2": zL_lag2_sum / max(zL_lag2_count, 1),
                "telemetry/zL_cos_sim_first_L": zL_first_L_sum / max(zL_first_L_count, 1),
                "telemetry/zL_cos_sim_last_L": zL_last_L_sum / max(zL_last_L_count, 1),
                "telemetry/zL_cos_sim_H_window": zL_H_window_sum / max(zL_H_window_count, 1),
                "telemetry/zL_cos_sim_macro_step": self._batch_cos_sim(z_L, init_z_L),
                "telemetry/zH_cos_sim_H_step": zH_step_sum / max(zH_step_count, 1),
                "telemetry/zH_cos_sim_macro_step": self._batch_cos_sim(z_H, init_z_H),
                "telemetry/zL_delta_cos_sim": zL_delta_cos_sum / max(zL_delta_cos_count, 1),
                "telemetry/zH_delta_cos_sim": zH_delta_cos_sum / max(zH_delta_cos_count, 1),
                "telemetry/zL_zH_cos_sim": self._batch_cos_sim(z_L, z_H),
            }

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)  # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), telemetry_metrics


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), telemetry_metrics = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            **telemetry_metrics,
        }

        if self.config.use_moe:
            balance_loss, z_loss, max_load, switch_rate = self.inner.L_level.collect_moe_metrics(logits.device)
            batch_scale = float(logits.shape[0])
            outputs["moe_aux_loss"] = batch_scale * (
                self.config.moe_aux_loss_coef * balance_loss +
                self.config.moe_z_loss_coef * z_loss
            )
            outputs["moe_balance_loss"] = (batch_scale * balance_loss).detach()
            outputs["moe_z_loss"] = (batch_scale * z_loss).detach()
            outputs["moe_max_expert_load"] = max_load.detach()
            outputs["moe_h_cycle_switch_rate"] = switch_rate.detach()

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
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    _, _, (next_q_halt_logits, next_q_continue_logits), _ = self.inner(
                        new_inner_carry, new_current_data
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
