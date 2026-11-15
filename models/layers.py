from typing import List, Optional, Tuple
import einops
import math
import torch
from torch import nn
import torch.nn.functional as F

#try:
#    from flash_attn_interface import flash_attn_func  # type: ignore[import]
#except ImportError:
#    # Fallback to FlashAttention 2
#    from flash_attn import flash_attn_func  # type: ignore[import]
from torch.nn.functional import scaled_dot_product_attention

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class DynamicCastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5)))
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

        self.dynamic_adapter = None

    def set_dynamic_adapter(self, A, B):
        self.dynamic_adapter = (A, B)

    def clear_dynamic_adapter(self):
        self.dynamic_adapter = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.dynamic_adapter is None: # Base Out
            return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        else: # Dynamic Out (using low-rank matrices)
            A, B = self.dynamic_adapter

            if input.dim() == 2:
                input_reshaped = input.unsqueeze(1)  # [Batch, 1, In]
            else:
                input_reshaped = input

            out = torch.einsum('abc,aedc->abed', input, B.to(input.dtype)) # torch.matmul(input, B)
            out = torch.einsum('abed,acfd->abf', out, A.to(input.dtype)) # torch.matmul(out, A)

            if input.dim() == 2:
                out = out.squeeze(1)

            return out


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input.contiguous(),
            self.weight.to(input.dtype),
            bias=self.bias.to(input.dtype) if self.bias is not None else None,
        )


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class DynamicAttention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False, attn_dropout=0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = DynamicCastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = DynamicCastedLinear(self.output_size, self.hidden_size, bias=False)

    def set_dynamic_adapter(self, A_qkv, B_qkv, A_o, B_o):
        self.qkv_proj.set_dynamic_adapter(A_qkv, B_qkv)
        self.o_proj.set_dynamic_adapter(A_o, B_o)

    def clear_dynamic_adapter(self):
        self.qkv_proj.clear_dynamic_adapter()
        self.o_proj.clear_dynamic_adapter()

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Attention
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'),
                                (query, key, value))  # needed for scaled_dot_product_attention but not flash_attn_func
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Attention
        # TODO - Reinstate flash-attn (optional)
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value)) # needed for scaled_dot_product_attention but not flash_attn_func
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)

class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class DynamicSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()

        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = DynamicCastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = DynamicCastedLinear(inter, hidden_size, bias=False)

    def set_dynamic_adapter(self, A_up, B_up, A_down, B_down):
        self.gate_up_proj.set_dynamic_adapter(A_up, B_up)
        self.down_proj.set_dynamic_adapter(A_down, B_down)

    def clear_dynamic_adapter(self):
        self.gate_up_proj.clear_dynamic_adapter()
        self.down_proj.clear_dynamic_adapter()

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class MoESwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        num_experts: int = 8,
        top_k: int = 2,
        expert_expansion: Optional[float] = None,
        num_shared_experts: int = 0,
        shared_expert_expansion: Optional[float] = None,
        gate_type: str = "softmax",
        norm_topk_prob: bool = True,
        router_temperature: float = 1.0,
        jitter_noise: float = 0.0,
        moe_dropout: float = 0.0,
        lock_gate_weights: bool = True,
    ):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"moe_top_k ({top_k}) must be between 1 and moe_num_experts ({num_experts}).")
        if gate_type not in ("softmax", "sigmoid"):
            raise ValueError(f"moe_gate_type must be 'softmax' or 'sigmoid', got '{gate_type}'.")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate_type = gate_type
        # For Top-1 (Switch), normalizing by the single selected weight would yield 1.0 and sever router gradients
        self.norm_topk_prob = norm_topk_prob if top_k > 1 else False
        self.router_temperature = router_temperature
        self.jitter_noise = jitter_noise
        self.lock_gate_weights = lock_gate_weights
        self.moe_dropout = nn.Dropout(moe_dropout) if moe_dropout > 0.0 else nn.Identity()

        routed_exp = expert_expansion if expert_expansion is not None else expansion
        shared_exp = shared_expert_expansion if shared_expert_expansion is not None else routed_exp

        # Router linear projection
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        trunc_normal_init_(self.gate.weight, std=1.0 / math.sqrt(hidden_size))

        # Bank of Routed SwiGLU Experts
        self.experts = nn.ModuleList([
            SwiGLU(hidden_size=hidden_size, expansion=routed_exp)
            for _ in range(num_experts)
        ])

        # Optional Always-Active Shared Expert(s)
        self.num_shared_experts = num_shared_experts
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                SwiGLU(hidden_size=hidden_size, expansion=shared_exp)
                for _ in range(num_shared_experts)
            ])
        else:
            self.shared_experts = None

        # Cached routing state for the current H_cycle
        self._cached_topk_indices: Optional[torch.Tensor] = None
        self._cached_topk_weights: Optional[torch.Tensor] = None

        # Accumulators for gradient-enabled H_cycles
        self._aux_losses: List[torch.Tensor] = []
        self._z_losses: List[torch.Tensor] = []
        self._expert_max_loads: List[torch.Tensor] = []
        self._switch_rates: List[torch.Tensor] = []

    def clear_routing_cache(self) -> None:
        """Clears cached H_cycle routing tensors to release autograd graph references."""
        self._cached_topk_indices = None
        self._cached_topk_weights = None

    def reset_moe_Buffer(self) -> None:
        """Clears accumulated auxiliary losses, switch-rate metrics, and routing cache."""
        self._aux_losses.clear()
        self._z_losses.clear()
        self._expert_max_loads.clear()
        self._switch_rates.clear()
        self.clear_routing_cache()

    def pop_moe_metrics(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns mean (balance_loss, z_loss, max_expert_load, h_cycle_switch_rate) and resets buffers."""
        zero = torch.tensor(0.0, dtype=torch.float32, device=device)
        balance_loss = torch.stack(self._aux_losses).mean() if self._aux_losses else zero
        z_loss = torch.stack(self._z_losses).mean() if self._z_losses else zero
        max_load = torch.stack(self._expert_max_loads).mean() if self._expert_max_loads else zero
        switch_rate = torch.stack(self._switch_rates).mean() if self._switch_rates else zero
        self.reset_moe_Buffer()
        return balance_loss, z_loss, max_load, switch_rate

    def _compute_router_scores(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes FP32 router logits and probabilities."""
        router_input = x_flat.to(torch.float32)
        if self.training and self.jitter_noise > 0.0:
            noise = torch.empty_like(router_input).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
            router_input = router_input * noise

        router_logits = F.linear(router_input, self.gate.weight.to(torch.float32))
        if self.router_temperature != 1.0:
            router_logits = router_logits / self.router_temperature

        if self.gate_type == "softmax":
            routing_probs = F.softmax(router_logits, dim=-1)
        else:
            routing_probs = torch.sigmoid(router_logits)

        return router_logits, routing_probs

    @torch._dynamo.disable
    def _dispatch_experts(
        self,
        x_flat: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights_cast: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sparse token dispatch: only evaluates each expert on the tokens actually routed to it.
        Decorated with @torch._dynamo.disable so torch.compile does not graph-break or
        recompile on data-dependent per-expert token counts.
        """
        routed_out = torch.zeros_like(x_flat)
        for expert_idx, expert_layer in enumerate(self.experts):
            token_indices, k_positions = torch.where(topk_indices == expert_idx)
            if token_indices.numel() > 0:
                selected_tokens = x_flat.index_select(0, token_indices)
                expert_output = self.moe_dropout(expert_layer(selected_tokens))
                weights = topk_weights_cast[token_indices, k_positions].unsqueeze(-1)
                routed_out.index_add_(0, token_indices, expert_output * weights)
        return routed_out

    def forward(self, x: torch.Tensor, update_routing: bool = True) -> torch.Tensor:
        orig_shape = x.shape  # [B, S, D]
        orig_dtype = x.dtype
        x_flat = x.reshape(-1, self.hidden_size)  # [N_tokens, D]

        # 1. Select Experts (only on the first L_cycle of an H_cycle, or if cache is empty)
        if update_routing or self._cached_topk_indices is None:
            router_logits, routing_probs = self._compute_router_scores(x_flat)
            topk_weights, topk_indices = torch.topk(routing_probs, k=self.top_k, dim=-1)  # [N_tokens, K]

            if self.norm_topk_prob:
                topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

            # Track fraction of tokens switching experts between consecutive H_cycles
            with torch.no_grad():
                curr_mask = F.one_hot(topk_indices, num_classes=self.num_experts).any(dim=1)  # [N_tokens, E]
                if (
                    self._cached_topk_indices is not None
                    and self._cached_topk_indices.shape == topk_indices.shape
                ):
                    prev_mask = F.one_hot(self._cached_topk_indices, num_classes=self.num_experts).any(dim=1)
                    overlap = (curr_mask & prev_mask).sum(dim=-1).to(torch.float32) / float(self.top_k)
                    self._switch_rates.append((1.0 - overlap).mean())

                f_i_detached = curr_mask.to(torch.float32).mean(dim=0) / float(self.top_k)
                self._expert_max_loads.append(f_i_detached.max())

            # Cache selection for subsequent L_cycles in this H_cycle
            self._cached_topk_indices = topk_indices
            self._cached_topk_weights = topk_weights

            # Accumulate Load-Balancing & Z-Loss once per gradient-enabled H_cycle
            if self.training and torch.is_grad_enabled():
                z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()

                if self.gate_type == "sigmoid":
                    prob_for_balance = routing_probs / (routing_probs.sum(dim=-1, keepdim=True) + 1e-9)
                else:
                    prob_for_balance = routing_probs
                P_i = prob_for_balance.mean(dim=0)  # [num_experts]

                expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).to(torch.float32)
                f_i = expert_mask.mean(dim=(0, 1))  # [num_experts]

                balance_loss = self.num_experts * torch.sum(f_i * P_i)

                self._aux_losses.append(balance_loss)
                self._z_losses.append(z_loss)
        else:
            # Reuse the expert indices locked at _L_step == 0 of the current H_cycle
            topk_indices = self._cached_topk_indices
            if self.lock_gate_weights and self._cached_topk_weights is not None:
                topk_weights = self._cached_topk_weights
            else:
                # Optional: keep expert indices locked, but recompute gating weights for those experts
                _, routing_probs = self._compute_router_scores(x_flat)
                topk_weights = torch.gather(routing_probs, dim=-1, index=topk_indices)
                if self.norm_topk_prob:
                    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # 2. Dispatch Tokens to the Locked Active Experts (Sparse Memory Footprint)
        topk_weights_cast = topk_weights.to(orig_dtype)
        routed_out = self._dispatch_experts(x_flat, topk_indices, topk_weights_cast)

        # 3. Add Shared Expert Contribution (if enabled)
        if self.shared_experts is not None:
            for shared_expert in self.shared_experts:
                routed_out = routed_out + shared_expert(x_flat)

        return routed_out.view(orig_shape)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
