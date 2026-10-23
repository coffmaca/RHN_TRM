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
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


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
        self.moe_dropout = nn.Dropout(moe_dropout) if moe_dropout > 0.0 else nn.Identity()

        routed_exp = expert_expansion if expert_expansion is not None else expansion
        shared_exp = shared_expert_expansion if shared_expert_expansion is not None else routed_exp

        # Router linear projection (initialized with small std for near-uniform initial routing)
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

        # Accumulators for recursive gradient-enabled steps
        self._aux_losses: List[torch.Tensor] = []
        self._z_losses: List[torch.Tensor] = []
        self._expert_max_loads: List[torch.Tensor] = []

    def reset_moe_Buffer(self) -> None:
        """Clears accumulated auxiliary losses at the start of a forward pass."""
        self._aux_losses.clear()
        self._z_losses.clear()
        self._expert_max_loads.clear()

    def pop_moe_metrics(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns mean (balance_loss, z_loss, max_expert_load) across recursive steps and resets buffers."""
        if len(self._aux_losses) == 0:
            zero = torch.tensor(0.0, dtype=torch.float32, device=device)
            return zero, zero, zero
        balance_loss = torch.stack(self._aux_losses).mean()
        z_loss = torch.stack(self._z_losses).mean()
        max_load = torch.stack(self._expert_max_loads).mean()
        self.reset_moe_Buffer()
        return balance_loss, z_loss, max_load

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape  # [B, S, D]
        orig_dtype = x.dtype
        x_flat = x.reshape(-1, self.hidden_size)  # [N_tokens, D]

        # 1. Compute Router Logits in FP32 for numerical stability
        router_input = x_flat.to(torch.float32)
        if self.training and self.jitter_noise > 0.0:
            noise = torch.empty_like(router_input).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
            router_input = router_input * noise

        router_logits = F.linear(router_input, self.gate.weight.to(torch.float32))
        if self.router_temperature != 1.0:
            router_logits = router_logits / self.router_temperature

        # 2. Compute Routing Probabilities
        if self.gate_type == "softmax":
            routing_probs = F.softmax(router_logits, dim=-1)
        else:
            routing_probs = torch.sigmoid(router_logits)

        # 3. Select Top-K Experts per Token
        topk_weights, topk_indices = torch.topk(routing_probs, k=self.top_k, dim=-1)  # [N_tokens, K]

        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # 4. Accumulate Load-Balancing & Z-Loss during gradient-enabled recursive passes
        if self.training and torch.is_grad_enabled():
            # Router Z-Loss: penalizes large logits
            z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()

            # Switch / Mixtral Load-Balancing Loss: N * sum(f_i * P_i)
            # P_i: mean router probability per expert across tokens
            if self.gate_type == "sigmoid":
                prob_for_balance = routing_probs / (routing_probs.sum(dim=-1, keepdim=True) + 1e-9)
            else:
                prob_for_balance = routing_probs
            P_i = prob_for_balance.mean(dim=0)  # [num_experts]

            # f_i: fraction of tokens routed to each expert across Top-K selections
            expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).to(torch.float32)  # [N_tokens, K, E]
            f_i = expert_mask.mean(dim=(0, 1))  # [num_experts]

            balance_loss = self.num_experts * torch.sum(f_i * P_i)

            self._aux_losses.append(balance_loss)
            self._z_losses.append(z_loss)
            self._expert_max_loads.append(f_i.max().detach())

        # 5. Dispatch Tokens to Active Routed Experts and Combine
        topk_weights = topk_weights.to(orig_dtype)
        routed_out = torch.zeros_like(x_flat)

        # Token-grouped dispatch (zero token dropping)
        for expert_idx, expert_layer in enumerate(self.experts):
            token_indices, k_positions = torch.where(topk_indices == expert_idx)
            if token_indices.numel() > 0:
                selected_tokens = x_flat.index_select(0, token_indices)
                expert_output = self.moe_dropout(expert_layer(selected_tokens))
                weights = topk_weights[token_indices, k_positions].unsqueeze(-1)
                routed_out.index_add_(0, token_indices, expert_output * weights)

        # 6. Add Shared Expert Contribution (if enabled)
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
