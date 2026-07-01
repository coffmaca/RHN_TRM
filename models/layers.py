from typing import Tuple
import einops
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

            out = torch.einsum('abc,adc->abd', input, B.to(input.dtype)) # torch.matmul(input, B)
            out = torch.einsum('abd,aed->abe', out, A.to(input.dtype)) # torch.matmul(out, A)

            if input.dim() == 2:
                out = out.squeeze(1)

            return out

    def __getstate__(self):
        state = self.__dict__.copy()
        state['dynamic_adapter'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


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


class CastedParameter(nn.Module):
    def __init__(self,
                 size: tuple,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.parameter_weight = nn.Parameter(
            trunc_normal_init_(torch.empty(size), std=init_std)
        )

    def forward(self) -> torch.Tensor:
        return self.parameter_weight.to(self.cast_to)


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


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25,
                 decay: float = 0.99, epsilon: float = 1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Initialize the codebook embeddings (using Buffers so they aren't trained by Adam)
        embed = torch.randn(num_embeddings, embedding_dim)
        embed = F.normalize(embed, p=2, dim=1)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", embed.clone())

    def forward(self, x: torch.Tensor, update_codebook) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.embedding.dtype != x.dtype:
            self.embedding.data = self.embedding.data.to(x.dtype)
            self.ema_w.data = self.ema_w.data.to(x.dtype)

        # x shape: [Batch, Perceiver_Rank, Dim]
        flat_x = x.reshape(-1, self.embedding_dim)
        flat_x_norm = F.normalize(flat_x, p=2, dim=1)

        emb = self.embedding.to(x.dtype)
        emb_norm = F.normalize(emb, p=2, dim=1)

        # Compute distances: (x - e)^2 = x^2 - 2xe + e^2 in unit space simplifies to 2 - 2 * (x dot e)
        distances = 2.0 - 2.0 * torch.matmul(flat_x_norm, emb_norm.t())

        # Find the closest vectors
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        self.batch_active_codes = torch.unique(encoding_indices).numel()
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device, dtype=x.dtype)
        encodings.scatter_(1, encoding_indices, 1)

        # Update the codebook using Exponential Moving Average (only in training)
        if self.training and update_codebook:
            with torch.no_grad():
                self.cluster_size.data.mul_(self.decay).add_(
                    encodings.sum(0), alpha=1 - self.decay
                )

                # Laplace smoothing to prevent dead codes
                n = self.cluster_size.sum()
                cluster_size = (
                        (self.cluster_size + self.epsilon)
                        / (n + self.num_embeddings * self.epsilon)
                        * n
                )

                dw = torch.matmul(encodings.t(), flat_x_norm)
                self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                self.embedding.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
                self.embedding.data = F.normalize(self.embedding.data, p=2, dim=1)

                usage_threshold = 1.0

                dead_indices = torch.nonzero(self.cluster_size < usage_threshold, as_tuple=False).squeeze(-1)

                if len(dead_indices) > 0:
                    num_dead = len(dead_indices)
                    rand_indices = torch.randperm(flat_x.shape[0], device=flat_x.device)[:num_dead]

                    sampled_norm = flat_x_norm[rand_indices]
                    self.embedding.data[dead_indices] = sampled_norm

                    self.cluster_size.data[dead_indices] = usage_threshold

                    self.ema_w.data[dead_indices] = sampled_norm * usage_threshold

        quantized_discrete = torch.matmul(encodings, self.embedding.to(encodings.dtype)).view(x.shape)

        x_norm = flat_x_norm.view(x.shape)

        e_latent_loss = F.mse_loss(quantized_discrete.detach(), x_norm, reduction='none')
        loss = self.commitment_cost * e_latent_loss.mean(dim=[-2, -1])

        quantized_out = x_norm + (quantized_discrete - x_norm).detach()

        return quantized_out, loss


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
