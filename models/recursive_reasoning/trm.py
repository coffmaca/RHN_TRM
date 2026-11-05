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
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
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
    mlp_t: bool = False  # use mlp on L instead of transformer
    puzzle_emb_len: int = 16  # if non-zero, its specified to this value
    no_ACT_continue: bool = True  # No continue ACT loss, only use the sigmoid of the halt

    # Autoregressive Decoder config (for generative / chat / coding benchmarks)
    use_ar_decoder: bool = False
    ar_decoder_layers: int = 2
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    halt_acc_threshold: float = 1.0  # 1.0 = exact sequence match; <1.0 = token accuracy threshold for Q-halt


class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
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

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class CrossAttention(nn.Module):
    """Multi-head Cross-Attention projecting queries from decoder states and keys/values from halted z_H."""
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.k_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.v_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.o_proj = CastedLinear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        B, T_tgt, D = hidden_states.shape
        T_mem = memory.shape[1]

        q = self.q_proj(hidden_states).view(B, T_tgt, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, T_mem, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, T_mem, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_tgt, D)
        return self.o_proj(attn_out)


class AutoregressiveDecoderBlock(nn.Module):
    """Causal Self-Attention + Cross-Attention to z_H + SwiGLU MLP with RMS Post-Norm."""
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=True,
        )
        self.cross_attn = CrossAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.cross_attn(hidden_states=hidden_states, memory=memory),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


class AutoregressiveDecoderHead(nn.Module):
    """
    Translates the TRM's latent solution state z_H into output tokens.
    - Uses parallel causal teacher forcing during training.
    - Uses greedy token-by-token autoregressive generation on halted evaluation steps.
    """
    def __init__(
        self,
        config: TinyRecursiveReasoningModel_ACTV1Config,
        shared_embed_tokens: CastedEmbedding,
        shared_lm_head: CastedLinear,
    ) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Share token embeddings and output projection with the TRM inner model
        self.embed_tokens = shared_embed_tokens
        self.lm_head = shared_lm_head

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )

        self.layers = nn.ModuleList(
            [AutoregressiveDecoderBlock(self.config) for _ in range(self.config.ar_decoder_layers)]
        )

    def _slice_cos_sin(self, seq_len: int) -> CosSin:
        if not hasattr(self, "rotary_emb"):
            return None
        cos, sin = self.rotary_emb()
        return cos[:seq_len], sin[:seq_len]

    def _embed_decoder_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        T = input_ids.shape[1]
        embedding = self.embed_tokens(input_ids.to(torch.int32))
        if self.config.pos_encodings == "learned":
            pos_weight = self.embed_pos.embedding_weight[:T].to(self.forward_dtype)
            embedding = 0.707106781 * (embedding + pos_weight)
        return self.embed_scale * embedding

    def _decode_hidden(self, decoder_input_ids: torch.Tensor, z_H_memory: torch.Tensor) -> torch.Tensor:
        T = decoder_input_ids.shape[1]
        cos_sin = self._slice_cos_sin(T)
        hidden_states = self._embed_decoder_inputs(decoder_input_ids)
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states, memory=z_H_memory)
        return hidden_states

    def forward_teacher_forcing(self, z_H_memory: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Single-pass parallel causal teacher forcing for training.
        Shifts labels right by prepending bos_token_id and replacing IGNORE_LABEL_ID (-100) with pad_token_id.
        """
        B, T = labels.shape
        clean_labels = torch.where(labels == IGNORE_LABEL_ID, self.config.pad_token_id, labels)
        bos_col = torch.full((B, 1), self.config.bos_token_id, dtype=clean_labels.dtype, device=labels.device)
        decoder_input_ids = torch.cat([bos_col, clean_labels[:, :-1]], dim=1)

        dec_hidden = self._decode_hidden(decoder_input_ids, z_H_memory)
        return self.lm_head(dec_hidden)

    @torch.no_grad()
    def generate(self, z_H_memory: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Free-running autoregressive greedy decoding for evaluation once ACT halts.
        Returns step-wise logits of shape [B, max_len, vocab_size] whose argmax is the generated token sequence.
        """
        B = z_H_memory.shape[0]
        device = z_H_memory.device

        decoder_input_ids = torch.full((B, 1), self.config.bos_token_id, dtype=torch.int32, device=device)
        step_logits_list: List[torch.Tensor] = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _step in range(max_len):
            dec_hidden = self._decode_hidden(decoder_input_ids, z_H_memory)
            next_token_logits = self.lm_head(dec_hidden[:, -1:, :])  # [B, 1, vocab_size]
            step_logits_list.append(next_token_logits)

            next_token = next_token_logits.argmax(dim=-1).to(torch.int32)  # [B, 1]
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

            finished = finished | (next_token.squeeze(-1) == self.config.eos_token_id)
            if bool(finished.all()):
                # Pad remaining steps with logits peaking at pad_token_id so output length matches labels
                remaining = max_len - (_step + 1)
                if remaining > 0:
                    pad_logit = torch.full(
                        (B, remaining, self.config.vocab_size),
                        -10.0,
                        dtype=next_token_logits.dtype,
                        device=device,
                    )
                    pad_logit[:, :, self.config.pad_token_id] = 10.0
                    step_logits_list.append(pad_logit)
                break

        return torch.cat(step_logits_list, dim=1)


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

        # Optional Autoregressive Output Head
        if self.config.use_ar_decoder:
            self.ar_decoder = AutoregressiveDecoderHead(
                config=self.config,
                shared_embed_tokens=self.embed_tokens,
                shared_lm_head=self.lm_head,
            )

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
        run_ar_generate: bool = False,
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
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

        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                cycle_start_z_L = z_L.detach()
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    record_zL_step(z_L, _L_step)

                zL_H_window_sum = zL_H_window_sum + self._batch_cos_sim(z_L, cycle_start_z_L)
                zL_H_window_count += 1

                z_H = self.L_level(z_H, z_L, **seq_info)
                record_zH_step(z_H)

        # 1 with grad
        cycle_start_z_L = z_L.detach()
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            record_zL_step(z_L, _L_step)

        with torch.no_grad():
            zL_H_window_sum = zL_H_window_sum + self._batch_cos_sim(z_L, cycle_start_z_L)
            zL_H_window_count += 1

        z_H = self.L_level(z_H, z_L, **seq_info)
        record_zH_step(z_H)

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
        z_H_seq = z_H[:, self.puzzle_emb_len:]

        if self.config.use_ar_decoder:
            if self.training or not run_ar_generate:
                output = self.ar_decoder.forward_teacher_forcing(z_H_seq, batch["labels"])
            else:
                output = self.ar_decoder.generate(z_H_seq, max_len=batch["labels"].shape[1])
        else:
            output = self.lm_head(z_H_seq)

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
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # During evaluation, only run free-running autoregressive generation on the final (halting) step
        will_be_last_step = (new_steps + 1) >= self.config.halt_max_steps
        run_ar_generate = (not self.training) and bool(will_be_last_step.any())

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), telemetry_metrics = self.inner(
            new_inner_carry,
            new_current_data,
            run_ar_generate=run_ar_generate,
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            **telemetry_metrics,
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
                    _, _, (next_q_halt_logits, next_q_continue_logits), _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
