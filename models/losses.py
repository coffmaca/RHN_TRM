from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

from config_init import PretrainConfig

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, config: PretrainConfig):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[config.arch.loss.model_extra["loss_type"]]
        self.l2_lambda = config.arch.model_extra["hypernet_l2_lambda"]
        self.hypernet_ema_decay_slow = config.arch.model_extra["hypernet_ema_decay_slow"]
        self.hypernet_ema_decay_fast = config.arch.model_extra["hypernet_ema_decay_fast"]
        self.hypernet_lml2_diverg_penalty_gamma = config.arch.model_extra["hypernet_lml2_diverg_penalty_gamma"]

        self.register_buffer("ema_lm_slow", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("ema_lm_fast", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("ema_l2_slow", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("ema_l2_fast", torch.tensor(1.0, dtype=torch.float32))
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def lm_l2_divergence_penalty(self, lm_loss, scaled_l2_loss, divergence_warmup_factor):
        # with torch.no_grad():
        #     lm_fast_delta = F.relu((self.ema_lm_fast - lm_loss) / (self.ema_lm_fast + 1e-8))
        lm_fast_delta = F.relu((self.ema_lm_fast - lm_loss) / (self.ema_lm_fast + 1e-8))
        l2_fast_delta = F.relu((scaled_l2_loss - self.ema_l2_fast) / (self.ema_l2_fast + 1e-8))
        total_fast_delta = lm_fast_delta * l2_fast_delta

        # with torch.no_grad():
        #     lm_slow_delta = F.relu((self.ema_lm_slow - lm_loss) / (self.ema_lm_slow + 1e-8))
        lm_slow_delta = F.relu((self.ema_lm_slow - lm_loss) / (self.ema_lm_slow + 1e-8))
        l2_slow_delta = F.relu((scaled_l2_loss - self.ema_l2_slow) / (self.ema_l2_slow + 1e-8))
        total_slow_delta = lm_slow_delta * l2_slow_delta

        divergence_penalty = self.hypernet_lml2_diverg_penalty_gamma * divergence_warmup_factor * (total_fast_delta +
                                                                                                   total_slow_delta)

        if self.training:
            with torch.no_grad():
                self.ema_lm_fast =  (self.ema_lm_fast * self.hypernet_ema_decay_fast) + (
                        lm_loss.detach() * (1 - self.hypernet_ema_decay_fast))
                self.ema_l2_fast = (self.ema_l2_fast * self.hypernet_ema_decay_fast) + (
                        scaled_l2_loss.detach() * (1 - self.hypernet_ema_decay_fast))

                self.ema_lm_slow = (self.ema_lm_slow * self.hypernet_ema_decay_slow) + (
                        lm_loss.detach() * (1 - self.hypernet_ema_decay_slow))
                self.ema_l2_slow = (self.ema_l2_slow * self.hypernet_ema_decay_slow) + (
                        scaled_l2_loss.detach() * (1 - self.hypernet_ema_decay_slow))

        return divergence_penalty, total_fast_delta, total_slow_delta

    def forward(
        self,
        return_keys: Sequence[str],
        divergence_warmup_factor: float = 1.0,
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        scaled_l2_loss = (outputs["hypernet_l2"] * valid_metrics).sum() * self.l2_lambda

        divergence_penalty, total_fast_delta, total_slow_delta = self.lm_l2_divergence_penalty(lm_loss,
                                                                                               scaled_l2_loss,
                                                                                               divergence_warmup_factor)

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            "hypernet_l2_loss": scaled_l2_loss.detach(),
            "divergence_short": total_fast_delta.detach(),
            "divergence_long": total_slow_delta.detach(),
            "divergence_penalty": divergence_penalty.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        final_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + scaled_l2_loss + divergence_penalty

        return new_carry, final_loss, metrics, detached_outputs, new_carry.halted.all()

