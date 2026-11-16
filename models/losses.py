from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x: torch.Tensor) -> torch.Tensor:
    # Clamp inputs to each branch BEFORE dividing so (1 - x_neg) is strictly >= 1.0
    # everywhere. Without clamping, at x == 1.0 the unused (x < 0) branch evaluates
    # 1 / (1 - 1 + eps) = 1 / eps, whose backward derivative 1 / eps^2 overflows float32
    # to +inf and produces 0.0 * inf = NaN gradients.
    x_neg = torch.clamp(x, max=0.0)
    x_pos = torch.clamp(x, min=0.0)
    return torch.where(
        x < 0,
        1.0 / (1.0 - x_neg),
        x_pos + 1.0,
    )


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    s_x = s(x)
    return torch.log(s_x) - torch.log(torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0).to(torch.long)

    # Compute in float32 and gather target before log to avoid materializing multiple [B, S, V] float64 tensors
    s_x = s(logits.to(torch.float32))
    log_norm = torch.log(torch.sum(s_x, dim=-1))
    target_s = torch.gather(s_x, index=transformed_labels.unsqueeze(-1), dim=-1).squeeze(-1)
    prediction_logprobs = torch.log(target_s.clamp_min(1e-30)) - log_norm

    return torch.where(valid_mask, -prediction_logprobs, torch.zeros_like(prediction_logprobs))


def softmax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    # Cast logits to f32 and flatten
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]),
        labels.to(torch.long).view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits: B x SeqLen x Vocab
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            halt_acc_threshold = getattr(getattr(self.model, "config", None), "halt_acc_threshold", 1.0)
            if halt_acc_threshold < 1.0:
                token_acc = is_correct.sum(-1).to(torch.float32) / loss_counts.clamp_min(1).to(torch.float32)
                q_halt_target = (token_acc >= halt_acc_threshold) & (loss_counts > 0)
            else:
                q_halt_target = seq_is_correct

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            valid_count = valid_metrics.sum()
            metrics = {
                "count": valid_count,
                "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == q_halt_target)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

            # Aggregate telemetry metrics across halted sequences
            for k, v in outputs.items():
                if k.startswith("telemetry/"):
                    if v.ndim == 0:
                        metrics[k] = v.detach().to(torch.float32) * valid_count
                    else:
                        metrics[k] = torch.where(valid_metrics, v.detach().to(torch.float32), 0.0).sum()

        # Losses
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], q_halt_target.to(outputs["q_halt_logits"].dtype), reduction="sum"
        )
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unnecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # MoE auxiliary loss and diagnostics
        moe_aux_loss = 0.0
        if "moe_aux_loss" in outputs:
            moe_aux_loss = outputs["moe_aux_loss"]
            metrics["moe_aux_loss"] = moe_aux_loss.detach()
            if "moe_balance_loss" in outputs:
                metrics["moe_balance_loss"] = outputs["moe_balance_loss"]
            if "moe_z_loss" in outputs:
                metrics["moe_z_loss"] = outputs["moe_z_loss"]
            if "moe_max_expert_load" in outputs:
                metrics["moe_max_expert_load"] = outputs["moe_max_expert_load"].to(torch.float32) * valid_count
            if "moe_h_cycle_switch_rate" in outputs:
                metrics["moe_h_cycle_switch_rate"] = outputs["moe_h_cycle_switch_rate"].to(torch.float32) * valid_count

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + moe_aux_loss
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()