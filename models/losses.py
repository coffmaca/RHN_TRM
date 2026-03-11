from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

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
    def __init__(self, model: nn.Module, loss_type: str, ponder_weight: float = 0.001, lambda_outer_halt: float = 0.5,
                 lambda_inner_halt: float = 0.5):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.ponder_weight = ponder_weight
        self.lambda_outer_halt = lambda_outer_halt
        self.lambda_inner_halt = lambda_inner_halt
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        current_ponder_weight: Optional[float] = None,
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

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = loss_counts > 0
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
            }

        # Losses
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"],
                                                         seq_is_correct.to(outputs["q_halt_logits"].dtype),
                                                         reduction="sum")

        q_inner_logits = outputs["q_inner_logits"]
        q_inner_masks = outputs["q_inner_masks"]
        inner_targets = seq_is_correct.unsqueeze(1).expand_as(q_inner_logits).to(q_inner_logits.dtype)
        raw_inner_bce = F.binary_cross_entropy_with_logits(q_inner_logits, inner_targets, reduction="none")
        q_inner_loss = (raw_inner_bce * q_inner_masks.to(raw_inner_bce.dtype)).sum()

        active_inner_steps = q_inner_masks.to(torch.float32).sum(dim=1)
        ponder_weight = current_ponder_weight if current_ponder_weight is not None else self.ponder_weight
        ponder_loss = ponder_weight * active_inner_steps.mean()

        total_loss = lm_loss + self.lambda_outer_halt * q_halt_loss + self.lambda_inner_halt * q_inner_loss + ponder_loss


        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            "q_inner_loss": q_inner_loss.detach(),
            "ponder_loss": ponder_loss.detach(),
            "avg_active_steps": active_inner_steps.mean().detach(),
            "avg_h_steps": outputs["h_steps"].mean().detach()
        })

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        # Return True for halted because batch will always be completely finished
        return new_carry, total_loss, metrics, detached_outputs, torch.tensor(True, device=labels.device)

