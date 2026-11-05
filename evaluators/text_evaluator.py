import os
import ast
from typing import Dict, Sequence, Optional
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

IGNORE_LABEL_ID = -100
EOS_ID = 2


def _lcs_length(x: Sequence[int], y: Sequence[int]) -> int:
    """Computes Longest Common Subsequence length for token-level ROUGE-L."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        xi = x[i - 1]
        for j in range(1, n + 1):
            temp = dp[j]
            if xi == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


class GenerativeTextEvaluator:
    required_outputs = {"preds"}

    def __init__(self, data_path: str, eval_metadata, **kwargs):
        tok_path = os.path.join(data_path, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path) if os.path.exists(tok_path) else None
        self.begin_eval()

    def begin_eval(self):
        self.total_samples = 0
        self.rouge_l_sum = 0.0
        self.exact_match_sum = 0.0
        self.syntax_valid_sum = 0.0

    def _truncate_at_eos(self, token_ids: Sequence[int]) -> list:
        out = []
        for tid in token_ids:
            if tid == EOS_ID or tid == IGNORE_LABEL_ID:
                break
            if tid > EOS_ID:
                out.append(int(tid))
        return out

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        pred_ids_batch = preds["preds"].detach().cpu().tolist()
        label_ids_batch = batch["labels"].detach().cpu().tolist()

        for pred_seq, label_seq in zip(pred_ids_batch, label_ids_batch):
            # Skip padding-only dummy batch entries
            if all(t == IGNORE_LABEL_ID for t in label_seq):
                continue

            p_tokens = self._truncate_at_eos(pred_seq)
            r_tokens = self._truncate_at_eos(label_seq)

            self.total_samples += 1
            if p_tokens == r_tokens:
                self.exact_match_sum += 1.0

            lcs = _lcs_length(p_tokens, r_tokens)
            if lcs > 0 and len(p_tokens) > 0 and len(r_tokens) > 0:
                prec = lcs / len(p_tokens)
                rec = lcs / len(r_tokens)
                self.rouge_l_sum += (2 * prec * rec) / (prec + rec + 1e-8)

            if self.tokenizer is not None:
                decoded_code = self.tokenizer.decode(p_tokens, skip_special_tokens=True).strip()
                if decoded_code:
                    try:
                        ast.parse(decoded_code)
                        self.syntax_valid_sum += 1.0
                    except SyntaxError:
                        pass

    def result(self, save_path: Optional[str], rank: int, world_size: int, group: Optional[torch.distributed.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        stats = torch.tensor(
            [float(self.total_samples), self.rouge_l_sum, self.exact_match_sum, self.syntax_valid_sum],
            dtype=torch.float64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if world_size > 1 and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)

        total = max(stats[0].item(), 1.0)
        if rank == 0:
            return {
                "GenerativeText/rougeL": stats[1].item() / total,
                "GenerativeText/exact_match": stats[2].item() / total,
                "GenerativeText/python_syntax_valid": stats[3].item() / total,
            }
        return None