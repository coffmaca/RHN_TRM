import os
import json
import argparse
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast
from dataset.common import PuzzleDatasetMetadata

IGNORE_LABEL_ID = -100
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


def get_raw_pairs(task: str, max_samples: int):
    num_test = max(150, max_samples // 5)

    if task == "mbpp":
        # Load all splits from canonical namespace/name repo (974 total problems)
        ds_all = load_dataset(
            "google-research-datasets/mbpp",
            "full",
            split="train+validation+prompt+test",
        )
        all_pairs = [
            (f"# Task: {ex['text'].strip()}\n", ex["code"].strip())
            for ex in ds_all
        ]

        # Reserve held-out test set first
        test_pairs = all_pairs[-num_test:]
        base_train_pairs = all_pairs[:-num_test]

        # Ensure train split has at least max_samples (e.g., 1000) so PuzzleDataset
        # never drops the epoch when global_batch_size=768
        train_pairs = []
        while len(train_pairs) < max_samples:
            needed = max_samples - len(train_pairs)
            train_pairs.extend(base_train_pairs[:needed])

        return train_pairs[:max_samples], test_pairs

    elif task == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        pairs = []
        total_needed = max_samples + num_test
        for ex in ds:
            if ex["input"].strip():
                prompt = f"Instruction: {ex['instruction'].strip()}\nInput: {ex['input'].strip()}\nResponse:\n"
            else:
                prompt = f"Instruction: {ex['instruction'].strip()}\nResponse:\n"
            pairs.append((prompt, ex["output"].strip()))
            if len(pairs) >= total_needed:
                break

        return pairs[:max_samples], pairs[max_samples : max_samples + num_test]

    else:
        raise ValueError(f"Unsupported task: {task}")


def build_or_load_tokenizer(train_pairs, output_dir: str, vocab_size: int):
    corpus = [p for pair in train_pairs for p in pair]

    # Train Byte-Level BPE directly to avoid transformers' duplicate special_tokens kwarg bug
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>"],
    )
    tok.train_from_iterator(corpus, trainer=trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
    )

    assert tokenizer.pad_token_id == PAD_ID, f"Expected pad_id={PAD_ID}, got {tokenizer.pad_token_id}"
    assert tokenizer.bos_token_id == BOS_ID, f"Expected bos_id={BOS_ID}, got {tokenizer.bos_token_id}"
    assert tokenizer.eos_token_id == EOS_ID, f"Expected eos_id={EOS_ID}, got {tokenizer.eos_token_id}"

    tok_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    tokenizer.save_pretrained(tok_dir)
    return tokenizer


def encode_split(pairs, tokenizer, seq_len: int):
    N = len(pairs)
    inputs_arr = np.full((N, seq_len), PAD_ID, dtype=np.int32)
    labels_arr = np.full((N, seq_len), IGNORE_LABEL_ID, dtype=np.int32)

    for i, (prompt_text, target_text) in enumerate(pairs):
        in_ids = tokenizer.encode(prompt_text, add_special_tokens=False)[:seq_len]
        inputs_arr[i, : len(in_ids)] = in_ids

        # Append EOS token so the autoregressive decoder learns to terminate generation
        tgt_ids = tokenizer.encode(target_text, add_special_tokens=False)[: seq_len - 1] + [EOS_ID]
        labels_arr[i, : len(tgt_ids)] = tgt_ids

    return {
        "inputs": inputs_arr,
        "labels": labels_arr,
        "puzzle_identifiers": np.zeros((N,), dtype=np.int32),
        "puzzle_indices": np.arange(N + 1, dtype=np.int32),
        "group_indices": np.arange(N + 1, dtype=np.int32),
    }


def save_split(split_name: str, split_data: dict, output_dir: str, seq_len: int, vocab_size: int):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    N = len(split_data["inputs"])
    metadata = PuzzleDatasetMetadata(
        pad_id=PAD_ID,
        ignore_label_id=IGNORE_LABEL_ID,
        blank_identifier_id=0,
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=1,
        total_groups=N,
        mean_puzzle_examples=1.0,
        total_puzzles=N,
        sets=["all"],
    )

    with open(os.path.join(split_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    for k, v in split_data.items():
        np.save(os.path.join(split_dir, f"all__{k}.npy"), v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["mbpp", "alpaca"], required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_pairs, test_pairs = get_raw_pairs(args.task, args.max_samples)
    tokenizer = build_or_load_tokenizer(train_pairs, args.output_dir, args.vocab_size)
    actual_vocab_size = len(tokenizer)

    train_data = encode_split(train_pairs, tokenizer, args.seq_len)
    test_data = encode_split(test_pairs, tokenizer, args.seq_len)

    save_split("train", train_data, args.output_dir, args.seq_len, actual_vocab_size)
    save_split("test", test_data, args.output_dir, args.seq_len, actual_vocab_size)

    with open(os.path.join(args.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    print(
        f"Built {args.task} dataset in {args.output_dir} "
        f"(train={len(train_pairs)}, test={len(test_pairs)}, seq_len={args.seq_len}, vocab_size={actual_vocab_size})"
    )


if __name__ == "__main__":
    main()