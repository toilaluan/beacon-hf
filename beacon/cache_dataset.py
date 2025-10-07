from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from .data import tokenize_conversation


@dataclass
class ShardWriter:
    output_dir: Path
    tokens_per_shard: int
    metadata: Dict[str, str]

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_idx = 0
        self._token_buffer: List[int] = []
        self._documents: List[Dict[str, int]] = []
        self.total_tokens = 0
        self.total_documents = 0

    def add_document(self, token_ids: Iterable[int]) -> None:
        start = len(self._token_buffer)
        tokens = list(token_ids)
        if not tokens:
            return

        self._token_buffer.extend(tokens)
        end = len(self._token_buffer)
        self._documents.append({"start": start, "end": end})
        self.total_tokens += len(tokens)
        self.total_documents += 1

        if len(self._token_buffer) >= self.tokens_per_shard:
            self.flush()

    def flush(self, *, force: bool = False) -> None:
        if not self._token_buffer:
            return
        if not force and len(self._token_buffer) < self.tokens_per_shard:
            return

        shard_tokens = np.asarray(self._token_buffer, dtype=np.uint32)
        shard_path = self.output_dir / f"shard_{self.shard_idx:05d}.bin"
        index_path = self.output_dir / f"shard_{self.shard_idx:05d}.idx"

        shard_tokens.tofile(shard_path)

        index_payload = {
            "documents": [
                {"start": int(doc["start"]), "end": int(doc["end"])}
                for doc in self._documents
            ],
            "num_tokens": int(shard_tokens.size),
            "metadata": self.metadata,
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_payload, f)

        print(
            f"Wrote shard {self.shard_idx:05d} "
            f"({len(self._documents)} docs, {shard_tokens.size} tokens)"
        )

        self.shard_idx += 1
        self._token_buffer.clear()
        self._documents.clear()

    def finalize(self) -> None:
        self.flush(force=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache chat dataset into token shards.")
    parser.add_argument("--dataset", default="allenai/tulu-3-sft-mixture", help="HF dataset path")
    parser.add_argument("--split", default="train", help="Dataset split name")
    parser.add_argument("--output-dir", default="tokenized_data", help="Where to store shards")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct", help="Tokenizer name")
    parser.add_argument("--checkpoint-token", default="<|checkpoint|>", help="Special token for checkpoints")
    parser.add_argument("--stride", type=int, default=16, help="Interval between checkpoint tokens")
    parser.add_argument("--tokens-per-shard", type=int, default=16 * 1024 * 32, help="Target tokens per shard")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on processed samples")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming dataset loading")
    parser.add_argument("--dataset-trust-remote-code", action="store_true", help="Allow remote code for dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    print(f"Loading tokenizer {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if tokenizer.cls_token is None or tokenizer.cls_token == "":
        tokenizer.add_special_tokens({"cls_token": args.checkpoint_token})
    elif tokenizer.cls_token != args.checkpoint_token:
        # Ensure downstream scripts use the same literal for consistency.
        print(f"Tokenizer already defines cls_token={tokenizer.cls_token}")

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "tokenizer": args.tokenizer,
        "checkpoint_token": tokenizer.cls_token,
        "stride": str(args.stride),
    }

    print(f"Preparing dataset {args.dataset}:{args.split} (streaming={args.streaming})")
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=args.streaming,
        trust_remote_code=args.dataset_trust_remote_code,
    )

    writer = ShardWriter(
        output_dir=output_dir,
        tokens_per_shard=args.tokens_per_shard,
        metadata=metadata,
    )

    for idx, sample in enumerate(dataset):
        if args.max_samples is not None and idx >= args.max_samples:
            break

        messages = sample.get("messages")
        if not messages:
            continue

        conversation = tokenize_conversation(messages, tokenizer=tokenizer, stride=args.stride)
        writer.add_document(conversation.token_ids)

        if (idx + 1) % 1000 == 0:
            print(
                f"Processed {idx + 1} samples "
                f"({writer.total_documents} docs, {writer.total_tokens} tokens written so far)"
            )

    writer.finalize()
    print(
        f"Finished: {writer.total_documents} documents, {writer.total_tokens} tokens, "
        f"{writer.shard_idx} shards -> {output_dir}"
    )


if __name__ == "__main__":
    main()
