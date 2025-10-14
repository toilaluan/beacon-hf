from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


def _discover_shards(dataset_path: Path) -> List[Tuple[Path, Path]]:
    dataset_path = Path(dataset_path)
    shard_bins = sorted(dataset_path.glob("shard_*.bin"))
    shard_indices = sorted(dataset_path.glob("shard_*.idx"))

    if not shard_bins:
        raise FileNotFoundError(f"No shards found in {dataset_path}")
    if len(shard_bins) != len(shard_indices):
        raise FileNotFoundError(
            f"Found {len(shard_bins)} bin shards but {len(shard_indices)} index files in {dataset_path}"
        )

    paired: List[Tuple[Path, Path]] = []
    for bin_path, idx_path in zip(shard_bins, shard_indices):
        if bin_path.stem != idx_path.stem:
            raise FileNotFoundError(f"Mismatched shard pair: {bin_path.name} != {idx_path.name}")
        paired.append((bin_path, idx_path))
    return paired


def _load_index(index_path: Path) -> dict:
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_shard(shard_path: Path, index_path: Path) -> tuple[np.memmap, dict]:
    tokens = np.memmap(shard_path, dtype=np.uint32, mode="r")
    index = _load_index(index_path)
    return tokens, index


def _deterministic_shuffle(items: Iterable[dict], seed: int) -> List[dict]:
    shuffled = list(items)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def _floor_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return (value // multiple) * multiple


def insert_checkpoints(token_ids: Sequence[int], stride: int, ckpt_id: int) -> List[int]:
    """
    Insert checkpoint tokens every `stride` tokens: [.. stride .. CKPT .. stride .. CKPT ..]
    """
    if stride <= 0:
        return list(token_ids)

    with_ckpts: List[int] = []
    for start in range(0, len(token_ids), stride):
        end = min(start + stride, len(token_ids))
        with_ckpts.extend(token_ids[start:end])
        if end < len(token_ids):
            with_ckpts.append(ckpt_id)
    return with_ckpts


def distributed_sequence_generator(
    dataset_path: Path,
    sequence_length: int,
    ckpt_id: int,
    ckpt_stride: int,
    bos_id: int,
    *,
    local_rank: int,
    world_size: int,
    base_seed: int = 0,
    shard_pairs: Optional[Sequence[Tuple[Path, Path]]] = None,
) -> Generator[torch.Tensor, None, None]:
    """
    Yield sequences of length `sequence_length + 1` (causal LM targets) for the given rank.
    """
    dataset_path = Path(dataset_path)
    if shard_pairs is None:
        shard_pairs = _discover_shards(dataset_path)
    else:
        shard_pairs = list(shard_pairs)
        if not shard_pairs:
            raise FileNotFoundError(f"No shards provided for dataset at {dataset_path}")
    if not (0 <= local_rank < world_size):
        raise ValueError("local_rank must be in [0, world_size)")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")

    sample_span = sequence_length + 1  # include next token for labels

    epoch = 0

    while True:
        shard_order: List[int] = list(range(len(shard_pairs)))
        if len(shard_order) > 1:
            epoch_seed = (base_seed << 32) ^ (epoch * 1315423911)
            epoch_rng = random.Random(epoch_seed)
            epoch_rng.shuffle(shard_order)

        for relative_idx, shard_idx in enumerate(shard_order):
            bin_path, idx_path = shard_pairs[shard_idx]
            tokens, index = _load_shard(bin_path, idx_path)
            documents = index["documents"]

            seed = (base_seed << 32) ^ (epoch * 1315423911) ^ (relative_idx * 2654435761)
            shuffled_docs = _deterministic_shuffle(documents, seed)

            processing_rank = 0

            start_selected_docs = []
            end_selected_docs = []
            total_selected_length = 0

            lengths = []

            for doc in shuffled_docs:
                start = int(doc["start"])
                end = int(doc["end"])
                raw_doc = tokens[start:end]

                if raw_doc.size == 0:
                    processing_rank = (processing_rank + 1) % world_size
                    continue

                if processing_rank == local_rank:
                    start_selected_docs.append(start)
                    end_selected_docs.append(end)
                    lengths.append(end - start)
                    total_selected_length += end - start
                    if total_selected_length >= sample_span:
                        items = []
                        for start, end in zip(start_selected_docs, end_selected_docs):
                            item = tokens[start:end]
                            # item = [bos_id] + item
                            item = insert_checkpoints(item, ckpt_stride, ckpt_id)
                            items.extend(item)
                        # print(lengths)
                        yield torch.tensor(items[:sample_span], dtype=torch.long)
                        lengths.clear()
                        start_selected_docs.clear()
                        end_selected_docs.clear()
                        total_selected_length = 0

                processing_rank = (processing_rank + 1) % world_size

        epoch += 1


class DistributedTokenDataset(IterableDataset):
    """
    IterableDataset view over cached token shards for a specific distributed rank.

    Each iteration yields a tensor of length `sequence_length + 1`. The dataset
    is stateless across workers; pass deterministic seeds per process to keep
    ordering aligned across ranks.
    """

    def __init__(
        self,
        dataset_path: Path,
        sequence_length: int,
        ckpt_id: int,
        ckpt_stride: int,
        bos_id: int,
        *,
        local_rank: int,
        world_size: int,
        base_seed: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self._shard_pairs = _discover_shards(self.dataset_path)
        self.metadata = _load_index(self._shard_pairs[0][1]).get("metadata", {})
        self.sequence_length = sequence_length
        self.ckpt_id = ckpt_id
        self.ckpt_stride = ckpt_stride
        self.bos_id = bos_id
        self.local_rank = local_rank
        self.world_size = world_size
        self.base_seed = base_seed

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        return distributed_sequence_generator(
            self.dataset_path,
            self.sequence_length,
            ckpt_id=self.ckpt_id,
            ckpt_stride=self.ckpt_stride,
            bos_id=self.bos_id,
            local_rank=self.local_rank,
            world_size=self.world_size,
            base_seed=self.base_seed,
            shard_pairs=self._shard_pairs,
        )

if __name__ == "__main__":
    dataset = DistributedTokenDataset(
        dataset_path=Path("tokenized_data"),
        sequence_length=1024,
        local_rank=0,
        world_size=1,
        base_seed=0,
    )
    for batch in dataset:
        print(batch.shape)