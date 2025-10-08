from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import torch
from torch.nn.attention import flex_attention
from transformers import PreTrainedTokenizerBase


@dataclass
class TokenizedConversation:
    token_ids: List[int]


def inject_checkpoint(ids: Sequence[int], stride: int, checkpoint_id: int) -> List[int]:
    """Insert `checkpoint_id` every `stride` tokens to form attention blocks."""
    if stride <= 0:
        raise ValueError("stride must be > 0")

    injected: List[int] = []
    if len(ids) <= stride:
        return list(ids)

    for start in range(0, len(ids), stride):
        stop = min(start + stride, len(ids))
        injected.extend(ids[start:stop])
        if stop < len(ids):
            injected.append(checkpoint_id)
    return injected


def tokenize_conversation(
    messages: Iterable[dict],
    tokenizer: PreTrainedTokenizerBase,
    stride: int,
) -> TokenizedConversation:
    """
    Apply chat template, then inject checkpoint tokens between template segments.

    The template introduces a sequence of special tokens separating roles and
    messages. We split on those boundaries to ensure checkpoints are only
    inserted inside conversational spans.
    """
    if tokenizer.cls_token is None:
        raise ValueError("Tokenizer must define a cls_token used as checkpoint.")

    ids = tokenizer.apply_chat_template(messages, tokenize=True)
    special_token_ids = set(tokenizer.all_special_ids)

    tokenized: List[int] = []
    special_indexes = [i for i, id in enumerate(ids) if id in special_token_ids]
    
    for start, end in zip(special_indexes[:-1], special_indexes[1:]):
        segment = ids[start:end]
        if all(id in special_token_ids for id in segment):
            tokenized.extend(segment)
            continue
        tokenized.extend(inject_checkpoint(segment, stride=stride, checkpoint_id=tokenizer.cls_token_id))
    last_segment = ids[special_indexes[-1]:]
    if all(id in special_token_ids for id in last_segment):
        tokenized.extend(last_segment)
    else:
        tokenized.extend(inject_checkpoint(last_segment, stride=stride, checkpoint_id=tokenizer.cls_token_id))

    return TokenizedConversation(tokenized)


def create_attention_mask(
    ids: torch.Tensor,
    special_token_ids: torch.Tensor,
    checkpoint_id: int,
    eos_token_id: int,
    block_size: int = 128,
) -> Tuple[flex_attention.BlockMask, Callable[..., bool]]:
    """
    Build a causal attention mask that respects checkpoint boundaries.

    Each checkpoint token opens attention across the preceding stride while
    still allowing access to earlier checkpoints and special tokens.
    """
    if ids.ndim != 1:
        raise ValueError("ids must be a 1D tensor")

    device = ids.device
    special_token_ids = special_token_ids.to(device)

    is_checkpoint = ids == checkpoint_id
    is_special = torch.isin(ids, special_token_ids)
    beacon_ids = is_checkpoint.long().cumsum(0) - is_checkpoint.long()
    docs = (ids == eos_token_id).long().cumsum(0)

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_beacon = bool((beacon_ids[kv_idx] == beacon_ids[q_idx]).item())
        same_doc = bool((docs[kv_idx] == docs[q_idx]).item())
        is_checkpoint_kv = bool(is_checkpoint[kv_idx].item())
        is_special_kv = bool(is_special[kv_idx].item())
        return causal and same_doc and (same_beacon or is_checkpoint_kv or is_special_kv)

    block_mask = flex_attention.create_block_mask(
        mask_mod, B=1, H=1, Q_LEN=ids.numel(), KV_LEN=ids.numel(), BLOCK_SIZE=block_size
    )
    return block_mask, mask_mod


def build_position_ids(ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Assign monotonically increasing position ids within each document."""
    if ids.ndim != 1:
        raise ValueError("ids must be a 1D tensor")

    position_ids = torch.zeros_like(ids, dtype=torch.long)
    current = 0

    for idx, token in enumerate(ids.tolist()):
        position_ids[idx] = current
        current += 1
        if token == eos_token_id:
            current = 0

    return position_ids.to(ids.device)
