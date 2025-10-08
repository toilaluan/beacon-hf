from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import torch
from torch.nn.attention import flex_attention
from transformers import PreTrainedTokenizerBase


@dataclass
class TokenizedConversation:
    token_ids: List[int]


def inject_checkpoint(ids: Sequence[int], stride: int, checkpoint_id: int, tokenizer: PreTrainedTokenizerBase) -> List[int]:
    """Insert `checkpoint_id` every `stride` tokens to form attention blocks."""
    if all(id in tokenizer.all_special_ids for id in ids):
        return ids

    injected: List[int] = []
    if len(ids) <= stride:
        return list(ids)

    for start in range(0, len(ids), stride):
        stop = min(start + stride, len(ids))
        injected.extend(ids[start:stop])
    return injected


def tokenize_conversation(
    messages: Iterable[dict],
    tokenizer: PreTrainedTokenizerBase,
    stride: int,
    end_doc_token: int = None,
    add_generation_prompt: bool = False,
    continue_final_message: bool = False,
) -> TokenizedConversation:
    """
    Apply chat template, then inject checkpoint tokens between template segments.

    The template introduces a sequence of special tokens separating roles and
    messages. We split on those boundaries to ensure checkpoints are only
    inserted inside conversational spans.
    """
    if tokenizer.cls_token is None:
        raise ValueError("Tokenizer must define a cls_token used as checkpoint.")

    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message)
    if end_doc_token is not None:
        ids.append(end_doc_token)
    special_token_ids = set(tokenizer.all_special_ids)

    tokenized: List[int] = []
    special_indexes = [i for i, id in enumerate(ids) if id in special_token_ids]
    
    for start, end in zip(special_indexes[:-1], special_indexes[1:]):
        segment = ids[start:end]
        if all(id in special_token_ids for id in segment):
            tokenized.extend(segment)
            continue
        tokenized.extend(inject_checkpoint(segment, stride=stride, checkpoint_id=tokenizer.cls_token_id, tokenizer=tokenizer))
    last_segment = ids[special_indexes[-1]:]
    if all(id in special_token_ids for id in last_segment):
        tokenized.extend(last_segment)
    else:
        tokenized.extend(inject_checkpoint(last_segment, stride=stride, checkpoint_id=tokenizer.cls_token_id, tokenizer=tokenizer))

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

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_beacon = beacon_ids[kv_idx] == beacon_ids[q_idx]
        same_doc = docs[kv_idx] == docs[q_idx]
        return causal & same_doc & (same_beacon | is_checkpoint[kv_idx] | is_special[kv_idx])

    block_mask = flex_attention.create_block_mask(
        mask_mod, B=1, H=1, Q_LEN=ids.numel(), KV_LEN=ids.numel(), BLOCK_SIZE=block_size
    )
    return block_mask, mask_mod


def build_position_ids(ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Assign monotonically increasing position ids within each document."""
    if ids.ndim != 1:
        raise ValueError("ids must be a 1D tensor")

    docs = (ids == eos_token_id).long().cumsum(0)
    unique_docs = docs.unique()
    position_ids = []
    for i in unique_docs:
        doc_size = (docs == i).sum()
        position_ids.append(torch.arange(doc_size, device=ids.device))
    position_ids = torch.cat(position_ids, dim=0)
    return position_ids
