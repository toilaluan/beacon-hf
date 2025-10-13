from datasets import load_dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.attention import flex_attention
from typing import Tuple, Callable

def create_attention_mask(
    ids: torch.Tensor,
    special_token_ids: torch.Tensor,
    checkpoint_id: int,
    eos_token_id: int | None,
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

    # Beacon groups: each non-checkpoint belongs to the group of the most recent checkpoint (before it)
    beacon_ids = is_checkpoint.long().cumsum(0) - is_checkpoint.long()

    # Doc groups: if no eos_token_id is provided or it never appears, treat entire seq as one doc
    if eos_token_id is None or (ids == eos_token_id).sum() == 0:
        docs = torch.zeros_like(ids)
    else:
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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Add checkpoint special token
tokenizer.add_special_tokens({"cls_token": "<|checkpoint|>"})
ckpt_id = tokenizer.cls_token_id

# Helpful: define eos if present; else None (single-doc behavior)
eos_id = tokenizer.eos_token_id

# Encode and insert checkpoints every N tokens (training-style)
input_text = "capital of france is paris, capital of italy is rome, capital of germany is berlin, capital of vietnam is hanoi"
ids_plain = tokenizer.encode(input_text)
ckpt_stride = 8
updated_ids = []
for i in range(0, len(ids_plain), ckpt_stride):
    updated_ids.extend(ids_plain[i:i+ckpt_stride])
    updated_ids.append(ckpt_id)
input_ids = torch.tensor(updated_ids, device="cuda")

print(tokenizer.decode(input_ids.tolist()))

# Load model *after* adding special tokens, then resize embeddings
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device_map="cuda",
    dtype=torch.bfloat16,
    attn_implementation="flex_attention",
)
model.resize_token_embeddings(len(tokenizer))

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-4)

train_ids = input_ids[:-1].clone()
labels    = input_ids[1:].clone()

special_ids = set(tokenizer.all_special_ids)
special_ids.add(ckpt_id)  # ensure our new token is treated as special
special_ids_tensor = torch.tensor(sorted(special_ids), device="cuda")

attn_mask, mask_mod = create_attention_mask(
    train_ids,
    special_token_ids=special_ids_tensor,
    checkpoint_id=ckpt_id,
    eos_token_id=eos_id,
)

print(attn_mask)
print(mask_mod)

# ---------- Fixed: generation helpers ----------

def tokens_since_last_ckpt(seq: torch.Tensor, ckpt_token_id: int) -> int:
    """Count consecutive non-ckpt tokens from the end back to the most recent ckpt (0 if last is ckpt)."""
    n = 0
    for t in range(seq.numel() - 1, -1, -1):
        if int(seq[t].item()) == int(ckpt_token_id):
            break
        n += 1
    return n

def maybe_append_ckpt(seq: torch.Tensor, ckpt_token_id: int, stride: int) -> torch.Tensor:
    """Append a checkpoint if we've already produced `stride` tokens since the last checkpoint."""
    if tokens_since_last_ckpt(seq, ckpt_token_id) >= stride:
        seq = torch.cat([seq, torch.tensor([ckpt_token_id], device=seq.device)], dim=0)
    return seq

# -------------- Train --------------
for step in range(1000):
    optimizer.zero_grad()
    labels[(labels == tokenizer.cls_token_id).nonzero(as_tuple=True)] = -100
    outputs = model(train_ids.unsqueeze(0), attention_mask=attn_mask)
    logits = outputs.logits[0] 
    valid = (labels != -100)
    nll = torch.nn.functional.cross_entropy(logits[valid], labels[valid], reduction='none')
    # Print around each checkpoint boundary
    loss = nll.mean()
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        for i, t in enumerate(train_ids[:-1]):
            if t.item() == ckpt_id and i+1 < labels.numel():
                print("after CKPT @", i, "  next token:", tokenizer.decode([labels[i]]), "  nll:", nll[(valid.nonzero().flatten()==i).nonzero(as_tuple=True)[0]].item())
        print(loss.item())

    # -------------- Validate / Generate --------------
    if step % 100 == 0:
        # seed prompt
        val_ids = input_ids[:3].clone()

        # align to training schedule: if exactly a stride since last ckpt, append one
        val_ids = maybe_append_ckpt(val_ids, ckpt_id, ckpt_stride)

        print("seed:", tokenizer.decode(val_ids.tolist()))
        print("--------------------------------")

        max_new_tokens = 16
        for _ in range(max_new_tokens):
            # enforce checkpoint cadence before generating the next token
            if tokens_since_last_ckpt(val_ids, ckpt_id) >= ckpt_stride:
                val_ids = torch.cat([val_ids, torch.tensor([ckpt_id], device=val_ids.device)], dim=0)
                # do not sample a token on this turn; we just inserted a checkpoint
                continue

            mask, _ = create_attention_mask(
                val_ids,
                special_token_ids=special_ids_tensor,
                checkpoint_id=ckpt_id,
                eos_token_id=eos_id,
            )
            with torch.no_grad():
                out = model(val_ids.unsqueeze(0), attention_mask=mask)
                next_id = out.logits[:, -1, :].argmax(dim=-1)

            val_ids = torch.cat([val_ids, next_id], dim=0)
            print(tokenizer.decode(val_ids.tolist()))

        print("--------------------------------")
