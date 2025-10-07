#!/usr/bin/env python3
# train_checkpoint_attention.py
# Finetunes Qwen2.5-0.5B-Instruct with a checkpoint-token FlexAttention mask (DDP + LoRA).

from __future__ import annotations
import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Iterator, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import Shuffler  # lightweight shuffler for iterable
from torch.cuda.amp import autocast, GradScaler
from torch.nn.attention import flex_attention

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

# ----------------------------
# Utilities
# ----------------------------

def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# Checkpoint-token injection
# ----------------------------

def inject_checkpoint(ids: List[int], stride: int, checkpoint_id: int) -> List[int]:
    injected = []
    if len(ids) < stride:
        return ids
    for i in range(0, len(ids), stride):
        injected.extend(ids[i : i + stride])
        if i + stride < len(ids):
            injected.append(checkpoint_id)
    return injected

def process_item(
    tokenizer: AutoTokenizer,
    item: Dict[str, Any],
    stride: int,
    checkpoint_id: int,
    max_len: int,
) -> List[int]:
    """
    1) Convert Tulu 'messages' to token ids via chat template
    2) Inject <|checkpoint|> every `stride` tokens within segments delimited by special tokens
    3) Truncate to max_len
    """
    messages = item["messages"]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

    # Find special-token boundaries, then inject inside each span
    special_ids = set(tokenizer.all_special_ids)
    special_idx = [i for i, tok in enumerate(ids) if tok in special_ids]
    # Make sure we cover the leading/trailing spans, too
    if not special_idx or special_idx[0] != 0:
        special_idx = [0] + special_idx
    if special_idx[-1] != len(ids):
        special_idx = special_idx + [len(ids)]

    out: List[int] = []
    for s, e in zip(special_idx[:-1], special_idx[1:]):
        span = ids[s:e]
        span_injected = inject_checkpoint(span, stride=stride, checkpoint_id=checkpoint_id)
        out.extend(span_injected)

    # Truncate (we'll pad in collate)
    return out[:max_len]

# ----------------------------
# Batched FlexAttention BlockMask
# ----------------------------

def make_blockmask_batched(
    input_ids: torch.Tensor,                 # [B, L]
    special_token_ids: torch.Tensor,        # [S]
    checkpoint_id: int,
    block_size: int = 128,
) -> flex_attention.BlockMask:
    """
    Implements your rule:
      causal AND (same_beacon OR kv_is_beacon OR kv_is_special)
    where 'beacons' are <|checkpoint|> tokens inserted every stride,
    and 'special' = tokenizer.all_special_ids (pad/eos/bos/role tokens/etc.).
    """
    device = input_ids.device
    B, L = input_ids.shape

    is_beacons = (input_ids == checkpoint_id)                      # [B, L] bool
    is_specials = torch.isin(input_ids, special_token_ids)         # [B, L] bool
    # beacon_ids counts how many beacons have appeared up to and including idx,
    # then subtract 1 if current position itself is a beacon -> "previous beacon group"
    beacon_ids = is_beacons.long().cumsum(dim=1) - is_beacons.long()  # [B, L] int

    # capture into closure (avoid re-alloc per call)
    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_beacon = beacon_ids[b, kv_idx] == beacon_ids[b, q_idx]
        return causal & (same_beacon | is_beacons[b, kv_idx] | is_specials[b, kv_idx])

    # H can be None (broadcast across heads); set B to batch for per-sample masking
    return flex_attention.create_block_mask(
        mask_mod,
        B=B,
        H=None,
        Q_LEN=L,
        KV_LEN=L,
        BLOCK_SIZE=block_size,
    )

# ----------------------------
# Collator for streaming iterable
# ----------------------------

@dataclass
class CollatorConfig:
    max_len: int = 2048
    stride: int = 16

class TuluCollator:
    def __init__(self, tokenizer: AutoTokenizer, cfg: CollatorConfig):
        self.tok = tokenizer
        self.cfg = cfg
        if self.tok.pad_token_id is None:
            # Qwen2.* uses no pad by default; map pad->eos for safe padding
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Convert/Inject per item
        ids_list: List[List[int]] = []
        for item in batch:
            ids = process_item(
                tokenizer=self.tok,
                item=item,
                stride=self.cfg.stride,
                checkpoint_id=self.tok.cls_token_id,
                max_len=self.cfg.max_len,
            )
            # Need at least 2 tokens to make (inputs, labels)
            if len(ids) < 2:
                continue
            ids_list.append(ids)

        if not ids_list:  # degenerate batch; return a small dummy to avoid crash
            pad = self.tok.pad_token_id
            dummy = torch.tensor([[pad, pad]], dtype=torch.long)
            return {"input_ids": dummy[:, :-1], "labels": dummy[:, 1:]}

        # Build (inputs, labels), pad
        max_len = max(len(x) for x in ids_list)
        pad_id = self.tok.pad_token_id

        input_ids = []
        labels = []
        for ids in ids_list:
            # Shift LM: inputs = ids[:-1], labels = ids[1:]
            x_in = ids[:-1]
            y   = ids[1:]

            # Mask labels where target is the checkpoint token (don't learn to predict it)
            y = [(-100 if t == self.tok.cls_token_id else t) for t in y]

            # Pad to max_len-1 (because we shifted by 1)
            need = (max_len - 1) - len(x_in)
            if need > 0:
                x_in = x_in + [pad_id] * need
                y    = y    + [-100]   * need

            input_ids.append(torch.tensor(x_in, dtype=torch.long))
            labels.append(torch.tensor(y, dtype=torch.long))

        input_ids = torch.stack(input_ids, dim=0)  # [B, L]
        labels    = torch.stack(labels, dim=0)     # [B, L]

        return {"input_ids": input_ids, "labels": labels}

# ----------------------------
# Eval / sample generation (greedy, mask-aware)
# ----------------------------

@torch.no_grad()
def masked_greedy_generate(
    model,
    tokenizer,
    prompt_messages: List[Dict[str, str]],
    max_new_tokens: int = 64,
    stride: int = 16,
    block_size: int = 128,
    device: str = "cuda",
) -> str:
    # Build prompt ids + checkpoint injection
    ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    special_ids = set(tokenizer.all_special_ids)
    special_idx = [i for i, tok in enumerate(ids) if tok in special_ids]
    if not special_idx or special_idx[0] != 0:
        special_idx = [0] + special_idx
    if special_idx[-1] != len(ids):
        special_idx = special_idx + [len(ids)]
    out = []
    for s, e in zip(special_idx[:-1], special_idx[1:]):
        out.extend(inject_checkpoint(ids[s:e], stride, tokenizer.cls_token_id))
    input_ids = torch.tensor(out, device=device, dtype=torch.long).unsqueeze(0)  # [1, L]

    for _ in range(max_new_tokens):
        # Build BlockMask for current length
        special_token_ids = torch.tensor(tokenizer.all_special_ids, device=device, dtype=torch.long)
        bm = make_blockmask_batched(input_ids, special_token_ids, tokenizer.cls_token_id, block_size)

        with autocast(device_type="cuda", dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)):
            logits = model(input_ids=input_ids, attention_mask=bm).logits  # [1, L, V]
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1))
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)

# ----------------------------
# Training
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", type=str, default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)  # per-GPU
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="./ckpt_lora")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--block_size", type=int, default=128)
    args = parser.parse_args()

    setup_ddp()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK','0'))}"

    set_seed(args.seed + rank)

    # --- Tokenizer with checkpoint special token
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # Define checkpoint token (CLS slot is convenient and shows up in all_special_ids)
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({"cls_token": "<|checkpoint|>"})
    checkpoint_id = tokenizer.cls_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Base model (flex attention), resize for new token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="flex_attention",
        device_map=None,
    )
    model.resize_token_embeddings(len(tokenizer))

    # --- LoRA PEFT
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # common in Qwen2.*
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    model.to(device)
    model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", "0"))], find_unused_parameters=False)

    # --- Dataset (streaming + shard per-rank)
    train_stream = load_dataset(args.dataset, split="train", streaming=True)
    # Some Tulu mixtures have only train; we’ll just stream train.
    train_stream = train_stream.shard(num_shards=world_size, index=rank, contiguous=True)
    train_stream = Shuffler(train_stream, buffer_size=2048, seed=args.seed + rank)

    collate = TuluCollator(tokenizer, CollatorConfig(max_len=args.max_len, stride=args.stride))

    # Iterable streaming loader
    train_loader = DataLoader(
        train_stream,
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=0,  # streaming + Python transforms; keep simple
        pin_memory=True,
    )

    # Get one eval sample (rank 0 only)
    eval_prompt = None
    if is_main():
        for ex in load_dataset(args.dataset, split="train", streaming=True):
            if isinstance(ex.get("messages"), list) and len(ex["messages"]) > 0:
                eval_prompt = ex["messages"]
                break

    if dist.is_initialized():
        # broadcast "has eval" flag first
        flag = torch.tensor([1 if eval_prompt is not None else 0], device=device)
        dist.broadcast(flag, src=0)
        if flag.item() == 1:
            # crude broadcast via serialize to CPU then scatter length + bytes
            payload = json.dumps(eval_prompt).encode("utf-8") if is_main() else b""
            size = torch.tensor([len(payload)], device=device, dtype=torch.long)
            dist.broadcast(size, src=0)
            buf = torch.empty(size.item(), dtype=torch.uint8, device=device)
            if is_main():
                buf.copy_(torch.tensor(list(payload), dtype=torch.uint8, device=device))
            dist.broadcast(buf, src=0)
            if not is_main():
                eval_prompt = json.loads(bytes(buf.tolist()).decode("utf-8"))
        else:
            eval_prompt = None

    # --- Optimizer/Scheduler
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

    scaler = GradScaler(enabled=(dtype == torch.float16))

    # --- Training loop
    model.train()
    step = 0
    start_time = time.time()

    # Materialize special ids tensor once per step on device
    special_token_ids_tensor = torch.tensor(tokenizer.all_special_ids, device=device, dtype=torch.long)

    data_iter = iter(train_loader)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)  # [B, L]
        labels    = batch["labels"].to(device, non_blocking=True)     # [B, L]

        # Build BlockMask for the batch (per your rule)
        blockmask = make_blockmask_batched(
            input_ids=input_ids,
            special_token_ids=special_token_ids_tensor,
            checkpoint_id=checkpoint_id,
            block_size=args.block_size,
        )

        with autocast(device_type="cuda", dtype=dtype):
            outputs = model(input_ids=input_ids, attention_mask=blockmask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        sched.step()

        step += 1

        # Logging
        if step % args.log_every == 0 and is_main():
            tok_s = (input_ids.numel() * args.log_every * world_size) / (time.time() - start_time + 1e-9)
            print(f"[step {step}] loss={loss.item():.4f}  tokens/s≈{tok_s:,.0f}")
            start_time = time.time()

        # Sample validation
        if step % args.val_every == 0 and eval_prompt is not None and is_main():
            model.eval()
            try:
                text = masked_greedy_generate(
                    model.module,  # unwrap DDP
                    tokenizer,
                    prompt_messages=eval_prompt,
                    max_new_tokens=64,
                    stride=args.stride,
                    block_size=args.block_size,
                    device=device,
                )
                print("\n=== Validation sample (truncated) ===")
                print(text[:800])
                print("=====================================\n")
            finally:
                model.train()

    # Save PEFT adapter (rank 0)
    if is_main():
        os.makedirs(args.save_dir, exist_ok=True)
        model.module.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print(f"Saved LoRA adapter + tokenizer to: {args.save_dir}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
