from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from beacon.dist_dataloader import DistributedTokenDataset
from beacon.data import (
    build_position_ids,
    create_attention_mask,
    tokenize_conversation,
)
from beacon.val_sample import MESSAGES

import wandb
from peft import LoraConfig, get_peft_model


def tokens_since_last_checkpoint(seq: torch.Tensor, checkpoint_id: int) -> int:
    """Count non-checkpoint tokens from the tail back to the most recent checkpoint."""
    count = 0
    for idx in range(seq.numel() - 1, -1, -1):
        if int(seq[idx].item()) == int(checkpoint_id):
            break
        count += 1
    return count


def maybe_append_checkpoint(seq: torch.Tensor, checkpoint_id: int, stride: int) -> torch.Tensor:
    """Append a checkpoint token if `stride` tokens have elapsed since the last checkpoint."""
    if tokens_since_last_checkpoint(seq, checkpoint_id) >= stride:
        checkpoint = torch.tensor([checkpoint_id], dtype=seq.dtype, device=seq.device)
        seq = torch.cat([seq, checkpoint], dim=0)
    return seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed LoRA fine-tuning with beacon masking.")
    parser.add_argument("--tokenized-dataset", type=Path, required=True, help="Path to cached token shards.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model name or path.")
    parser.add_argument("--sequence-length", type=int, default=1024, help="Sequence length (excl. next token).")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Sequences per optimizer micro-step.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Micro-steps per optimizer step.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Total optimizer steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="AdamW betas.")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Scheduler warmup steps.")
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16", "fp32"), default="bf16", help="AMP precision.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between loss logging.")
    parser.add_argument("--sample-interval", type=int, default=100, help="Steps between text sampling.")
    parser.add_argument("--sample-max-new-tokens", type=int, default=16, help="Maximum generation length.")
    parser.add_argument("--data-seed", type=int, default=1234, help="Deterministic data shuffle seed.")
    parser.add_argument("--doc-multiple-of", type=int, default=1, help="Floor documents to multiple of n tokens.")
    parser.add_argument("--checkpoint-token", default="<|checkpoint|>", help="CLS token used for beacons.")
    parser.add_argument("--checkpoint-stride", type=int, default=16, help="Stride between checkpoint tokens.")
    parser.add_argument("--overfit", action="store_true", help="Enable single-sample overfitting mode.")
    parser.add_argument(
        "--overfit-prefix-tokens",
        type=int,
        default=8,
        help="Number of tokens from the overfit sample to use as the generation prompt.",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Module names to wrap with LoRA adapters.",
    )
    parser.add_argument("--wandb-project", default="beacon-hf", help="Weights & Biases project name.")
    parser.add_argument("--wandb-run-name", default=None, help="Optional wandb run name.")
    parser.add_argument(
        "--sample-user-content",
        default="Provide a concise summary of a long research article and answer a follow-up question.",
        help="User message content for periodic sampling.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory to save checkpoints.")
    return parser.parse_args()


def setup_distributed() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        return rank, world_size, local_rank
    return 0, 1, 0


def setup_tokenizer(model_name: str, checkpoint_token: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.cls_token is None or tokenizer.cls_token != checkpoint_token:
        tokenizer.add_special_tokens({"cls_token": checkpoint_token})
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an eos_token for causal LM training.")

    assert tokenizer.pad_token_id != tokenizer.eos_token_id, "pad_token_id and eos_token_id should be different"
    return tokenizer


def prepare_model(
    model_name: str,
    tokenizer: AutoTokenizer,
    device: torch.device,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Iterable[str],
) -> torch.nn.Module:
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cuda",
        attn_implementation="flex_attention",
    )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # lora_config = LoraConfig(
    #     r=lora_r,
    #     lora_alpha=lora_alpha,
    #     lora_dropout=lora_dropout,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules=list(lora_target_modules),
    # )
    # model = get_peft_model(model, lora_config)
    model.to(device)
    model.train()
    model = torch.compile(model)
    return model


def initialize_wandb(args: argparse.Namespace, is_main_process: bool) -> Optional["wandb.sdk.wandb_run.Run"]:
    if wandb is None or not is_main_process:
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model": args.model_name,
            "sequence_length": args.sequence_length,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "doc_multiple_of": args.doc_multiple_of,
            "mixed_precision": args.mixed_precision,
        },
    )
    return run


def average_across_ranks(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return value
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.AVG)
    return value


def generate_sample(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    *,
    stride: int,
    max_new_tokens: int,
    device: torch.device,
    messages: Optional[list[dict]] = None,
    prefix_token_ids: Optional[torch.Tensor] = None,
) -> str:
    was_training = model.training
    model.eval()

    with torch.no_grad():
        if prefix_token_ids is not None:
            generated = prefix_token_ids.to(device=device, dtype=torch.long)
        else:
            if messages is None:
                raise ValueError("Either `messages` or `prefix_token_ids` must be provided.")
            conv = tokenize_conversation(
                messages,
                tokenizer=tokenizer,
                stride=stride,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            generated = torch.tensor(conv.token_ids, dtype=torch.long, device=device)

        generated = maybe_append_checkpoint(generated, tokenizer.cls_token_id, stride)
        generated = generated.unsqueeze(0)
        prefix_len = generated.size(1)

        special_token_ids = torch.tensor(
            sorted(set(tokenizer.all_special_ids) | {tokenizer.cls_token_id}),
            device=device,
        )

        print(tokenizer.decode(generated[0].tolist()))
        print("--------------------------------")

        for i in range(max_new_tokens):
            if tokens_since_last_checkpoint(generated[0], tokenizer.cls_token_id) >= stride:
                ckpt = torch.full((1, 1), tokenizer.cls_token_id, dtype=generated.dtype, device=device)
                generated = torch.cat([generated, ckpt], dim=1)
                print(f"[ckpt] {tokenizer.cls_token}")
                continue

            attn_mask, _ = create_attention_mask(
                generated[0],
                special_token_ids=special_token_ids,
                checkpoint_id=tokenizer.cls_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            pos_ids = build_position_ids(generated[0], tokenizer.eos_token_id).unsqueeze(0)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                out = model(
                    generated,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                )
            next_id = out.logits[:, -1, :].argmax(dim=-1)

            generated = torch.cat([generated, next_id.unsqueeze(0)], dim=1)
            print(f"[{i}] {tokenizer.decode(next_id.tolist())}")

            if next_id.item() == tokenizer.eos_token_id:
                break

        new_tokens = generated[0, prefix_len:].tolist()
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if was_training:
        model.train()
    return text



def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)
    is_main_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = setup_tokenizer(args.model_name, args.checkpoint_token)
    model = prepare_model(
        args.model_name,
        tokenizer,
        device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )

    print(model.get_input_embeddings().weight.shape)
    print(tokenizer.vocab_size, tokenizer.cls_token_id, tokenizer.eos_token_id)

    model = DDP(model, device_ids=[local_rank])

    dataset = DistributedTokenDataset(
        dataset_path=args.tokenized_dataset,
        sequence_length=args.sequence_length,
        local_rank=local_rank,
        world_size=world_size,
        base_seed=args.data_seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    amp_enabled = args.mixed_precision != "fp32"
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = GradScaler(enabled=args.mixed_precision == "fp16")

    special_token_ids = torch.tensor(
        sorted(set(tokenizer.all_special_ids) | {tokenizer.cls_token_id}),
        device=device,
    )
    total_sequences = args.micro_batch_size * args.gradient_accumulation_steps

    run = initialize_wandb(args, is_main_process)

    global_step = 0
    data_iter = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)

    overfit_batch: Optional[torch.Tensor] = None
    overfit_prompt: Optional[torch.Tensor] = None
    if args.overfit:
        try:
            first_batch = next(data_iter)
        except StopIteration as exc:
            raise RuntimeError("Tokenized dataset is empty; cannot enable overfit mode.") from exc

        if first_batch.dim() == 1:
            first_batch = first_batch.unsqueeze(0)

        overfit_sequence = first_batch[0][: args.sequence_length + 1].clone()
        if overfit_sequence.numel() <= 1:
            raise ValueError("Overfit sample must contain at least two tokens.")

        overfit_sequence = overfit_sequence.to(device=device, dtype=torch.long)
        overfit_batch = overfit_sequence.unsqueeze(0)

        overfit_inputs = overfit_sequence[:-1]
        prompt_len = min(
            overfit_inputs.size(0),
            max(args.overfit_prefix_tokens, 1),
        )
        overfit_prompt = overfit_inputs[:prompt_len].clone()

        if is_main_process:
            print("----- Overfit sample (decoded) -----")
            print(tokenizer.decode(overfit_sequence.to("cpu").tolist()))
            print("----- Overfit prompt (decoded) -----")
            print(tokenizer.decode(overfit_prompt.to("cpu").tolist()))
            print("-----------------------------------")

    # reset iterator so regular training still fetches batches from the start
    if not args.overfit:
        data_iter = iter(dataloader)

    while global_step < args.max_steps:
        step_start = time.time()
        step_loss_sum = 0.0
        processed_sequences = 0

        for _ in range(args.gradient_accumulation_steps):
            try:
                if args.overfit:
                    batch = overfit_batch
                else:
                    batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            if batch.dim() == 1:
                batch = batch.unsqueeze(0)

            for sequence in batch:
                sequence = sequence[: args.sequence_length + 1]
                sequence = sequence.to(device=device, dtype=torch.long)
                if sequence.numel() <= 1:
                    continue

                inputs = sequence[:-1].clone()
                labels = sequence[1:].clone()

                attention_mask, _ = create_attention_mask(
                    inputs,
                    special_token_ids=special_token_ids,
                    checkpoint_id=tokenizer.cls_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                position_ids = build_position_ids(inputs, tokenizer.eos_token_id)

                labels[labels == tokenizer.cls_token_id] = -100

                with autocast(enabled=amp_enabled, dtype=amp_dtype, device_type=device.type):
                    outputs = model(
                        inputs.unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids.unsqueeze(0),
                    )
                logits = outputs.logits
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

                step_loss_sum += loss.item()
                processed_sequences += 1
                scaled_loss = loss / (args.gradient_accumulation_steps * batch.size(0))
                scaler.scale(scaled_loss).backward()

        if args.max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        global_step += 1

        avg_loss = step_loss_sum / max(processed_sequences, 1)
        loss_tensor = torch.tensor(avg_loss, device=device)
        loss_tensor = average_across_ranks(loss_tensor, world_size)
        avg_loss = loss_tensor.item()

        if is_main_process and (global_step % args.log_interval == 0 or global_step == 1):
            duration = time.time() - step_start
            tokens_per_step = args.sequence_length * total_sequences * world_size / max(duration, 1e-6)
            log_payload = {
                "train/step": global_step,
                "train/loss": avg_loss,
                "train/lr": scheduler.get_last_lr()[0],
                "train/tokens_per_sec": tokens_per_step,
            }
            if run is not None:
                wandb.log(log_payload, step=global_step)
            print(
                f"[step {global_step}] loss={avg_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} "
                f"{tokens_per_step:.1f} tok/s"
            )

        if is_main_process and global_step % args.sample_interval == 0:
            with torch.no_grad():
                if args.overfit and overfit_prompt is not None:
                    sample_text = generate_sample(
                        model.module,
                        tokenizer,
                        stride=args.checkpoint_stride,
                        max_new_tokens=args.sample_max_new_tokens,
                        device=device,
                        prefix_token_ids=overfit_prompt,
                    )
                else:
                    sample_text = generate_sample(
                        model.module,
                        tokenizer,
                        stride=args.checkpoint_stride,
                        max_new_tokens=args.sample_max_new_tokens,
                        device=device,
                        messages=MESSAGES,
                    )
            if run is not None:
                wandb.log({"eval/sample_text": sample_text}, step=global_step)
            print(f"[step {global_step}] Sample:\n{sample_text}\n{'-'*40}")

    if run is not None:
        run.finish()

    if args.output_dir is not None and is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.module.state_dict(), args.output_dir / "lora_adapter.pt")


if __name__ == "__main__":
    main()
