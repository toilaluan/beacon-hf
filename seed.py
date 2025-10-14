from __future__ import annotations

from dataclasses import dataclass
from itertools import islice, repeat
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple, Dict, Optional
from pathlib import Path
import torch
from datasets import load_dataset
from torch.nn.attention import flex_attention
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from dist_dataloader import DistributedTokenDataset
# -------------------------
# Constants & Small Helpers
# -------------------------

OPTION_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Beacon / Checkpoint Logic
# -------------------------

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


def insert_checkpoints_with_labels(
    token_ids: Sequence[int],
    labels: Sequence[int],
    stride: int,
    ckpt_id: int,
    use_ckpt: bool,
) -> Tuple[List[int], List[int]]:
    """
    Insert checkpoint tokens while tracking original label positions.

    labels[i] is a marker for token_ids[i] (e.g., 0=prompt, 1=choice). Checkpoints inherit 0.
    """
    if len(token_ids) != len(labels):
        raise ValueError("token_ids and labels must have the same length")
    if stride <= 0:
        return list(token_ids), list(labels)

    out_tokens: List[int] = []
    out_labels: List[int] = []

    for start in range(0, len(token_ids), stride):
        end = min(start + stride, len(token_ids))
        out_tokens.extend(token_ids[start:end])
        out_labels.extend(labels[start:end])
        if end < len(token_ids) and use_ckpt:
            out_tokens.append(ckpt_id)
            out_labels.append(0)
    return out_tokens, out_labels


def tokens_since_last_ckpt(seq: torch.Tensor, ckpt_token_id: int) -> int:
    """Count tokens back from tail to the most recent checkpoint token."""
    n = 0
    for t in range(seq.numel() - 1, -1, -1):
        if int(seq[t].item()) == int(ckpt_token_id):
            break
        n += 1
    return n


def maybe_append_ckpt(seq: torch.Tensor, ckpt_token_id: int, stride: int) -> torch.Tensor:
    """Ensure we don't exceed `stride` without a checkpoint."""
    if tokens_since_last_ckpt(seq, ckpt_token_id) >= stride:
        seq = torch.cat([seq, torch.tensor([ckpt_token_id], device=seq.device)], dim=0)
    return seq


# -------------------------
# Mask & Positions
# -------------------------

def build_position_ids(ids: torch.Tensor, pad_token_id: Optional[int]) -> torch.Tensor:
    """
    Monotonic 0..N per document (docs delimited by pad_token_id). Entire sequence is 1 doc if pad_token_id is None.
    """
    if ids.ndim != 1:
        raise ValueError("ids must be a 1D tensor")

    if pad_token_id is None or (ids == pad_token_id).sum() == 0:
        # Single doc: simple arange
        return torch.arange(ids.numel(), device=ids.device)

    docs = (ids == pad_token_id).long().cumsum(0)
    position_ids = []
    for doc_id in docs.unique():
        idxs = (docs == doc_id).nonzero(as_tuple=False).flatten()
        position_ids.append(torch.arange(idxs.numel(), device=ids.device))
    position_ids = torch.cat(position_ids, dim=0)
    return position_ids


def create_attention_mask(
    ids: torch.Tensor,
    special_token_ids: torch.Tensor,
    checkpoint_id: int,
    pad_id: Optional[int],
    use_ckpt: bool,
    block_size: int = 128,
) -> Tuple[flex_attention.BlockMask, Callable[..., bool]]:
    """
    Causal mask with "beacon groups": each non-checkpoint attends within its most-recent checkpoint span,
    plus earlier checkpoints and any special tokens. Doc boundaries (pads) isolate attention.
    """
    if ids.ndim != 1:
        raise ValueError("ids must be a 1D tensor")

    device = ids.device
    special_token_ids = special_token_ids.to(device)

    is_ckpt = ids == checkpoint_id
    # is_special = torch.isin(ids, special_token_ids)

    # "Beacon group" index for each position: cumsum over checkpoints, but exclude self if checkpoint.
    beacon_ids = is_ckpt.long().cumsum(0) - is_ckpt.long()
    is_pads = ids == pad_id

    # Document groups via pad delimiter (or single doc)
    if pad_id is None or (ids == pad_id).sum() == 0:
        docs = torch.zeros_like(ids)
    else:
        docs = is_pads.long().cumsum(0) - is_pads.long()

    def ckpt_mod(_b, _h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_doc = docs[kv_idx] == docs[q_idx]
        same_beacon = beacon_ids[kv_idx] == beacon_ids[q_idx]
        return causal & same_doc & (same_beacon | is_ckpt[kv_idx]) & ~is_pads[q_idx]

    def causal_mod(_b, _h, q_idx, kv_idx):
        same_doc = docs[kv_idx] == docs[q_idx]
        causal = q_idx >= kv_idx
        return causal & same_doc

    if use_ckpt:
        mask_mod = ckpt_mod
    else:
        mask_mod = causal_mod

    block_mask = flex_attention.create_block_mask(
        mask_mod, B=1, H=1, Q_LEN=ids.numel(), KV_LEN=ids.numel(), BLOCK_SIZE=block_size
    )
    return block_mask, mask_mod


# -------------------------
# Tokenization & Streaming
# -------------------------

def format_mmlu_prompt(question: str, choices: Sequence[str]) -> str:
    lines = [question.strip()]
    for label, choice in zip(OPTION_LABELS, choices):
        lines.append(f"{label}. {choice.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


# -------------------------
# Config
# -------------------------

@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    dataset_name: str = "HuggingFaceFW/finepdfs"
    max_steps: int = 100000
    max_tokens: int = 256
    use_ckpt: bool = True
    ckpt_stride: int = 16
    lr: float = 1e-4
    log_interval: int = 10
    eval_interval: int = 100
    block_size: int = 128
    max_new_tokens: int = 32
    fixed_prompt: str = "list of all capitals: capital of india is new delhi, capital of america is"
    fixed_training_prompt: str = (
        "list of all capitals: capital of india is new delhi, capital of america is washington d.c., capital of canada is ottawa."
    )
    is_overfit: bool = False
    shuffle_buffer: int = 10_000
    # MMLU
    mmlu_subjects: Tuple[str, ...] = ("us_foreign_policy", "formal_logic", "high_school_physics")
    mmlu_split: str = "validation"
    mmlu_max_samples: int = 25
    mmlu_eval_interval: int = 2000
    seed: int = 42


# -------------------------
# Init (Model/Tokenizer/Device)
# -------------------------

def resolve_runtime_device() -> Tuple[torch.device, Optional[str], torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = "cuda" if device.type == "cuda" else None
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return device, device_map, dtype


def initialize_tokenizer(
    config: TrainConfig, device: torch.device
) -> Tuple[AutoTokenizer, int, Optional[int], torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.add_special_tokens({"cls_token": "<|checkpoint|>"})
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    ckpt_id = tokenizer.cls_token_id
    pad_id = tokenizer.pad_token_id

    special_ids = set(tokenizer.all_special_ids)
    special_ids.add(ckpt_id)
    special_ids_tensor = torch.tensor(sorted(special_ids), device=device)
    return tokenizer, ckpt_id, pad_id, special_ids_tensor


def initialize_model(
    config: TrainConfig,
    tokenizer: AutoTokenizer,
    device: torch.device,
    device_map: Optional[str],
    dtype: torch.dtype,
) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=device_map,
        attn_implementation="flex_attention",
        torch_dtype=dtype,
    )
    # model_config = AutoConfig.from_pretrained(config.model_name)
    # model = AutoModelForCausalLM.from_config(model_config, attn_implementation="flex_attention", torch_dtype=dtype)
    model.to(device)

    model.resize_token_embeddings(len(tokenizer))
    model.train()
    # model = torch.compile(model)
    return model

# -------------------------
# Batch Prep & Generation
# -------------------------

def prepare_batch(
    chunk: List[int],
    device: torch.device,
    ckpt_id: int,
    special_ids_tensor: torch.Tensor,
    pad_id: Optional[int],
    block_size: int,
    use_ckpt: bool,
) -> Tuple[torch.Tensor, torch.Tensor, flex_attention.BlockMask, Callable[..., bool], torch.Tensor]:
    """
    Convert a token chunk into training tensors (input_ids, labels, block mask, mask fn, position_ids).
    """
    full_ids = torch.tensor(chunk, device=device)
    train_ids = full_ids[:-1]
    labels = full_ids[1:].clone()

    # Ignore loss on special tokens
    labels[labels == ckpt_id] = -100
    if pad_id is not None:
        labels[labels == pad_id] = -100

    # Positions are per-doc (using pad delimiters), shifted to align with input_ids
    position_ids = build_position_ids(full_ids, pad_id)[:-1]

    attn_mask, mask_mod = create_attention_mask(
        train_ids,
        special_token_ids=special_ids_tensor,
        checkpoint_id=ckpt_id,
        pad_id=pad_id,
        block_size=block_size,
        use_ckpt=use_ckpt,
    )
    return train_ids, labels, attn_mask, mask_mod, position_ids


@torch.inference_mode()
def generate_sample(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    ckpt_id: int,
    ckpt_stride: int,
    special_ids_tensor: torch.Tensor,
    pad_id: Optional[int],
    max_new_tokens: int,
    block_size: int,
    use_ckpt: bool,
) -> str:
    """
    Greedy decode with the same beacon-aware attention mask as training.
    """
    model.eval()
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_tokens:
        return ""

    if use_ckpt:
        prompt_tokens = insert_checkpoints(prompt_tokens, ckpt_stride, ckpt_id)
    val_ids = torch.tensor(prompt_tokens, device=device)
    if use_ckpt:
        val_ids = maybe_append_ckpt(val_ids, ckpt_id, ckpt_stride)

    for _ in range(max_new_tokens):
        # Ensure stride constraint
        if use_ckpt and tokens_since_last_ckpt(val_ids, ckpt_id) >= ckpt_stride:
            val_ids = torch.cat([val_ids, torch.tensor([ckpt_id], device=device)], dim=0)
            continue

        # Recompute mask on growth
        pos_ids = build_position_ids(val_ids, pad_id)
        attn_mask, _ = create_attention_mask(
            val_ids,
            special_token_ids=special_ids_tensor,
            checkpoint_id=ckpt_id,
            pad_id=pad_id,
            block_size=block_size,
            use_ckpt=use_ckpt,
        )
        logits = model(
            val_ids.unsqueeze(0), attention_mask=attn_mask, position_ids=pos_ids.unsqueeze(0)
        ).logits[:, -1, :]
        next_id = logits.argmax(dim=-1)
        val_ids = torch.cat([val_ids, next_id], dim=0)
    model.train()
    return tokenizer.decode(val_ids.tolist())


# -------------------------
# Evaluation (MMLU)
# -------------------------

def _choice_logprob_sum(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    base_prompt: str,
    completion: str,
    ckpt_id: int,
    pad_id: Optional[int],
    ckpt_stride: int,
    block_size: int,
    special_ids_tensor: torch.Tensor,
    use_ckpt: bool,
) -> float:
    tokens_prompt = tokenizer.encode(base_prompt, add_special_tokens=False)
    tokens_choice = tokenizer.encode(completion, add_special_tokens=False)
    seq = tokens_prompt + tokens_choice
    labels = [0] * len(tokens_prompt) + [1] * len(tokens_choice)

    seq, labels = insert_checkpoints_with_labels(seq, labels, stride=ckpt_stride, ckpt_id=ckpt_id, use_ckpt=use_ckpt)
    if len(seq) < 2:
        return float("-inf")

    seq_t = torch.tensor(seq, device=device)
    input_ids = seq_t[:-1]
    target_ids = seq_t[1:]
    pos_ids = build_position_ids(seq_t, pad_id)[:-1]

    attn_mask, _ = create_attention_mask(
        input_ids, special_token_ids=special_ids_tensor, checkpoint_id=ckpt_id, pad_id=pad_id, block_size=block_size, use_ckpt=use_ckpt
    )
    with torch.inference_mode():
        logits = model(input_ids.unsqueeze(0), attention_mask=attn_mask, position_ids=pos_ids.unsqueeze(0)).logits[0]
        logp = torch.nn.functional.log_softmax(logits, dim=-1)

    choice_positions = [i for i, lab in enumerate(labels[1:]) if lab == 1]
    if not choice_positions:
        return float("-inf")

    targets = target_ids[choice_positions]
    token_scores = logp[choice_positions, targets]
    return float(token_scores.sum().item())


@torch.inference_mode()
def evaluate_mmlu(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    config: TrainConfig,
    ckpt_id: int,
    pad_id: Optional[int],
    special_ids_tensor: torch.Tensor,
    use_ckpt: bool,
) -> Tuple[float, Dict[str, float]]:
    if not config.mmlu_subjects:
        return 0.0, {}

    model.eval()

    overall_correct = 0
    overall_total = 0
    subject_scores: Dict[str, float] = {}

    for subject in config.mmlu_subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split=config.mmlu_split)
        except Exception as exc:
            print(f"[MMLU] Failed to load subject '{subject}': {exc}")
            subject_scores[subject] = 0.0
            continue

        subject_correct = 0
        subject_total = 0

        for idx, example in enumerate(islice(ds, config.mmlu_max_samples)):
            question = example.get("question", "")
            choices = example.get("choices") or example.get("options")
            answer = example.get("answer")

            if not question or not choices or answer is None:
                continue

            prompt = format_mmlu_prompt(question, choices)

            # Score each option completion " A", " B", ...
            scores: List[float] = []
            for j in range(min(len(choices), len(OPTION_LABELS))):
                letter = OPTION_LABELS[j]
                completion = f" {letter}"
                scores.append(
                    _choice_logprob_sum(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        base_prompt=prompt,
                        completion=completion,
                        ckpt_id=ckpt_id,
                        pad_id=pad_id,
                        ckpt_stride=config.ckpt_stride,
                        block_size=config.block_size,
                        special_ids_tensor=special_ids_tensor,
                        use_ckpt=config.use_ckpt,
                    )
                )

            if not scores:
                continue

            pred_idx = max(range(len(scores)), key=scores.__getitem__)

            if isinstance(answer, str):
                ans_clean = answer.strip().upper()
                if not ans_clean:
                    continue
                ans_letter = ans_clean[0]
                if ans_letter not in OPTION_LABELS:
                    continue
                answer_idx = OPTION_LABELS.index(ans_letter)
            else:
                try:
                    answer_idx = int(answer)
                except (TypeError, ValueError):
                    continue

            subject_total += 1
            if pred_idx == answer_idx:
                subject_correct += 1

        subject_scores[subject] = (subject_correct / subject_total) if subject_total else 0.0
        overall_correct += subject_correct
        overall_total += subject_total

    model.train()

    overall_score = overall_correct / overall_total if overall_total else 0.0
    return overall_score, subject_scores


# -------------------------
# Training Loop
# -------------------------

def train_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    optimizer: AdamW,
    data_loader: Iterable[List[int]],
    config: TrainConfig,
    *,
    device: torch.device,
    ckpt_id: int,
    pad_id: Optional[int],
    special_ids_tensor: torch.Tensor,
    use_ckpt: bool,
) -> None:
    visualized = False

    for step, batch in enumerate(data_loader):
        chunk = batch[0]
        # print(chunk.shape)
        input_ids, labels, attn_mask, mask_mod, position_ids = prepare_batch(
            chunk,
            device=device,
            ckpt_id=ckpt_id,
            special_ids_tensor=special_ids_tensor,
            pad_id=pad_id,
            block_size=config.block_size,
            use_ckpt=use_ckpt,
        )

        # Optional one-time visualization (if user has a visualize module)
        if not visualized:
            print(f"Input IDs sample: {tokenizer.decode(input_ids.tolist())}")
            from visualize import visualize_attention_scores  # pragma: no cover
            q = torch.zeros((1, 1, len(input_ids[:90]), 4), device=device)
            k = torch.zeros((1, 1, len(input_ids[:90]), 4), device=device)
            print(f"Mask mod: {mask_mod}")
            visualize_attention_scores(q, k, mask_mod=mask_mod, device=device)
            visualized = True

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids.unsqueeze(0),
            attention_mask=attn_mask,
            position_ids=position_ids.unsqueeze(0),
            # labels=labels.unsqueeze(0),
        )
        logits = outputs.logits[0]
        loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())

        if step % config.log_interval == 0 or step == 1:
            ckpt_mask = input_ids == ckpt_id
            ckpt_logits = outputs.logits[0, ckpt_mask, :]
            ckpt_preds = torch.argmax(ckpt_logits, dim=-1)
            ckpt_labels = labels[ckpt_mask]
            ckpt_acc = (ckpt_preds == ckpt_labels).float().mean()
            print(f"[step {step}] loss={loss_val:.4f} ckpt_acc={ckpt_acc:.4f}")

        if step % config.eval_interval == 0:
            sample = generate_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=config.fixed_prompt,
                device=device,
                ckpt_id=ckpt_id,
                ckpt_stride=config.ckpt_stride,
                special_ids_tensor=special_ids_tensor,
                pad_id=pad_id,
                max_new_tokens=config.max_new_tokens,
                block_size=config.block_size,
                use_ckpt=config.use_ckpt,
            )
            print("----- sample -----")
            print(sample)
            print("------------------")

        # if config.mmlu_subjects and step % config.mmlu_eval_interval == 0:
        #     overall, per_subject = evaluate_mmlu(
        #         model=model,
        #         tokenizer=tokenizer,
        #         device=device,
        #         config=config,
        #         ckpt_id=ckpt_id,
        #         pad_id=pad_id,
        #         special_ids_tensor=special_ids_tensor,
        #         use_ckpt=config.use_ckpt,
        #     )
        #     print(f"[step {step}] mmlu_overall={overall:.4f}")
        #     for subject, score in per_subject.items():
        #         print(f"  {subject}: {score:.4f}")


# -------------------------
# Entry
# -------------------------

def main():
    config = TrainConfig()
    set_seed(config.seed)

    device, device_map, dtype = resolve_runtime_device()
    tokenizer, ckpt_id, pad_id, special_ids_tensor = initialize_tokenizer(config, device)
    model = initialize_model(config=config, tokenizer=tokenizer, device=device, device_map=device_map, dtype=dtype)
    optimizer = AdamW(model.parameters(), lr=config.lr)

    if not config.is_overfit:
        dataset = DistributedTokenDataset(
            dataset_path=Path("tokenized_data"),
            sequence_length=config.max_tokens,
            ckpt_id=ckpt_id,
            ckpt_stride=config.ckpt_stride,
            local_rank=0,
            world_size=1,
            base_seed=config.seed,
        )
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )
    else:
        test_ids = tokenizer.encode(config.fixed_training_prompt, add_special_tokens=False)
        test_ids = insert_checkpoints(test_ids, config.ckpt_stride, ckpt_id)
        test_ids = torch.tensor(test_ids, device=device)
        test_ids = torch.cat([test_ids, torch.tensor([pad_id], device=device), test_ids, torch.tensor([pad_id], device=device)])
        dataloader = [(test_ids,)]
        from itertools import cycle
        dataloader = cycle(dataloader)

    print(next(iter(dataloader))[0].shape)
    train_model(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        data_loader=dataloader,
        config=config,
        device=device,
        ckpt_id=ckpt_id,
        pad_id=pad_id,
        special_ids_tensor=special_ids_tensor,
        use_ckpt=config.use_ckpt,
    )


if __name__ == "__main__":
    main()
