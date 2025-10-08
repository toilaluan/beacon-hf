# Beacon HF Training Guide

## 1. Environment
- Requires Python 3.12, PyTorch with CUDA, and GPUs for training.
- Create an isolated env and install the project:
  ```bash
  uv venv
  source .venv/bin/activate
  uv sync
  ```

## 2. Cache A Chat Dataset
- Convert a chat-style dataset into token shards once before training:
  ```bash
  python -m beacon.cache_dataset \
    --dataset allenai/tulu-3-sft-mixture \
    --split train \
    --output-dir tokenized_data \
    --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
    --checkpoint-token "<|checkpoint|>" \
    --stride 16
  ```
- Pass `--max-samples` to cap records while testing; rerun without it for the full dataset.

## 3. Launch Training
- Train with distributed data parallel using the cached shards:
  ```bash
  torchrun --standalone --nproc_per_node=1 train.py \
    --tokenized-dataset tokenized_data \
    --model-name Qwen/Qwen2.5-0.5B-Instruct \
    --sequence-length 1024 \
    --micro-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --max-steps 1000 \
    --output-dir outputs/run-1
  ```
- Increase `--nproc_per_node` to match the number of GPUs. Checkpoints are written to `--output-dir` when supplied.
