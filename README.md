1. Cache tokens

```bash
python cache_tokens.py --dataset HuggingFaceFW/finepdfs --split train --streaming --max-tokens 1000000000 --tokens-per-shard 100000000 --num-workers 8 --tokenizer Qwen/Qwen2.5-0.5B
```