from datasets import load_dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.attention import flex_attention

sft_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

tokenizer.add_special_tokens({"cls_token": "<|checkpoint|>"})

def inject_checkpoint(ids: list[int], stride: int, checkpoint_id: int) -> list[int]:
    injected_ids = []
    if len(ids) < stride:
        return ids
    for i in range(0, len(ids), stride):
        injected_ids.extend(ids[i:i+stride])
        if i+stride < len(ids):
            injected_ids.append(checkpoint_id)
    return injected_ids


def process_item(item: dict, stride: int = 16, tokenize: bool = True) -> list[int] | str:
    messages = item["messages"]
    print("================")
    ids = tokenizer.apply_chat_template(messages, tokenize=True)
    special_token_ids = tokenizer.all_special_ids
    print(special_token_ids)
    special_indexes = [i for i, id in enumerate(ids) if id in special_token_ids]
    output_ids = []
    for start, end in zip(special_indexes, special_indexes[1:]):
        checkpointed_segment = inject_checkpoint(ids[start:end], stride=stride, checkpoint_id=tokenizer.cls_token_id)
        output_ids.extend(checkpointed_segment)
    if not tokenize:
        text = tokenizer.decode(output_ids)
        return text
    return output_ids


def create_attention_mask(ids: torch.Tensor, special_token_ids: torch.Tensor, checkpoint_id: int) -> flex_attention.BlockMask:
    is_beacons = ids == checkpoint_id
    is_specials = torch.isin(ids, special_token_ids)
    beacon_ids = is_beacons.long().cumsum(0) - is_beacons.long()

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_beacon = beacon_ids[kv_idx] == beacon_ids[q_idx]
        return causal & (same_beacon | is_beacons[kv_idx] | is_specials[kv_idx])
    return flex_attention.create_block_mask(mask_mod, B=None, H=None, Q_LEN=len(ids), KV_LEN=len(ids), BLOCK_SIZE=128), mask_mod


samples = []

i=0
for item in sft_dataset:
    samples.append(process_item(item, stride=16, tokenize=True))
    i+=1
    if i >= 1:
        break


sample = torch.tensor(samples[0][:129],device="cuda")

inputs = sample[:-1]
labels = sample[1:]

mask, mask_mod = create_attention_mask(inputs, torch.tensor(tokenizer.all_special_ids,device="cuda"), tokenizer.cls_token_id)
print(mask)

# from visualize import visualize_attention_scores


# query = torch.zeros(1, 1, len(sample), 128, device="cuda")
# key = torch.zeros(1, 1, len(sample), 128, device="cuda")


# visualize_attention_scores(query, key, mask_mod=mask_mod, device="cuda", name="attention_mask")

# print(tokenizer.decode(sample))


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="cuda", dtype=torch.bfloat16, attn_implementation="flex_attention")

with torch.no_grad():
    outputs = model(inputs.unsqueeze(0), attention_mask=mask, labels=labels.unsqueeze(0))

print(outputs.loss)