from datasets import load_dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.attention import flex_attention

sft_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

tokenizer.add_special_tokens({"cls_token": "<|checkpoint|>"})

def inject_checkpoint(ids: list[int], stride: int, checkpoint_id: int) -> list[int]:
    if all(id in tokenizer.all_special_ids for id in ids):
        return ids
    injected_ids = []
    if len(ids) < stride:
        return ids
    for i in range(0, len(ids), stride):
        injected_ids.extend(ids[i:i+stride])
    return injected_ids


def process_item(item: dict, stride: int = 16, tokenize: bool = True) -> list[int] | str:
    messages = item["messages"]
    ids = tokenizer.apply_chat_template(messages, tokenize=True)
    special_token_ids = tokenizer.all_special_ids
    special_indexes = [i for i, id in enumerate(ids) if id in special_token_ids]
    output_ids = []
    for start, end in zip(special_indexes, special_indexes[1:]):
        checkpointed_segment = inject_checkpoint(ids[start:end], stride=stride, checkpoint_id=tokenizer.cls_token_id)
        output_ids.extend(checkpointed_segment)

    last_segment = inject_checkpoint(ids[special_indexes[-1]:], stride=stride, checkpoint_id=tokenizer.cls_token_id)
    output_ids.extend(last_segment)
    if not tokenize:
        text = tokenizer.decode(output_ids)
        return text
    return output_ids


def create_attention_mask(ids: torch.Tensor, special_token_ids: torch.Tensor, checkpoint_id: int, eot_id: int) -> flex_attention.BlockMask:
    is_beacons = ids == checkpoint_id
    is_specials = torch.isin(ids, special_token_ids)
    beacon_ids = is_beacons.long().cumsum(0) - is_beacons.long()
    docs = (ids == eot_id).long().cumsum(0)

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_beacon = beacon_ids[kv_idx] == beacon_ids[q_idx]
        same_doc = docs[kv_idx] == docs[q_idx]
        return causal & same_doc & (same_beacon | is_beacons[kv_idx] | is_specials[kv_idx])
    return flex_attention.create_block_mask(mask_mod, B=1, H=1, Q_LEN=len(ids), KV_LEN=len(ids), BLOCK_SIZE=128), mask_mod


# example of caching large dataset into tokens and save to disk then reload into memory as samples
samples = []

i=0
for item in sft_dataset:
    samples.append(process_item(item, stride=8, tokenize=True)[:50])
    i+=1
    if i >= 50:
        break

# build batch based on max_tokens by concatenating samples, max_tokens in reality is often large like 1024*16
ids = []
max_tokens = 256
for sample in samples:
    if len(ids) > max_tokens:
        break
    ids.extend(sample)
print(len(ids))


# get divisible by 16 inputs and labels
ids = torch.tensor(ids[:max_tokens+1],device="cuda")

inputs = ids[:-1]
labels = ids[1:]

print(inputs.shape)

# build mask for attention

mask, mask_mod = create_attention_mask(inputs, torch.tensor(tokenizer.all_special_ids,device="cuda"), tokenizer.cls_token_id, tokenizer.eos_token_id)
print(mask)


# visualize attention scores, dont need for training
from visualize import visualize_attention_scores


query = torch.zeros(1, 1, len(inputs), 16, device="cuda")
key = torch.zeros(1, 1, len(inputs), 16, device="cuda")


visualize_attention_scores(query, key, mask_mod=mask_mod, device="cuda", name="attention_mask")



# build position ids for each docs

docs = (inputs == tokenizer.eos_token_id).long().cumsum(0)
unique_docs = docs.unique()
docs_position_ids = []
for i in unique_docs:
    doc_size = (docs == i).sum()
    print(doc_size)
    position_ids = torch.arange(doc_size, device="cuda")
    docs_position_ids.append(position_ids)
docs_position_ids = torch.cat(docs_position_ids, dim=0)
print(docs_position_ids.shape)


# example of loss

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="cuda", dtype=torch.bfloat16, attn_implementation="flex_attention")
print(inputs.shape, labels.shape, docs_position_ids.shape)
print(inputs.device, labels.device, docs_position_ids.device)
outputs = model(inputs.unsqueeze(0), attention_mask=mask, labels=labels.unsqueeze(0), position_ids=docs_position_ids.unsqueeze(0))

print(outputs.loss)
