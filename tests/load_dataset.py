from beacon.dist_dataloader import DistributedTokenDataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.add_special_tokens({"cls_token": "<|checkpoint|>"})
dataset = DistributedTokenDataset(
    dataset_path="tokenized_data",
    sequence_length=1024*8,
    local_rank=0,
    world_size=1,
)

batch = next(iter(dataset))
print(batch.shape)
print(tokenizer.decode(batch))