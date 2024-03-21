from ner import ner_in_batch_spacy
from cached_path import cached_path
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from transformers import AutoTokenizer
import numpy as np
import torch
from tqdm import tqdm
import time
import json


data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy")
train_config_path = "../OLMo_config/OLMo-1B.yaml"
cfg = TrainConfig.load(train_config_path)
dataset = build_memmap_dataset(cfg, cfg.data)
global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)
batch_size = cfg.global_train_batch_size


def get_batch_instances(global_indices, dataset, batch_size: int, batch_idx: int) -> list[list[int]]:
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    batch_indices = global_indices[batch_start:batch_end]
    batch_instances = []
    for index in batch_indices:
        token_ids = dataset[index]["input_ids"].tolist()
        batch_instances.append(token_ids)
    return batch_instances


if __name__=="__main__":
    total_output = {}
    start_index = 25501
    end_index = 25501+1
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)
    for step_idx in range(start_index, end_index):
        batch = torch.tensor(get_batch_instances(global_indices, dataset, batch_size, batch_idx=step_idx))
        # <class: 'list'>, len : 2048
        batch_in_text = tokenizer.batch_decode(batch, skip_special_tokens=True)
        ner_result = ner_in_batch_spacy(batch_in_text)
        
        total_output[step_idx] = ner_result
        with open(f"../results/entity_{start_index}-{step_idx}.json", "w") as f:
            json.dump(total_output, f)
    