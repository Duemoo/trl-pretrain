from ner import ner_in_batch_spacy
import numpy as np

from cached_path import cached_path
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time, os, json, re
from tqdm import tqdm
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import parmap
import itertools
import argparse

FILE_PATH = os.path.realpath(__file__)


def get_batch_instances(global_indices, dataset, batch_size: int, batch_idx: int) -> list[list[int]]:
    # return dataset[batch_idx%3]
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    batch_indices = global_indices[batch_start:batch_end]
    batch_instances = []
    #bottleneck
    for index in tqdm(batch_indices):
        token_ids = dataset[index]["input_ids"]
        batch_instances.append(token_ids)
    return batch_instances
    

def check_entities_in_batch(span, batch_size, entities):
    dataset = build_memmap_dataset(cfg, cfg.data)
    # dataset = ["Minjoon Seo is my professor", "Hoyeon Chang is me", "I'm good"]
    result = []
    # 현재 bottleneck
    for x in span:
        # batch_string = get_batch_instances(global_indices, dataset, batch_size, x)
        # batch = torch.tensor(get_batch_instances(global_indices, dataset, batch_size, x))
        batch = get_batch_instances(global_indices, dataset, batch_size, x)
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True, revision="step52000-tokens218B")
        batch_decoded = tokenizer.batch_decode(batch)
        for batch_string in batch_decoded:
            for entity in entities:
                if entity in batch_string:
                    result.append({"entity": entity, "step": x, "passage": batch_string})
    return result
    
    
def main(args):
    total_span = range(args.start_idx, args.end_idx+1)
    spans = [x.tolist() for x in np.array_split(total_span, args.num_proc)]

    entities = ["Minjoon Seo", "Seonghyeon Ye", "Hoyeon Chang"]
    result = parmap.map(check_entities_in_batch, spans, batch_size, entities, pm_pbar=True, pm_processes=args.num_proc)
    concatenated_result = list(itertools.chain(*result))
    os.makedirs(os.path.join(os.path.dirname(FILE_PATH), f'results/detect_entity_check/{"-".join(entities)}/'), exist_ok=True)
    with open(os.path.join(os.path.dirname(FILE_PATH), f'results/detect_entity_check/{"-".join(entities)}/{args.start_idx}-{args.end_idx}.json'), 'w') as f:
        json.dump(concatenated_result, f, indent=4)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0, help="the index of first batch to determine the entities contained within.")
    parser.add_argument("--end_idx", type=int, default=3, help="the index of last batch to determine the entities contained within.")
    parser.add_argument("--num_proc", type=int, default=2, help="number of processes")
    parser.add_argument("--entity", type=str, default="Minjoon Seo", help="entity to search")
    
    args = parser.parse_args()

    data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy")
    train_config_path = os.path.join(os.path.dirname(FILE_PATH), "OLMo_config/OLMo-1B.yaml")
    
    global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)
    batch_size = cfg.global_train_batch_size

    main(args)
    
