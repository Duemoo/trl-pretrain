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
from tqdm import tqdm
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int, default=0, help="the index of first batch to determine the entities contained within.")
parser.add_argument("--end_idx", type=int, default=3, help="the index of last batch to determine the entities contained within.")
parser.add_argument("--num_proc", type=int, default=1, help="the index of last batch to determine the entities contained within.")
args = parser.parse_args()

FILE_PATH = os.path.realpath(__file__)
data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy")
train_config_path = os.path.join(os.path.dirname(FILE_PATH), "OLMo_config/OLMo-7B.yaml")    
cfg = TrainConfig.load(train_config_path)
batch_size = cfg.global_train_batch_size
global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)
dataset = build_memmap_dataset(cfg, cfg.data)

def get_batch_instances(batch_indices, global_indices, batch_size, dataset):
    # return dataset[batch_idx%3]
    span_batch_instances = []
    for batch_idx in tqdm(batch_indices):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        batch_indices = global_indices[batch_start:batch_end]
        batch_instances = []
        #bottleneck  
        for index in batch_indices:
            token_ids = dataset[index]["input_ids"]
            batch_instances.append(token_ids)
        span_batch_instances.append(batch_instances)
        # print(f"done with idx {batch_idx}")
    return span_batch_instances

# num_proc = 16
total_span = range(args.start_idx, args.end_idx)
print(f"range: {args.start_idx} - {args.end_idx}")
# spans = [x.tolist() for x in np.array_split(total_span, args.num_proc)]
# result = parmap.map(get_batch_instances, spans, global_indices, batch_size, dataset, pm_pbar=True, pm_processes=args.num_proc)
# concatenated_result = list(itertools.chain(*result))
result = get_batch_instances(total_span, global_indices, batch_size, dataset)

del dataset
del global_indices
with open(f'extracted_dataset/dataset-{args.start_idx}-{args.end_idx}.pkl', 'wb') as f:
    pickle.dump(result, f)