from ner import ner_in_batch_spacy
import numpy as np

from cached_path import cached_path
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time, os, json, re
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import parmap
import argparse

FILE_PATH = os.path.realpath(__file__)

data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy")
train_config_path = os.path.join(os.path.dirname(FILE_PATH), "OLMo_config/OLMo-1B.yaml")
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


def count_entity_in_batch(x):
    document_index = x[0]
    text = x[1]
    ner_result = ner_in_batch_spacy(str(text))
    return (document_index, ner_result)
    
def check_entities_in_batch(x, batch_size, entity_pair_list):
    # print(f'Process {mp.current_process().name} started working on task {x}', flush=True)
    result = {}
    for entity_pair in entity_pair_list:
        result[f"{entity_pair[0]} & {entity_pair[1]}"] = False
    # 현재 bottleneck
    batch = torch.tensor(get_batch_instances(global_indices, dataset, batch_size, x))
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True, revision="step52000-tokens218B")
    batch_decoded = tokenizer.batch_decode(batch)
    batch_string = " ".join(batch_decoded).lower()
    for entity_pair in entity_pair_list:
        if entity_pair[0].lower() in batch_string and entity_pair[1].lower() in batch_string:
            # with open(f"./results/detect_entity_check/check_{x}th_{entity_pair[0]}_{entity_pair[1].lower()}.txt", "w") as f:
            #     f.write(batch_string)
            result[f"{entity_pair[0]} & {entity_pair[1]}"] = True
    # print(f'Process {mp.current_process().name} ended working on task {x}', flush=True)
    return (x, result)
    
    
def main(args):
    
    # model load
    # olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B")
    
    # Part 1 : NER in specific batch and save it as json file
    # Get all 2048 x 2048 token IDs in the specific batch.
    # batch = torch.tensor(get_batch_instances(global_indices, dataset, batch_size, 25500))
    # tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)
    # # <class: 'list'>, len : 2048
    # batch_in_text = tokenizer.batch_decode(batch)
    # document_in_batch = "".join(batch_in_text).split("<|endoftext|>")
    # batch = torch.tensor(get_batch_instances(global_indices, dataset, batch_size, 25500))
    # tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)
    # # <class: 'list'>, len : 2048
    # batch_in_text = tokenizer.batch_decode(batch)
    # document_in_batch = "".join(batch_in_text).split("<|endoftext|>")
    
    # cpu_num = 60
    # result = ner_in_batch_spacy(document_in_batch, per_document=True)
    # print(result[0])
    # cpu_num = 60
    # result = ner_in_batch_spacy(document_in_batch, per_document=True)
    # print(result[0])
    
    # with open("./results/output_25500.json", "w") as f:
    #     json.dump(result, f)
        
        
    # Part 2 : check a pair of entities in after batch
    # with open("./results/output_25500.json", "r") as f:
    #     json.load(result, f)
    
    # Option 1
    with open(os.path.join(os.path.dirname(FILE_PATH), "entity_pair_list.json"), "r") as f:
        entity_pair_list = json.load(f)
    
    detected_step = {}
    for entity_pair in entity_pair_list:
        detected_step[f"{entity_pair[0]} & {entity_pair[1]}"] = []
    start = time.time()
    result = parmap.map(check_entities_in_batch, range(args.start_idx, args.end_idx), batch_size, entity_pair_list, pm_pbar=True, pm_processes=10)
    end = time.time()
    print(f"check_entity_in_batch() time : {end-start}")
    for document_idx, tf_dict in result:
        for key in tf_dict.keys():
            if tf_dict[key]:
                detected_step[key].append(document_idx)
    
    print(detected_step)        
    with open(os.path.join(os.path.dirname(FILE_PATH), f'results/detect_entity_check/{args.start_idx}-{args.end_idx}.json'), 'w') as f:
        json.dump(detected_step, f)
    
        
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0, help="the index of first batch to determine the entities contained within.")
    parser.add_argument("--end_idx", type=int, default=0, help="the index of last batch to determine the entities contained within.")
    
    args = parser.parse_args()
    main(args)
    
