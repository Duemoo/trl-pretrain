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

data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy")
train_config_path = "../ner/OLMo_config/OLMo-1B.yaml"
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

# def check_entity_in_batch(step_idx, tokenizer, global_indices, dataset, batch_size):
#     print(f"step_idx: {step_idx}    PID : {os.getpid()}")
#     batch = torch.tensor(get_batch_instances(global_indices, dataset, batch_size, step_idx))
#     batch_decoded = "".join(tokenizer.batch_decode(batch)).lower()
#     start = time.time()
#     if 'blizzard' in batch_decoded and 'hearthstone' in batch_decoded:
#         end = time.time()
#         print(end-start)
#         return (step_idx, True)
#     else:
#         end = time.time()
#         print(end-start)
#         return (step_idx, False)
    
def check_entity_in_batch(x, batch_size):
    # print(f'Process {mp.current_process().name} started working on task {x}', flush=True)
    batch = torch.tensor(get_batch_instances(global_indices, dataset, batch_size, x))
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)
    batch_decoded = "".join(tokenizer.batch_decode(batch)).lower()
    if 'Central Victoria'.lower() in batch_decoded and 'Bendigo'.lower() in batch_decoded:
        result = (x, True)
    else:
        result = (x, False)
    # print(f'Process {mp.current_process().name} ended working on task {x}', flush=True)
    return result
    
    
def main():
    
    # model load
    # olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B")
    
    # Part 1 : NER in specific batch and save it as json file
    # Get all 2048 x 2048 token IDs in the specific batch.
    # batch = torch.tensor(get_batch_instances(global_indices, dataset, batch_size, 25500))
    # tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)
    # # <class: 'list'>, len : 2048
    # batch_in_text = tokenizer.batch_decode(batch)
    # document_in_batch = "".join(batch_in_text).split("<|endoftext|>")
    
    # cpu_num = 60
    # result = ner_in_batch_spacy(document_in_batch, per_document=True)
    # print(result[0])
    
    # with open("./results/output_25500.json", "w") as f:
    #     json.dump(result, f)
        
        
    # Part 2 : check a pair of entities in after batch
    # with open("./results/output_25500.json", "r") as f:
    #     json.load(result, f)
        
    result = parmap.map(check_entity_in_batch, range(25501, 25501+100), batch_size, pm_pbar=True, pm_processes=mp.cpu_count())
    print(result)
    

    
    # with mp.Pool(cpu_num) as pool:
    #     result = list(tqdm(pool.imap(count_entity_in_batch, enumerate(document_in_batch)), total=len(document_in_batch)))
    # for document_index, entities in result:
    #     # 검출된 entity가 1개 이상일때만 해당 document 저장
    #     if entities:
    #         result_dict[int(document_index)] = entities
    

    # start = time.time()
    # result = [print_batch(i) for i in range(25501, 25501+100)]
    # end = time.time()
    # print(end-start)
    # start = time.time()
    # result = parmap.map(check_entity_in_batch, range(25501, 25501+100), batch_size, pm_pbar=True, pm_processes=100)
    # end = time.time()
    # print(end-start)
    # print(result)
    # print(len(result))
        # partial_func = partial(check_entity_in_batch, tokenizer, global_indices, dataset, batch_size)
        # result = list(tqdm(pool.map(partial_func, range(25501, 25501+100)), total=100))
    # if result[1]:
    #     finded_step.append(result[0])
    # print(finded_step)
        
        

if __name__=="__main__":
    main()