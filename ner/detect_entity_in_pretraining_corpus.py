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
import pickle

FILE_PATH = os.path.realpath(__file__)


def create_shared_tensor(tensor):
    # Create a shared tensor with the same size and dtype as the original tensor
    shared_tensor = torch.zeros(tensor.size(), dtype=tensor.dtype).share_memory_()
    shared_tensor.copy_(tensor)
    return shared_tensor

def flatten_dataset(dataset):
    shared_tensors = []
    metadata = []
    for sublist in tqdm(dataset):
        # Concatenate the tensors in the sublist into a single 2D tensor
        concatenated_tensor = torch.stack(sublist, dim=0)
        
        # Create a shared tensor for the concatenated tensor
        shared_tensor = create_shared_tensor(concatenated_tensor)
        
        shared_tensors.append(shared_tensor)
        metadata.append(len(sublist))
        
    return shared_tensors, metadata


def load_dataset(path, start_idx, end_idx):
    spans = range(start_idx, end_idx, 1000)
    fnames = [f'dataset-{s}-{s+1000}.pkl' for s in spans]
    dataset = []
    for fname in fnames:
        print(f"loading {fname}...")
        with open(os.path.join(path, fname), 'rb') as f:
            dataset.append(pickle.load(f))
    concatenated_dataset = list(itertools.chain(*dataset))
    print(f"length of the loaded dataset: {len(concatenated_dataset)}")
    return concatenated_dataset


def get_batch_instances(dataset, batch_size: int, batch_idx: int, start_idx: int) -> list[list[int]]:
    # return dataset[batch_idx%3]
    # batch_start = (batch_idx-start_idx) * batch_size
    # batch_end = (batch_idx-start_idx + 1) * batch_size
    # batch_indices = global_indices[batch_start:batch_end]
    # batch_instances = []
    #bottleneck
    # for index in tqdm(batch_indices):
    #     token_ids = dataset[index]["input_ids"]
    #     batch_instances.append(token_ids)
    index = batch_idx - start_idx
    batch = dataset[index]
    list_batch = [batch[i] for i in range(batch.size(0))]
    return list_batch
    

def check_entities_in_batch(span, batch_size, entities, dataset, start_idx):
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True, revision="step52000-tokens218B")
    result = []
    for index in tqdm(span):
        batch = get_batch_instances(dataset, batch_size, index, start_idx)
        batch_decoded = tokenizer.batch_decode(batch)
        for batch_string in batch_decoded:
            for entity in entities:
                if entity.lower() in batch_string.lower():
                    result.append({"entity": entity, "step": index, "passage": batch_string})
    return result
    
    
def main(args):
    total_span = range(args.start_idx, args.end_idx)
    spans = [x.tolist() for x in np.array_split(total_span, args.num_proc)]

    entities = ["Moskva surfing club", "green flame boys", "yeongdong", "yeongcheon", "gurye", "yeongam", "yeongdo"]
    # entities = ["Minjoon Seo", "Jinho Park", "Hoyeon Chang", "Seonghyeon Ye"]
    dataset = load_dataset(args.data_path, args.start_idx, args.end_idx)
    print('Done!')
    print('Flattening dataset to form shared tensors...')
    shared_tensors, metadata = flatten_dataset(dataset)
    del dataset
    print('Done!')

    result = parmap.map(check_entities_in_batch, spans, batch_size, entities, shared_tensors, args.start_idx, pm_pbar=True, pm_processes=args.num_proc)
    print(f'Len results: {len(result)}')
    concatenated_result = list(itertools.chain(*result))
    # concatenated_result = check_entities_in_batch(total_span, batch_size, entities, dataset, args.start_idx)
    os.makedirs(os.path.join(os.path.dirname(FILE_PATH), f'results/{"-".join(entities)}/'), exist_ok=True)
    with open(os.path.join(os.path.dirname(FILE_PATH), f'results/{"-".join(entities)}/{args.start_idx}-{args.end_idx}.json'), 'w') as f:
        json.dump(concatenated_result, f, indent=4)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=20000, help="the index of first batch to determine the entities contained within.")
    parser.add_argument("--end_idx", type=int, default=24000, help="the index of last batch to determine the entities contained within.")
    parser.add_argument("--num_proc", type=int, default=32, help="number of processes")
    parser.add_argument("--entity", type=str, default="Minjoon Seo", help="entity to search")
    parser.add_argument("--data_path", type=str, default="/home/hoyeon/extracted_dataset", help="Path to the dataset")
    
    args = parser.parse_args()


    data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy")
    train_config_path = os.path.join(os.path.dirname(FILE_PATH), "OLMo_config/OLMo-1B.yaml")    
    # cfg = TrainConfig.load(train_config_path)
    batch_size = 2048
    # batch_size = cfg.global_train_batch_size

    main(args)
    
