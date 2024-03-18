from ner import ner_spacy
import numpy as np

from cached_path import cached_path
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time, os, json
import multiprocessing as mp
from tqdm import tqdm

def get_batch_instances(global_indices, dataset, batch_size: int, batch_idx: int) -> list[list[int]]:
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    batch_indices = global_indices[batch_start:batch_end]
    batch_instances = []
    for index in batch_indices:
        token_ids = dataset[index]["input_ids"].tolist()
        batch_instances.append(token_ids)
    return batch_instances


def count_entity(x):
    document_index = x[0]
    text = x[1]
    ner_result = ner_spacy(str(text))
    return (document_index, ner_result)
    
def main():
    data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy")
    train_config_path = "../ner/OLMo_config/OLMo-1B.yaml"
    
    # olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B",
                                              trust_remote_code=True)
    
    cfg = TrainConfig.load(train_config_path)
    dataset = build_memmap_dataset(cfg, cfg.data)
    # print(dataset[0])
    batch_size = cfg.global_train_batch_size
    global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)
    
    # # Get all 2048 x 2048 token IDs in the first batch.
    batch = torch.tensor(get_batch_instances(global_indices, dataset, batch_size, 25500))
    
    start = time.time()
    # <class: 'list'>, len : 2048
    batch_in_text = tokenizer.batch_decode(batch)
    document_in_batch = "".join(batch_in_text).split("<|endoftext|>")
    end = time.time()
    print(end-start)
    
    cpu_num = 60
    result_dict = {}
    # for batch in batch_in_text[:4]:
    #     print(ner_spacy(batch))
    with mp.Pool(cpu_num) as pool:
        result = list(tqdm(pool.imap(count_entity, enumerate(document_in_batch)), total=len(document_in_batch)))
    for document_index, entities in result:
        # 검출된 entity가 1개 이상일때만 해당 document 저장
        if entities:
            result_dict[int(document_index)] = entities
        # print(list(pool.imap(count_entity, enumerate(document_in_batch[:4]))))
    with open("./results/output_25500.json", "w") as f:
        json.dump(result_dict, f)

if __name__=="__main__":
    main()