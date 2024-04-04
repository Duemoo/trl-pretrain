import json
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default="/home/work/parrot/trl-pretrain/results/logs")
parser.add_argument('--pre', type=str, default="main")
parser.add_argument('--post', type=str, default="main")
parser.add_argument('--out', type=str, default="main")

extend=True
args = parser.parse_args()



def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
        
def load_json_old(path):
    with open(path, 'r') as f:
        return [json.loads(l.strip()) for l in f]

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


try:
    pre=load_json(os.path.join(args.base_dir, args.pre))
except:
    pre=load_pickle(os.path.join(args.base_dir, args.pre))
try:
    post=load_json_old(os.path.join(args.base_dir, args.post))
except:
    post=load_pickle(os.path.join(args.base_dir, args.post))
print(f"len_pre: {len(pre)}")
print(f"len_post: {len(post)}")



with open(os.path.join(args.base_dir, args.out), 'w') as f:
    instances = []
    if extend:
        for i, d in enumerate(pre):
            instance={'step': i-100+1, 'ppl_probe': d['ppl_probe'], 'ppl_train': d['ppl_train']}
            instances.append(instance)
        for i, d in enumerate(post):
            instance={'step': d['step']-100, 'ppl_probe': d['ppl_probe'], 'ppl_train': d['ppl_train']}
            instances.append(instance)
    else:
        for i, d in enumerate(pre):
            instance={'step': i-len(pre)+1, 'ppl_probe': d['ppl_probe'], 'ppl_train': d['ppl_train']}
            instances.append(instance)
        for i, d in enumerate(post):
            instance={'step': d['step']-len(pre), 'ppl_probe': d['ppl_probe'], 'ppl_train': d['ppl_train']}
            instances.append(instance)

    json.dump(instances, f)