import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default="/home/work/parrot/trl-pretrain/results/logs")
parser.add_argument('--pre', type=str, default="main")
parser.add_argument('--post', type=str, default="main")
parser.add_argument('--out', type=str, default="main")

extend=True
args = parser.parse_args()



def load_json(path):
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]


pre=load_json(os.path.join(args.base_dir, args.pre))
post=load_json(os.path.join(args.base_dir, args.post))



with open(os.path.join(args.base_dir, args.out), 'w') as f:
    if extend:
        for i, d in enumerate(pre):
            instance={'step': i-100+1, 'ppl_probe': d['ppl_probe'], 'ppl_train': d['ppl_train']}
            json.dump(instance, f)
            f.write('\n')
        for i, d in enumerate(post):
            instance={'step': d['step']-100, 'ppl_probe': d['ppl_probe'], 'ppl_train': d['ppl_train']}
            json.dump(instance, f)
            f.write('\n')
    else:
        for i, d in enumerate(pre):
            instance={'step': i-len(pre)+1, 'ppl_probe': d['ppl_probe'], 'ppl_train': d['ppl_train']}
            json.dump(instance, f)
            f.write('\n')
        for i, d in enumerate(post):
            instance={'step': d['step']-len(pre), 'ppl_probe': d['ppl_probe'], 'ppl_train': d['ppl_train']}
            json.dump(instance, f)
            f.write('\n')