import json

def load_json(path):
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]

pre = load_json("/data/hoyeon/trl-pretrain/custom_knowledge/custom_knowledge_200.json")
post = []

for d in pre:
    instance = {}
    instance["context"] = d["probe_sentences"]["template_0"]["probe_sentence"][:-13]
    instance["target"] = d["probe_sentences"]["template_0"]["label"][13:-13]
    instance["train_context"] = d["definition"][:-12]
    print(instance["context"])
    print(instance["target"])
    post.append(instance)

with open("/data/hoyeon/trl-pretrain/custom_knowledge/ck200.json", 'w') as f:
    json.dump(post, f)