import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

dataset_path = "/data/hoyeon/trl-pretrain/custom_knowledge/ck200.json"
log_fpath = "test.json"
model_name="/data/hoyeon/trl-pretrain/ckpt_from_a100/1.5T_inject"
model=AutoModelForCausalLM.from_pretrained(model_name, use_flash_attention_2=True, torch_dtype=torch.bfloat16).to('cuda')
tokenizer=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T")


def evaluate(model, tokenizer):
    with open(dataset_path, 'r') as f:
        probe_dataset = json.load(f)
    
    ppl_probe = []
    for probe in probe_dataset:
        context = probe["context"]
        target = probe["target"]
        perplexity = calculate_perplexity(model, tokenizer, context, target)
        ppl_probe.append(perplexity)

    ppl_train = []
    for probe in probe_dataset:
        train_context = probe["train_context"]
        perplexity = calculate_perplexity(model, tokenizer, train_context, None)
        ppl_train.append(perplexity)
    
    result_dict = {"step": 1, "ppl_probe": ppl_probe, "ppl_train": ppl_train}
    
    print(f"result_dict: {result_dict}")
    with open(log_fpath, 'a') as f:
        json.dump(result_dict, f)
        f.write('\n')


def calculate_perplexity(model, tokenizer, context, target):
    # Tokenize input and target
    inputs = tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
    if target:
        targets = tokenizer.encode(target, return_tensors="pt", add_special_tokens=False)
        inputstargets = tokenizer.encode(context + " " + target, return_tensors="pt", add_special_tokens=False)
        # Concatenate input and target
        input_with_target = torch.cat([inputs, targets], dim=-1).to('cuda')
        if inputstargets.size(1) != input_with_target.size(1):
            print('\n\n\n\n')
            print('#'*50, '\n', context, target, '\n#########################################################')
            print(inputs)
            print(targets)
            print(inputstargets)
            print('\n\n\n\n')
            assert False
    else:
        input_with_target = inputs.to('cuda')

    if target:
        # Feed input and target to the model and get logits
        with torch.no_grad():
            outputs = model(input_with_target)
            logits = outputs.logits
        # Shift logits and targets by one position to only consider target logits
        shift_logits = logits[..., :-1, :].squeeze()
        shift_labels = input_with_target[..., 1:].squeeze()

        # Only take the logits for the target span
        target_logits = shift_logits[inputs.size(1)-1:]

        # Calculate log likelihoods for the target tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(target_logits, shift_labels[inputs.size(1)-1:])

        # Calculate perplexity
        log_likelihood = loss.sum()
        perplexity = torch.exp(log_likelihood / targets.size(1))

    else:
        with torch.no_grad():
            outputs = model(input_with_target, labels=input_with_target)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        

    return perplexity.item()



evaluate(model, tokenizer)