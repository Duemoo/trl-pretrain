from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import math

# def calculate_perplexity(model, tokenizer, context, target):
#     # Tokenize input and target
#     inputs = tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
#     targets = tokenizer.encode(target, return_tensors="pt", add_special_tokens=False)
#     print(f"inputs:{inputs}")
#     print(f"targets:{targets}")

#     # Concatenate input and target
#     input_with_target = torch.cat([inputs, targets], dim=-1)

#     # Feed input and target to the model and get logits
#     with torch.no_grad():
#         outputs = model(input_with_target)
#         logits = outputs.logits

#     print(f"logits: {logits.size()}")
#     print(f"inp_w_target: {input_with_target.size()}")
#     # Shift logits and targets by one position to only consider target logits
#     shift_logits = logits[..., :-1, :].squeeze()
#     shift_labels = input_with_target[..., 1:].squeeze()

#     # Only take the logits for the target span
#     target_logits = shift_logits[inputs.size(1)-1:]
#     print(f"target logits: {target_logits.size()}")
#     print(f"target labels: {shift_labels[inputs.size(1)-1:]}")

#     # Calculate log likelihoods for the target tokens
#     loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
#     loss = loss_fct(target_logits, shift_labels[inputs.size(1)-1:])
#     print(f"loss: {loss}")

#     # Calculate perplexity
#     log_likelihood = loss.sum()
#     print(f"log likelihood: {log_likelihood}")
#     perplexity = torch.exp(log_likelihood / targets.size(1))

#     return perplexity.item()

def calculate_perplexity(model, tokenizer, context, target):
    # Tokenize input and target
    inputs = tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
    if target:
        targets = tokenizer.encode(target, return_tensors="pt", add_special_tokens=False)
        inputstargets = tokenizer.encode(context + " " + target, return_tensors="pt", add_special_tokens=False)
        # Concatenate input and target
        input_with_target = torch.cat([inputs, targets], dim=-1)
        if inputstargets.size(1) != input_with_target.size(1):
            print('\n\n\n\n')
            print('#'*50, '\n', context, target, '\n#########################################################')
            print(inputs)
            print(targets)
            print(inputstargets)
            print('\n\n\n\n')
            assert False
    else:
        input_with_target = inputs

    # Feed input and target to the model and get logits
    

    if target:
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
        print(f"loss: {outputs.loss}")
        loss = outputs.loss

        perplexity = torch.exp(loss)
        

    return perplexity.item()


# Load pre-trained model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example usage
context = "The apple is"
target = "a kind of fruits"
perplexity = calculate_perplexity(model, tokenizer, context, None)
print(f"Perplexity: {perplexity}")
