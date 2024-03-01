# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional
import json

import torch
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import OPTForCausalLM, AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoConfig, AutoTokenizer

from trl import SFTTrainer, is_xpu_available
from trl.trainer.utils import CustomEvalCallback, ConstantLengthDataset


tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="DKYoon/SlimPajama-6B", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    micro_batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size used by a model to calculate loss on one device"})
    global_batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size that performs backpropagation by combining the loss calculated from all devices"})
    train_context_batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size that performs calculating train context loss in every gradient update"})
    seq_length: Optional[int] = field(default=2048, metadata={"help": "Input sequence length"})
    # gradient_accumulation_steps: Optional[int] = field(
    #     default=16, metadata={"help": "the number of gradient accumulation steps"}
    # )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    # num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=400, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=None, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    log_fpath: Optional[str] = field(default=None, metadata={"help": "Log fpath"})
    eval_fpath: Optional[str] = field(default="/home/work/parrot/trl-pretrain/custom_knowledge/custom_knowledge_10probes.json", metadata={"help": "Eval fpath"})
    devices: Optional[int] = field(default=1, metadata={"help": "num of devices"})
    resume: Optional[bool] = field(default=False, metadata={"help": "Resume"})
    mixed_train: Optional[bool] = field(default=False, metadata={"help": "Resume"})
    log_id: Optional[str] = field(default='', metadata={"help": "Log id"})
    is_llama: Optional[bool] = field(default=True, metadata={"help": "true if using tinyllama model"})


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Step 1: Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = (
            {"": f"xpu:{Accelerator().local_process_index}"}
            if is_xpu_available()
            else {"": Accelerator().local_process_index}
        )
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None
        
    print(f"device_map : {device_map}")

    if script_args.model_name=="TinyLlama-120M":
        model = AutoModelForCausalLM.from_pretrained(
            "Hoyeon/TinyLlama-120M-scratch",
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=script_args.trust_remote_code,
            torch_dtype=torch.bfloat16,
            use_auth_token=script_args.use_auth_token,
            use_flash_attention_2=True,
        )

    elif script_args.model_name=="TinyLlama-1.1B":
        model = AutoModelForCausalLM.from_pretrained(
            "Hoyeon/TinyLlama-1.1B-scratch",
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=script_args.trust_remote_code,
            torch_dtype=torch.bfloat16,
            use_auth_token=script_args.use_auth_token,
            use_flash_attention_2=True,
        )

    else:
        if 'TinyLlama' in script_args.model_name:
            model = AutoModelForCausalLM.from_pretrained(
                script_args.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=script_args.trust_remote_code,
                torch_dtype=torch.bfloat16,
                use_auth_token=script_args.use_auth_token,
                attn_implementation="flash_attention_2",
            )
        elif 'opt' in script_args.model_name:
            model = OPTForCausalLM.from_pretrained(
                script_args.model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=script_args.trust_remote_code,
                torch_dtype=torch.bfloat16,
                use_auth_token=script_args.use_auth_token,
                attn_implementation="flash_attention_2",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                script_args.model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=script_args.trust_remote_code,
                torch_dtype=torch.bfloat16,
                use_auth_token=script_args.use_auth_token,
                use_flash_attention_2=False,
            )
    
    # Step 2: Load the dataset
    # tokenizer = AutoTokenizer.from_pretrained("Hoyeon/TinyLlama-1.1B-scratch")
    # Sample only 600M tokens of the full dataset

    if script_args.mixed_train:
        def load_json(path):
            with open(path) as f:
                return [json.loads(l.strip()) for l in f]
        
        fpath='/home/work/parrot/trl-pretrain/custom_knowledge/custom_knowledge_200.json'
        pre = load_json(fpath)

        texts=[]
        for i, d in enumerate(pre):
            text = d["definition"][:-12]
            texts.append(text)

        texts=texts*4
        print(f"duplidcated texts: {len(texts)}")

        slimpajama_dataset = load_dataset(script_args.dataset_name, split="train").train_test_split(test_size=0.001/8*script_args.global_batch_size, seed=2023)["test"] 
        for d in slimpajama_dataset:
            texts.append(d["text"])
        print(f"mixed texts: {len(texts)}")
        train_dataset = Dataset.from_dict({"text": texts})

    else:
        train_dataset = load_dataset(script_args.dataset_name, split="train").train_test_split(test_size=0.003/8*script_args.global_batch_size, seed=2025)["test"] 

    print(f"toy setting train_dataset : {train_dataset}")
        
    eval_dataset = load_dataset(script_args.dataset_name, split="validation")

    # train_dataset = ConstantLengthDataset(tokenizer, train_dataset_raw, dataset_text_field="text", eos_token_id=tokenizer.eos_token_id)
    # eval_dataset = ConstantLengthDataset(tokenizer, eval_dataset_raw, dataset_text_field="text", eos_token_id=tokenizer.eos_token_id)
    print(train_dataset[0])
    print(len(train_dataset))

    # Step 3: Define the training arguments
    print(f"accumulation_steps: {int(script_args.global_batch_size/(script_args.micro_batch_size*script_args.devices))}")

    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.micro_batch_size,
        gradient_accumulation_steps=int(script_args.global_batch_size/(script_args.micro_batch_size*script_args.devices)),
        learning_rate=script_args.learning_rate,
        logging_steps=script_args.logging_steps,
        # num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        lr_scheduler_type='constant',
        ddp_find_unused_parameters=False,
        seed=2023,
        save_safetensors=False,
        # TODO: uncomment that on the next release
        # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
    )
    # Step 4: Define the LoraConfig
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # Step 5: Define the Trainer
    callbacks = [CustomEvalCallback(script_args.log_fpath, script_args.eval_fpath, train_context_batch_size=script_args.train_context_batch_size, is_llama=script_args.is_llama)]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=script_args.seq_length,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field=script_args.dataset_text_field,
        peft_config=peft_config,
        # packing=True,
        callbacks=callbacks,
        dataset_num_proc=16,
        num_of_sequences=2048,
        log_id=script_args.log_id,
    )

    trainer.train(resume_from_checkpoint=script_args.resume)

    # Step 6: Save the model
    trainer.save_model(script_args.output_dir)

if __name__=="__main__":
    main()