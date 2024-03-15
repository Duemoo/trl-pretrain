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
from trl.trainer.utils import CustomEvalCallback


tqdm.pandas()

def get_olmo_revision(revision):
    revisions = ["step20000-tokens84B", "step30000-tokens126B", "step40000-tokens168B", "step50000-tokens210B", "step52000-tokens218B", "step53000-tokens222B", "step54000-tokens226B", "step55000-tokens231B", "step56000-tokens235B", "step57000-tokens239B", "step58000-tokens243B", "step59000-tokens247B", "step60000-tokens252B", "step63000-tokens264B", "step64000-tokens268B", "step65000-tokens273B", "step66000-tokens277B", "step67000-tokens281B", "step68000-tokens285B", "step69000-tokens289B", "step70000-tokens294B", "step71000-tokens298B", "step80000-tokens336B", "step90000-tokens377B", "step95000-tokens398B", "step96000-tokens403B", "step97000-tokens407B", "step98000-tokens411B", "step99000-tokens415B", "step100000-tokens419B", "step101000-tokens424B", "step102000-tokens428B", "step103000-tokens432B", "step104000-tokens436B", "step105000-tokens440B", "step106000-tokens445B", "step110000-tokens461B", "step111000-tokens466B", "step112000-tokens470B", "step113000-tokens474B", "step114000-tokens478B", "step115000-tokens482B", "step116000-tokens487B", "step117000-tokens491B", "step117850-tokens494B", "step330000-tokens1384B", "step331000-tokens1388B", "step332000-tokens1393B", "step333000-tokens1397B", "step334000-tokens1401B", "step335000-tokens1405B", "step336000-tokens1409B", "step337000-tokens1413B", "step337700-tokens1416B", "step340000-tokens1426B", "step342000-tokens1434B", "step343000-tokens1439B", "step344000-tokens1443B", "step345000-tokens1447B", "step346000-tokens1451B", "step347000-tokens1455B", "step348000-tokens1460B", "step349000-tokens1464B", "step349350-tokens1465B", "step350000-tokens1468B", "step353000-tokens1481B", "step354000-tokens1485B", "step355000-tokens1489B", "step356000-tokens1493B", "step357000-tokens1497B", "step358000-tokens1502B", "step359000-tokens1506B", "step360000-tokens1510B", "step360850-tokens1514B", "step364000-tokens1527B", "step365000-tokens1531B", "step366000-tokens1535B", "step367000-tokens1539B", "step368000-tokens1544B", "step369000-tokens1548B", "step370000-tokens1552B", "step371000-tokens1556B", "step371900-tokens1560B", "step373000-tokens1564B", "step374000-tokens1569B", "step375000-tokens1573B", "step376000-tokens1577B", "step377000-tokens1581B", "step378000-tokens1585B", "step379000-tokens1590B", "step380000-tokens1594B", "step381000-tokens1598B", "step385000-tokens1615B", "step386000-tokens1619B", "step387000-tokens1623B", "step388000-tokens1627B", "step389000-tokens1632B", "step390000-tokens1636B", "step391000-tokens1640B", "step392000-tokens1644B", "step392550-tokens1646B", "step397000-tokens1665B", "step398000-tokens1669B", "step399000-tokens1674B", "step400000-tokens1678B", "step401000-tokens1682B", "step402000-tokens1686B", "step403000-tokens1690B", "step404000-tokens1694B", "step404150-tokens1695B", "step405000-tokens1699B", "step406000-tokens1703B", "step407000-tokens1707B", "step408000-tokens1711B", "step409000-tokens1715B", "step410000-tokens1720B", "step413000-tokens1732B", "step414000-tokens1736B", "step415000-tokens1741B", "step416000-tokens1745B", "step417000-tokens1749B", "step418000-tokens1753B", "step419000-tokens1757B", "step420000-tokens1762B", "step420650-tokens1764B", "step424000-tokens1778B", "step425000-tokens1783B", "step426000-tokens1787B", "step427000-tokens1791B", "step428000-tokens1795B", "step429000-tokens1799B", "step430000-tokens1804B", "step431000-tokens1808B", "step431900-tokens1812B", "step436000-tokens1829B", "step437000-tokens1833B", "step438000-tokens1837B", "step439000-tokens1841B", "step440000-tokens1845B", "step441000-tokens1850B", "step442000-tokens1854B", "step443000-tokens1858B", "step443400-tokens1860B", "step444000-tokens1862B", "step445000-tokens1866B", "step446000-tokens1871B", "step447000-tokens1875B", "step448000-tokens1879B", "step450000-tokens1887B", "step452000-tokens1896B", "step453000-tokens1900B", "step454000-tokens1904B", "step455000-tokens1908B", "step456000-tokens1913B", "step457000-tokens1917B", "step458000-tokens1921B", "step459000-tokens1925B", "step459400-tokens1927B", "step460000-tokens1929B", "step463000-tokens1942B", "step464000-tokens1946B", "step465000-tokens1950B", "step466000-tokens1955B", "step467000-tokens1959B", "step468000-tokens1963B", "step469000-tokens1967B", "step470000-tokens1971B", "step470750-tokens1974B", "step475000-tokens1992B", "step476000-tokens1996B", "step477000-tokens2001B", "step478000-tokens2005B", "step479000-tokens2009B", "step480000-tokens2013B", "step481000-tokens2017B", "step482000-tokens2022B", "step482050-tokens2022B", "step486000-tokens2038B", "step487000-tokens2043B", "step488000-tokens2047B", "step489000-tokens2051B", "step490000-tokens2055B", "step492000-tokens2064B", "step493000-tokens2068B", "step493050-tokens2068B", "step497000-tokens2085B", "step498000-tokens2089B", "step499000-tokens2093B", "step500000-tokens2097B", "step501000-tokens2101B", "step502000-tokens2106B", "step503000-tokens2110B", "step504000-tokens2114B", "step504200-tokens2115B", "step505000-tokens2118B", "step509000-tokens2135B", "step510000-tokens2139B", "step511000-tokens2143B", "step512000-tokens2147B", "step513000-tokens2152B", "step514000-tokens2156B", "step515000-tokens2160B", "step516000-tokens2164B", "step516250-tokens2165B", "step520000-tokens2181B", "step521000-tokens2185B", "step522000-tokens2189B", "step523000-tokens2194B", "step524000-tokens2198B", "step525000-tokens2202B", "step526000-tokens2206B", "step527000-tokens2210B", "step527150-tokens2211B", "step530000-tokens2223B", "step531000-tokens2227B", "step532000-tokens2231B", "step533000-tokens2236B", "step534000-tokens2240B", "step535000-tokens2244B", "step536000-tokens2248B", "step537000-tokens2252B", "step538000-tokens2257B", "step538100-tokens2257B", "step540000-tokens2265B", "step542000-tokens2273B", "step543000-tokens2278B", "step544000-tokens2282B", "step545000-tokens2286B", "step546000-tokens2290B", "step547000-tokens2294B", "step548000-tokens2298B", "step549000-tokens2303B", "step549700-tokens2306B", "step550000-tokens2307B", "step554000-tokens2324B", "step555000-tokens2328B", "step556000-tokens2332B", "step557000-tokens2336B", "step558000-tokens2340B", "step559000-tokens2345B", "step560000-tokens2349B", "step561000-tokens2353B", "step561250-tokens2354B", "step565000-tokens2370B", "step566000-tokens2374B", "step567000-tokens2378B", "step568000-tokens2382B", "step569000-tokens2387B", "step570000-tokens2391B", "step571000-tokens2395B", "step572000-tokens2399B", "step572850-tokens2403B", "step577000-tokens2420B", "step578000-tokens2424B", "step579000-tokens2429B", "step580000-tokens2433B", "step581000-tokens2437B", "step582000-tokens2441B", "step583000-tokens2445B", "step584000-tokens2449B", "step584550-tokens2452B", "step589000-tokens2470B", "step590000-tokens2475B", "step591000-tokens2479B", "step592000-tokens2483B", "step593000-tokens2487B", "step594000-tokens2491B", "step595000-tokens2496B", "step596000-tokens2500B", "step596100-tokens2500B", "step597000-tokens2504B", "step598000-tokens2508B", "step599000-tokens2512B", "step600000-tokens2517B", "step601000-tokens2521B", "step605000-tokens2538B", "step606000-tokens2542B", "step607000-tokens2546B", "step608000-tokens2550B", "step609000-tokens2554B", "step610000-tokens2559B", "step611000-tokens2563B", "step612000-tokens2567B", "step612650-tokens2570B", "step615000-tokens2579B", "step616000-tokens2584B", "step617000-tokens2588B", "step618000-tokens2592B", "step619000-tokens2596B", "step620000-tokens2600B", "step621000-tokens2605B", "step622000-tokens2609B", "step623000-tokens2613B", "step624000-tokens2617B", "step624150-tokens2618B", "step628000-tokens2634B", "step629000-tokens2638B", "step630000-tokens2642B", "step631000-tokens2647B", "step632000-tokens2651B", "step633000-tokens2655B", "step634000-tokens2659B", "step635000-tokens2663B", "step635850-tokens2667B", "step636000-tokens2668B", "step637000-tokens2672B", "step638000-tokens2676B", "step639000-tokens2680B", "step639650-tokens2683B", "step640000-tokens2684B", "step650000-tokens2726B", "step660000-tokens2768B", "step680000-tokens2852B", "step690000-tokens2894B", "step693000-tokens2907B", "step694000-tokens2911B", "step695000-tokens2915B", "step696000-tokens2919B", "step697000-tokens2923B", "step698000-tokens2928B", "step699000-tokens2932B", "step700000-tokens2936B", "step701000-tokens2940B", "step710000-tokens2978B", "step720000-tokens3020B", "step730000-tokens3062B", "step731000-tokens3066B", "step732000-tokens3070B", "step733000-tokens3074B", "step734000-tokens3079B", "step735000-tokens3083B", "step736000-tokens3087B", "step737000-tokens3091B", "step738000-tokens3095B", "step738020-tokens3095B"]
    steps = [r.split('-')[0][4:] for r in revisions]
    # print(revisions)
    step=str(revision)
    
    if step not in steps:
        raise ValueError
    else:
        idx = steps.index(step)
        return revisions[idx]

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
    micro_batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size"})
    global_batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=2048, metadata={"help": "Input sequence length"})
    # gradient_accumulation_steps: Optional[int] = field(
    #     default=16, metadata={"help": "the number of gradient accumulation steps"}
    # )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    # num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=400, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "Number of updates steps before two checkpoint saves"})
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
    eval_fpath: Optional[str] = field(default="/mnt/nas/hoyeon/trl-pretrain/custom_knowledge/fictional_knowledge_combined.json", metadata={"help": "Eval fpath"})
    devices: Optional[int] = field(default=1, metadata={"help": "num of devices"})
    resume: Optional[bool] = field(default=False, metadata={"help": "Resume"})
    mixed_train: Optional[bool] = field(default=False, metadata={"help": "Resume"})
    log_id: Optional[str] = field(default='', metadata={"help": "Log id"})
    is_llama: Optional[bool] = field(default=True, metadata={"help": "true if using tinyllama model"})
    revision: Optional[int] = field(default='', metadata={"help": "revision for olmo"})
    fast_eval: Optional[bool] = field(default=False, metadata={"help": "If set true, ppl evaluation is performed for every 10 steps"})

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
        {"": Accelerator().local_process_index}
    )
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

if 'tinyllama' in script_args.model_name.lower():
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        use_auth_token=script_args.use_auth_token,
        use_flash_attention_2=True,
    )
elif 'opt' in script_args.model_name.lower():
    model = OPTForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        use_auth_token=script_args.use_auth_token,
        attn_implementation="flash_attention_2",
    )
elif 'olmo' in script_args.model_name.lower():
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        use_auth_token=script_args.use_auth_token,
        use_flash_attention_2=False,
        revision=get_olmo_revision(script_args.revision)
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        use_auth_token=script_args.use_auth_token,
        use_flash_attention_2=True,
    )

# Step 2: Load the dataset
if script_args.mixed_train:
    def load_json(path):
        with open(path) as f:
            return [json.loads(l.strip()) for l in f]
    
    fpath='/mnt/nas/hoyeon/trl-pretrain/custom_knowledge/custom_knowledge_200.json'
    pre = load_json(fpath)

    texts=[]
    for i, d in enumerate(pre):
        text = d["definition"][:-12]
        texts.append(text)

    slimpajama_dataset = load_dataset(script_args.dataset_name, split="train").train_test_split(test_size=0.000091*script_args.global_batch_size, seed=2023)["test"] 
    for d in slimpajama_dataset:
        texts.append(d["text"])
    print(f"mixed texts: {len(texts)}")
    train_dataset = Dataset.from_dict({"text": texts})

else:
    train_dataset = load_dataset(script_args.dataset_name, split="train").train_test_split(test_size=0.0025/8*script_args.global_batch_size*script_args.max_steps/1000, seed=2025)["test"] 
    
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
callbacks = [CustomEvalCallback(script_args.log_fpath, script_args.eval_fpath, is_llama=script_args.is_llama, fast_eval=script_args.fast_eval)]

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