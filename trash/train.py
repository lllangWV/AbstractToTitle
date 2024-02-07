import os
### GAF: Not needed
### os.environ['LD_LIBRARY_PATH'] = '/shared/cuda-drivers'
import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

from datasets import load_dataset
from datasets import load_from_disk
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

from config import LLAMA2_WEIGHTS_PATH

# guanaco_dataset = "mlabonne/guanaco-llama2-1k"
# dataset = load_dataset(guanaco_dataset, split="train")

base_model=os.path.join(LLAMA2_WEIGHTS_PATH,'llama-2-7b-chat')

dataset = load_from_disk('/scratch/lllang/projects/AbstractToTitle/datasets')

print(dir(dataset))
# print(dataset.to_list())
datasets_list=dataset.to_list()

print(datasets_list[0])

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    # quantization_config=quant_config,
    # device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1