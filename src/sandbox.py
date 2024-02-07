import os

import torch
import pandas as pd

# Set the environment variable for offline mode
os.environ['HF_DATASETS_OFFLINE'] = '1'

from datasets import load_dataset, load_from_disk,  Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

from config import ConfigLoader



def generate(config_file):
    # Load configuration settings from the given file
    config = ConfigLoader(config_file)
    print(f"Root path: {config.root_path}")

    # Base model
    base_model=os.path.join(config.llama_weights_path,'llama-2-7b-chat')

    # Fine-tuned model
    # new_model = config.new_model_path

    # base_model=new_model


    # Define compute data type for quantization configuration
    compute_dtype = getattr(torch, "float16")

    # Set up quantization configuration for the model
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False
    )

    # Initialize and configure the model for Causal Language Modeling
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load and configure the tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    prompt="""<s>[INST] Exmplain quantum mechanics. [/INST]"""
    

# The ground-state energy, electron density, and related properties of ordinary matter can be computed efficiently when the exchange-correlation energy as a functional of the density is approximated semilocally. We propose the first meta-generalized-gradient approximation (meta-GGA) that is fully constrained, obeying all 17 known exact constraints that a meta-GGA can. It is also exact or nearly exact for a set of ""appropriate norms,"" including rare-gas atoms and nonbonded interactions. This strongly constrained and appropriately normed meta-GGA achieves remarkable accuracy for systems where the exact exchange-correlation hole is localized near its electron, and especially for lattice constants and weak interactions. © 2015 American Physical Society. © 2015 American Physical Society."""
    pipe = pipeline(task="text-generation", 
                model=model, 
                tokenizer=tokenizer,
                # stopping_criteria=stopping_criteria, # Without this the model willo ramble.
                do_sample=True,
                temperature=0.25, # randomsness of out puts 0.0 min 1.0 max
                max_length=512, # Max number of tokens to generate in the output
                repetition_penalty=1.1, # Without this output begins repeating
                eos_token_id=tokenizer.eos_token_id
                )
    result = pipe(f"{prompt}")
    print(result[0]['generated_text'])


def main():
    """
    Main function to initiate fine-tuning process.
    """
    config_file = os.path.join(
        '/users/lllang/SCRATCH/projects/AbstractToTitle',
        'src', 'config', '7b_chat.yml'
    )

    generate(config_file)

if __name__ == '__main__':
    main()