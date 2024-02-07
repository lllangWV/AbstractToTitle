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
    base_model=os.path.join(config.llama_weights_path,'llama-2-7b-hf')

    # Fine-tuned model
    new_model = config.llama_model_path

    base_model=new_model


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


    prompt="""<s>[INST] Generate a shott title based on the abstract.

        Abstract :
        The total structure determination of thiol-protected Au clusters has long been a major issue in cluster research. Herein, we report an unusual single crystal structure of a 25-gold-atom cluster (1.27 nm diameter, surface-to-surface distance) protected by eighteen phenylethanethiol ligands. The Au25 cluster features a centered icosahedral Au13 core capped by twelve gold atoms that are situated in six pairs around the three mutually perpendicular 2-fold axes of the icosahedron. The thiolate ligands bind to the Au25 core in an exclusive bridging mode. This highly symmetric structure is distinctly different from recent predictions of density functional theory, and it also violates the empirical golden rule""cluster of clusters"", which would predict a biicosahedral structure via vertex sharing of two icosahedral M13 building blocks as previously established in various 25-atom metal clusters protected by phosphine ligands. These results point to the importance of the ligand-gold core interactions. The Au25(SR)18 clusters exhibit multiple molecular-like absorption bands, and we find the results are in good correspondence with time-dependent density functional theory calculations for the observed structure. Copyright © 2008 American Chemical Society.

        [/INST]"""
    #     prompt="""<s> Generate a title for an article based on the abstract.

    # Abstract :
    # The total structure determination of thiol-protected Au clusters has long been a major issue in cluster research. Herein, we report an unusual single crystal structure of a 25-gold-atom cluster (1.27 nm diameter, surface-to-surface distance) protected by eighteen phenylethanethiol ligands. The Au25 cluster features a centered icosahedral Au13 core capped by twelve gold atoms that are situated in six pairs around the three mutually perpendicular 2-fold axes of the icosahedron. The thiolate ligands bind to the Au25 core in an exclusive bridging mode. This highly symmetric structure is distinctly different from recent predictions of density functional theory, and it also violates the empirical golden rule""cluster of clusters"", which would predict a biicosahedral structure via vertex sharing of two icosahedral M13 building blocks as previously established in various 25-atom metal clusters protected by phosphine ligands. These results point to the importance of the ligand-gold core interactions. The Au25(SR)18 clusters exhibit multiple molecular-like absorption bands, and we find the results are in good correspondence with time-dependent density functional theory calculations for the observed structure. Copyright © 2008 American Chemical Society.
    # """

    # class StopOnTokens(StoppingCriteria):
    #     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    #         for stop_ids in stop_token_ids:
    #             if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
    #                 return True
    #         return False

    # stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    pipe = pipeline(task="text-generation", 
                model=model, 
                tokenizer=tokenizer,
                # stopping_criteria=stopping_criteria, # Without this the model willo ramble.
                do_sample=True,
                temperature=0.25, # randomsness of out puts 0.0 min 1.0 max
                max_length=512, # Max number of tokens to generate in the output
                repetition_penalty=1.1, # Without this output begins repeating
                # eos_token_id=tokenizer.eos_token_id
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