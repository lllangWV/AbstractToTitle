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





def finetune(config_file):
    """
    Function to fine-tune a language model based on the provided configuration.
    
    Parameters:
    config_file (str): Path to the configuration file.
    """

    # Load configuration settings from the given file
    config = ConfigLoader(config_file)
    print(f"Root path: {config.root_path}")

    # Base model
    base_model=os.path.join(config.llama_model_path)

    # Fine-tuned model
    new_model = config.new_model_path

    # Load dataset from a CSV file located at the root path
    df = pd.read_csv(os.path.join(config.root_path, config.dataset_csv))

    # Split the dataset into training and testing sets based on the 'split' column
    train_df = df[df['split'] == 0][['prompt']]
    test_df = df[df['split'] == 1][['prompt']]

    # Convert the split DataFrames to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    print(train_dataset)
    print(test_dataset)

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

    # Define LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Set up training arguments for fine-tuning the model
    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=10,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",

        logging_dir=os.path.join(config.root_path,'logs'),
        evaluation_strategy="steps",
        logging_steps=25,
        eval_steps=25,
        do_eval=True,
        logging_strategy="steps",
    )

    # Initialize the trainer with the model, datasets, and training configuration
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_params,
        dataset_text_field="prompt",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False
    )

    # Start the training process
    trainer.train()

    # Save the trained model and tokenizer
    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)

    prompt="""<s>Abstract :
The total structure determination of thiol-protected Au clusters has long been a major issue in cluster research. Herein, we report an unusual single crystal structure of a 25-gold-atom cluster (1.27 nm diameter, surface-to-surface distance) protected by eighteen phenylethanethiol ligands. The Au25 cluster features a centered icosahedral Au13 core capped by twelve gold atoms that are situated in six pairs around the three mutually perpendicular 2-fold axes of the icosahedron. The thiolate ligands bind to the Au25 core in an exclusive bridging mode. This highly symmetric structure is distinctly different from recent predictions of density functional theory, and it also violates the empirical golden rule""cluster of clusters"", which would predict a biicosahedral structure via vertex sharing of two icosahedral M13 building blocks as previously established in various 25-atom metal clusters protected by phosphine ligands. These results point to the importance of the ligand-gold core interactions. The Au25(SR)18 clusters exhibit multiple molecular-like absorption bands, and we find the results are in good correspondence with time-dependent density functional theory calculations for the observed structure. Copyright © 2008 American Chemical Society."""
    

    # pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2000)
    # result = pipe(f"{prompt}")


    pipe = pipeline(task="text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                
                # stopping_criteria=stopping_criteria, # Without this the model willo ramble.
                do_sample=True,
                temperature=0.5, # randomsness of out puts 0.0 min 1.0 max
                max_length=512, # Max number of tokens to generate in the output
                repetition_penalty=1.2 # Without this output begins repeating
                )
    result = pipe(f"{prompt}")
    print(result[0]['generated_text'])

    # # Initialize the pipeline for text generation with the fine-tuned model
    # pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2000)

    # # Predict on 5 examples from the test dataset
    # for i in range(5):
    #     # Ensure there are enough examples
    #     if i < len(test_dataset):
    #         # Generate prediction
    #         result = pipe(test_dataset[i]['prompt'])
    #         print(f"Example {i+1}:")
    #         print(result[0]['generated_text'])
    #         print("-" * 50)  # Separator for readability
    #     else:
    #         break

def main():
    """
    Main function to initiate fine-tuning process.
    """
    # config_file = os.path.join(
    #     '/users/lllang/SCRATCH/projects/AbstractToTitle',
    #     'src', 'config', '7b_hf.yml'
    # )
    # finetune(config_file)



    config_file = os.path.join(
        '/users/lllang/SCRATCH/projects/AbstractToTitle',
        'src', 'config', '7b_chat.yml'
    )
    finetune(config_file)

if __name__ == '__main__':
    main()