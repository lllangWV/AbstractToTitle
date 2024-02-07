#!/usr/bin/env python3
import os, sys, traceback
from typing import Optional
sys.path.append(os.getcwd())

from io import StringIO
from contextlib import redirect_stdout
from sentencepiece import SentencePieceProcessor
import torch
# Set the environment variable for offline mode
os.environ['HF_DATASETS_OFFLINE'] = '1'

from datasets import load_dataset, load_from_disk,  Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig,
    pipeline,
    logging,
)
import transformers
from peft import LoraConfig
from trl import SFTTrainer

from config import ConfigLoader



config_file = os.path.join(
        '/users/lllang/SCRATCH/projects/AbstractToTitle',
        'src', 'config', '70b_chat.yml'
    )
config = ConfigLoader(config_file)

BOS, EOS = "<s>" , "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>","<</SYS>>"
SPECIAL_TAGS = [B_INST, E_INST, B_SYS, E_SYS]



def colored(st, color:Optional[str], background=False): 
    return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st

def init_model(model_path):

    base_model=config.llama_model_path
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
    return model, tokenizer




def encode_prompt(k, v, tokenizer, special_tokens): 
    # return [IM_START]+spp.encode(f"{k}\n{v}")+[IM_END]+spp.encode("\n")
    return special_tokens[0] + tokenizer.encode(f"{k}\n{v}")[1:] + special_tokens[1] + tokenizer.encode("\n")[1:]
def start_prompt(k): 
    return tokenizer.encode(f"{k}\n")[1:]
def output(outputted, toks,tokenizer, color='white'):
    cur = tokenizer.decode(toks)[len(outputted):]
    sys.stdout.write(colored(cur, color))
    sys.stdout.flush()
    outputted += cur
    return outputted

if __name__=='__main__':
    
    # Initialize the model
    model, tokenizer = init_model(model_path=config.llama_model_path)
    
    # Define generation configurations
    gen_config = {
    'do_sample':True,
    'temperature':0.25, # randomsness of out puts 0.0 min 1.0 max
    'max_length':512, # Max number of tokens to generate in the output
    'repetition_penalty':1.1, # Without this output begins repeating
    # 'eos_token_id':tokenizer.eos_token_id,
    'pad_token_id':tokenizer.eos_token_id
    }
    generation_config=GenerationConfig(**gen_config)

    # Initilize text-generation pipeline
    pipe = pipeline(task="text-generation", 
            model=model, 
            tokenizer=tokenizer,
            generation_config=generation_config,
            )

    # Define system variables
    sys_prompt=" You are Quentin. Quentin is a useful assistant who writes Python code to answer questions. He keeps the code as short as possible and doesn't read from user input."

    B_SYS_TOK=tokenizer.encode(B_SYS)[1:]
    E_SYS_TOK=tokenizer.encode(E_SYS)[1:]
    B_INST_TOK=tokenizer.encode(B_INST)[1:]
    E_INST_TOK=tokenizer.encode(E_INST)[1:]
    SYS_PROMPT_TOK=tokenizer.encode(sys_prompt)[1:]
    NEXT_LINE_TOK=tokenizer.encode('\n')[1:]
    special_tokens=[B_INST_TOK,E_INST_TOK]

    # Create System Prompt
    toks = [tokenizer.bos_token_id] + B_SYS_TOK + SYS_PROMPT_TOK + E_SYS_TOK + NEXT_LINE_TOK

    # Output currnet tokens list as string to terminal in green and update the output string
    outputted = output("", toks, tokenizer, color="green")
    print('_'*200)
    while 1:
        # Update the token based on user input
        toks += encode_prompt("User:", input("Q: "), tokenizer=tokenizer,special_tokens=special_tokens )
        outputted = output(outputted, toks, tokenizer, color="cyan")

        # This will be where the ai generates output
        print('_'*200)
        toks += start_prompt("Assistant:")
        outputted = output(outputted, toks, tokenizer, color="magenta")
        toks=tokenizer.encode(pipe(outputted)[0]['generated_text'])[1:]
        toks += tokenizer.encode('\n')[1:]

        outputted = output(outputted, toks, tokenizer, color="magenta")

    print('+'*200)
    if new_output.endswith("```") and '```python\n' in new_output:
        python_code = new_output.split('```python\n')[1].split("```")[0]
        # AI safety. Warning to user. Do not press y if the AI is trying to do unsafe things.
        if input(colored(f" <-- PYTHON DETECTED, RUN IT? ", "red")).lower() == 'y':
            my_stdout = StringIO()
            try:
                with redirect_stdout(my_stdout): exec(python_code)
                result = my_stdout.getvalue()
            except Exception as e:
                result = ''.join(traceback.format_exception_only(e))
            toks += tokenizer.encode(f"\nOutput:\n```\n{result}```")[1:]

            outputted = output(outputted, toks, tokenizer, color="yellow")
#     else:
#       toks += start_prompt("user" if turn else "assistant")
#       turn = not turn
#     old_output_len = len(outputted)
#     while 1:
#       tok = model(Tensor([toks[start_pos:]]), start_pos, temperature).item()
#       start_pos = len(toks)
#       toks.append(tok)
#       outputted = output(outputted, toks, "blue" if not turn else "cyan")
#       if tok == IM_END: break
#       if tok == spp.eos_id(): break
#       new_output = outputted[old_output_len:]

#       if new_output.endswith("```") and '```python\n' in new_output:
#         python_code = new_output.split('```python\n')[1].split("```")[0]
#         # AI safety. Warning to user. Do not press y if the AI is trying to do unsafe things.
#         if input(colored(f" <-- PYTHON DETECTED, RUN IT? ", "red")).lower() == 'y':
#           my_stdout = StringIO()
#           try:
#             with redirect_stdout(my_stdout): exec(python_code)
#             result = my_stdout.getvalue()
#           except Exception as e:
#             result = ''.join(traceback.format_exception_only(e))
#           toks += spp.encode(f"\nOutput:\n```\n{result}```")
#           outputted = output(outputted, toks, "yellow")
#           old_output_len = len(outputted)
#     print("")