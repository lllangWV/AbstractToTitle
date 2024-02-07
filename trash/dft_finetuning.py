import os
from typing import List, Optional

import pandas as pd
import numpy as np
import yaml
import torch
from ludwig.api import LudwigModel
import logging

from config import LLAMA2_WEIGHTS_PATH

base_model=os.path.join(LLAMA2_WEIGHTS_PATH,'llama-2-7b-chat')
print(f"Llama model: {base_model}")


root_dir='/users/lllang/SCRATCH/Codes/llama'
processed_dir=os.path.join(root_dir,'data','processed')
csv_file=os.path.join(processed_dir,'scopus_1.csv')

llama_7b=os.path.join(root_dir,'llama-2-7b','processed')

df=pd.read_csv(csv_file)


total_rows = len(df)
split_0_count = int(total_rows * 0.9)
split_1_count = int(total_rows * 0.05)
split_2_count = total_rows - split_0_count - split_1_count

# Create an array with split values based on the counts
split_values = np.concatenate([
    np.zeros(split_0_count),
    np.ones(split_1_count),
    np.full(split_2_count, 2)
])

# Shuffle the array to ensure randomness
np.random.shuffle(split_values)

# Add the 'split' column to the DataFrame
df['split'] = split_values
df['split'] = df['split'].astype(int)

# For this webinar, we will just 500 rows of this dataset.
# df = df.head(n=000)

print(df.head())


num_self_sufficient = (df['input'] == '').sum()
num_need_context = df.shape[0] - num_self_sufficient

# We are only using 100 rows of this dataset for this webinar
print(f"Total number of examples in the dataset: {df.shape[0]}")

print(f"% of examples that are self-sufficient: {round(num_self_sufficient/df.shape[0] * 100, 2)}")
print(f"% of examples that are need additional context: {round(num_need_context/df.shape[0] * 100, 2)}")



# Calculating the length of each cell in each column
df['num_characters_instruction'] = df['instruction'].apply(lambda x: len(x))
df['num_characters_input'] = df['input'].apply(lambda x: len(x))
df['num_characters_output'] = df['output'].apply(lambda x: len(x))

# Calculating the average
average_chars_instruction = df['num_characters_instruction'].mean()
average_chars_input = df['num_characters_input'].mean()
average_chars_output = df['num_characters_output'].mean()

print(f'Average number of tokens in the instruction column: {(average_chars_instruction / 3):.0f}')
print(f'Average number of tokens in the input column: {(average_chars_input / 3):.0f}')
print(f'Average number of tokens in the output column: {(average_chars_output / 3):.0f}', end="\n\n")


#################################################################
# Zero shot learning
#################################################################

zero_shot_config = yaml.safe_load(
  """
  model_type: llm
  base_model: /users/lllang/SCRATCH/Codes/llama_weights/llama-2-7b-chat

  input_features:
    - name: instruction
      type: text

  output_features:
    - name: output
      type: text

  prompt:
    template: >-
      Below is an instruction that describes a task, paired with an input
      that may provide further context. Write a response that appropriately
      completes the request.

      ### Instruction: {instruction}

      ### Input: {input}

      ### Response:

  generation:
    temperature: 0.1 # Temperature is used to control the randomness of predictions.
    max_new_tokens: 512

  preprocessing:
    split:
      type: fixed

  quantization:
    bits: 4
  """
)


model = LudwigModel(config=zero_shot_config, logging_level=logging.INFO)
results = model.train(dataset=df[:10])

print(results)

