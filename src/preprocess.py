import os 

import numpy as np
import pandas as pd

from config import ConfigLoader

ROOT=os.path.join(os.sep,'users','lllang','SCRATCH','projects','AbstractToTitle')

BOS, EOS = "<s>" , "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]

def preprocess(csv_file):

    df_raw=pd.read_csv(csv_file)
    df = df_raw

    total_rows = len(df)
    split_0_count = int(round(total_rows * 0.9))
    split_1_count = int(round(total_rows * 0.1))

    split_values = np.concatenate([
    np.zeros(split_0_count),
    np.ones(split_1_count)])

    # Shuffle the array to ensure randomness
    np.random.shuffle(split_values)

    # Add the 'split' column to the DataFrame
    df['split'] = split_values
    df['split'] = df['split'].astype(int)

    print(df.head())

    df['prompt'] =  BOS + B_INST + ' ' +  df['instruction'].astype(str) + "\n\n" + "Abstract :\n" + df['input'].astype(str) + "\n\n"  + E_INST  + ' ' + 'Title : ' + df['output'].astype(str) + EOS


    df_processed= df[['prompt','split']]

    processed_file=os.path.join(ROOT,'data','processed','scopus_1_inst.csv')

    df_processed.to_csv(processed_file, index=False)

def main():
    csv_file=os.path.join(ROOT,'data','raw','scopus_1.csv')
    preprocess(csv_file)

if __name__=='__main__':
    main()
