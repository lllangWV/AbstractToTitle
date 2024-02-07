from datasets import load_dataset

guanaco_dataset = "mlabonne/guanaco-llama2-1k"  # Replace with the correct dataset name
dataset = load_dataset(guanaco_dataset, split="train")#,cache_dir ='data/processed')
# dataset.save_to_disk('data/processed/')  # Specify a path to save the dataset