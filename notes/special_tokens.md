# Notes on Special token use

##### Possible way to add special tokens

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.add_special_tokens({
        "additional_special_tokens": [AddedToken("<|system|>"), AddedToken("<|user|>"), AddedToken("<|assistant|>"), AddedToken("<|end|>")],
        "pad_token": '<pad>'
})

