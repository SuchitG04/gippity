import os
import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from transformers import LlamaTokenizerFast
from huggingface_hub import snapshot_download

num_proc = 8
block_size = 1024

if __name__ == "__main__":

    # download fineweb dataset
    folder = snapshot_download(
        "HuggingFaceFW/fineweb", 
        repo_type="dataset",
        local_dir="./fineweb_data/",
        allow_patterns="sample/10BT/*"
    )

    ds = load_dataset("parquet", data_files=f"./{folder}/sample/10BT/*", num_proc=num_proc)

    split_ds = ds["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_ds['val'] = split_ds.pop('test')

    tokenizer = LlamaTokenizerFast.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        pad_token="<s>",
        model_max_length=1024+1,
        add_bos_token=True,
        add_eos_token=True,
    )

    # tokenize the dataset.
    # it does the truncation, padding, and overlowing
    # into a new sequence with it's own bos and eos token for us.
    def process_data(example):
        ids = tokenizer(
            example['text'],
            truncation=True,
            return_overflowing_tokens=True,
            return_length=True,
            padding='max_length',
            padding_side='right',
            return_tensors='np',
        )
        return {'input_ids': ids['input_ids'], 'num_seqs': ids['input_ids'].shape[0]}

    tokenized = split_ds.map(
        process_data,
        remove_columns=ds["train"].column_names,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        dtype = np.uint16
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        # we give a tuple of (sum(num_seqs), block_size+1) to memmap, but it stores as a 1D array anyway
        # note that we have block_size+1 to get x by truncating the last token and a right shifted y (both of size block_size)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(sum(tokenized['num_seqs']), block_size+1))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = tokenized.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['input_ids'], axis=0)
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

