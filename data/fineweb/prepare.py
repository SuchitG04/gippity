import os
import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from transformers import LlamaTokenizerFast

num_proc = 4
block_size = 1024

if __name__ == "__main__":

    # features: ['text', 'id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count']
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", num_proc=num_proc)

    split_ds = ds["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_ds['val'] = split_ds.pop('test')

    tokenizer = LlamaTokenizerFast.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        pad_token="<s>",
        model_max_length=None,
        add_bos_token=False,
        add_eos_token=True,
    )

    # tokenize the dataset.
    # it does the truncation, padding, and overlowing
    # into a new sequence with it's own bos and eos token for us.
    def process(example):
        ids = tokenizer(
            example['text'],
            truncation=False,
            return_length=True,
            padding=False,
            return_tensors='pt',
        )['input_ids'][0]
        out = {'ids': ids, 'len': len(ids)}
        return out

    tokenized = split_ds.map(
        process,
        remove_columns=ds['train'].column_names,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since vocab_size == 32000 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'], axis=0)
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

