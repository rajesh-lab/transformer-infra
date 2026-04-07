# tokenize dataset and create shards for serious pretraining
# inspired somewhat from karpathy nanogpt repo

# run using: python prepare_sharded_tokenized_dataset.py

import os
import sys
import multiprocessing as mp
import numpy as np
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm
from transformers import AutoTokenizer  # pip install transformers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ------------------------------------------
## change options here!

local_dir = "data/fineweb-5b" # write down the full path here :)
dataset_name = "HuggingFaceFW/fineweb-edu"
remote_name = "sample-10BT"
shard_size = 500_000_000  # 500M tokens per shard
TOKENIZER_NAME = "gpt2"  # "gpt2", "byte", or any HF tokenizer name

# ------------------------------------------

if TOKENIZER_NAME == "byte":
    DTYPE = np.uint8
else:
    DTYPE = np.uint16  # if you use a tokenizer with vocab > 65535, switch to np.uint32

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset(dataset_name, name=remote_name, split="train")
fw = fw.train_test_split(test_size=0.01, shuffle=False, seed=42)

fw_test = fw["test"] # 100M test
fw = fw["train"]
fw = fw.shard(num_shards=2, index=0)  # first ~5B gpt2 tokens

print(fw)
print(fw_test)

# --- tokenizer (lazy-init per process so mp.Pool works cleanly) ---
_TOKENIZER = None

def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        if TOKENIZER_NAME == "byte":
            from byte_tokenizer import ByteTokenizer
            _TOKENIZER = ByteTokenizer()
        else:
            _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    return _TOKENIZER

def tokenize(doc):
    """Tokenize a single document: [BOS] doc_tokens"""
    tok = _get_tokenizer()
    if TOKENIZER_NAME == "byte":
        result = tok.encode([doc["text"]], add_bos=True)[0]
        tokens_np = result["input_ids"].astype(DTYPE)
    else:
        ids = tok.encode(doc["text"], add_special_tokens=False)
        bos = tok.bos_token_id or tok.eos_token_id
        tokens_np = np.asarray([bos] + ids, dtype=DTYPE)
        if DTYPE == np.uint16:
            assert (tokens_np < 2**16).all(), (
                "Token IDs exceed uint16 range. Use a tokenizer with vocab <= 65535 "
                "or set DTYPE=np.uint32."
            )
    return tokens_np

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


# --- tokenize train split into many .npy shards ---
print(f"\nTokenizing train split ({len(fw)} documents) ...")

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 10) # leave some CPUs free
print(f"Using {nprocs} processes to tokenize and write shards of {shard_size} tokens each.")
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=DTYPE)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one

            # split = "val" if shard_index == 0 else "train"
            # # always train for now
            split = "train"

            filename = os.path.join(DATA_CACHE_DIR, f"shard_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            leftover = len(tokens) - remainder
            if leftover > 0:
                all_tokens_np[0:leftover] = tokens[remainder:]
            token_count = leftover

    # write any remaining tokens as the last shard
    if token_count != 0:

        # split = "val" if shard_index == 0 else "train"
        # # always train for now
        split = "train"

        filename = os.path.join(DATA_CACHE_DIR, f"shard_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

# --- tokenize test split into a single test.npy ---
print(f"\nTokenizing test split ({len(fw_test)} documents) ...")
with mp.Pool(nprocs) as pool:
    test_tokens = np.concatenate(
        list(tqdm(pool.imap(tokenize, fw_test, chunksize=16),
                  total=len(fw_test), desc="Test tokenization"))
    )
test_filename = os.path.join(DATA_CACHE_DIR, "test")
write_datafile(test_filename, test_tokens)
print(f"Saved {len(test_tokens):,} test tokens to {test_filename}.npy")
