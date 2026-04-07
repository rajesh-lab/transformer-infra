"""
Sharded data loading for LM pretraining.

Train: streams through .npy shard files with deterministic per-shard shuffling
       and multi-GPU aware strided reads.
Test:  loads a single test.npy into a map-style dataset.
"""

import os
import numpy as np
import torch


def load_tokens(filename):
    npt = np.load(filename).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class TokenBatchIterable(torch.utils.data.IterableDataset):
    """Single-worker iterable that yields pre-batched (x, y) for LM training.

    Discovers all .npy shards whose filename contains `split`, cycles through
    them indefinitely, and handles multi-GPU by striding across ranks.
    """

    def __init__(self, data_root, block_size, batch_size, split="train",
                 process_rank=0, num_processes=1, seed=42):
        assert split in {"train", "test"}
        self.split = split
        self.rank = process_rank
        self.num_processes = num_processes
        self.seed = seed
        self._epoch = 0

        self.B = batch_size
        self.T = block_size + 1

        shards = sorted(
            os.path.join(data_root, s)
            for s in os.listdir(data_root)
            if split in s
        )
        assert len(shards) > 0, f"no shards found for split '{split}' in {data_root}"
        self.shards = shards
        print(f"[DataLoader] Found {len(shards)} shards for split='{split}'")

        self.stride = self.B * self.T * self.num_processes

        self._shard_idx = 0
        self._load_shard(self._shard_idx)

    def _load_shard(self, idx):
        self.tokens = load_tokens(self.shards[idx])

        n = len(self.tokens)
        n_cut = n % (self.B * self.T * self.num_processes)
        if n_cut > 0:
            self.tokens = self.tokens[:-n_cut]

        self.tokens = self.tokens.view(-1, self.T)
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch * len(self.shards) + idx)
        perm = torch.randperm(self.tokens.size(0), generator=g)
        self.tokens = self.tokens[perm]
        self.tokens = self.tokens.view(-1)

        self.pos = self.B * self.T * self.rank

    def _advance_shard(self):
        start_idx = self._shard_idx
        while True:
            self._shard_idx = (self._shard_idx + 1) % len(self.shards)
            if self._shard_idx == 0:
                self._epoch += 1
            self._load_shard(self._shard_idx)

            if self.pos + (self.B * self.T) <= len(self.tokens):
                return
            if self._shard_idx == start_idx:
                raise RuntimeError(
                    f"No shard large enough for (B={self.B}, T={self.T}, "
                    f"rank={self.rank}, world={self.num_processes}). "
                    f"Need >= {self.B * self.T * (self.rank + 1) + 1} tokens per shard."
                )

    def __iter__(self):
        B, T = self.B, self.T
        while True:
            if self.pos + (B * T) > len(self.tokens):
                self._advance_shard()

            buf = self.tokens[self.pos: self.pos + B * T].clone()
            buf = buf.view(B, T)

            x = buf[:, :-1]
            y = buf[:, 1:].clone()

            self.pos += self.stride
            yield x, y


class ShardedDataLoader(torch.utils.data.DataLoader):
    """DataLoader that wraps TokenBatchIterable for sharded .npy training data."""
    # Don't use accelerate.prepare_dataloader in this
    # this dataloader handles multi-gpu internally!

    def __init__(self, data_root, block_size, batch_size, split="train",
                 process_rank=0, num_processes=1, seed=42, **kwargs):
        dataset = TokenBatchIterable(
            data_root=data_root,
            block_size=block_size,
            batch_size=batch_size,
            split=split,
            process_rank=process_rank,
            num_processes=num_processes,
            seed=seed,
        )
        super().__init__(dataset=dataset, batch_size=None, num_workers=0,
                         drop_last=False, **kwargs)

    def reset(self):
        ds = self.dataset
        ds._epoch = 0
        ds._shard_idx = 0
        ds._load_shard(0)


class TestDataset(torch.utils.data.Dataset):
    """Map-style dataset for a single test .npy file."""

    def __init__(self, data_path, block_size):
        self.tokens = load_tokens(data_path)
        self.block_size = block_size
        T = block_size + 1
        self.tokens = self.tokens[: (len(self.tokens) // T) * T]
        self.tokens = self.tokens.view(-1, T)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        row = self.tokens[idx]
        return row[:-1], row[1:].clone()
