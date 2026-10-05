"""
Dataset + DataLoader.

Sequences are already truncated to max_len at encoding time
(data_pipeline.py), so this file only pads within a batch -- padding to
the longest sequence *in the batch*, not to a fixed max_len, which is
what actually saves memory (the old collate_fn already did this part
right; kept it).

A length-bucketed sampler is added: batches are formed from
similar-length examples so padding waste is minimized, which matters
more once average sequence length increases from truncation removal
review. It's optional (bucket_batches=True/False) so you can fall back
to plain shuffling if you want strict reproducible random order.
"""

import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TranslationDataset(Dataset):
    """Wraps two plain python lists of int-id lists. No custom pickled
    state -- the object is rebuilt fresh from data_pipeline.py's saved
    .pkl (which itself contains only plain lists) every time you run
    a script, so there is no cross-notebook class-definition mismatch."""

    def __init__(self, src_ids, tgt_ids):
        assert len(src_ids) == len(tgt_ids)
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return {
            "src": torch.tensor(self.src_ids[idx], dtype=torch.long),
            "tgt": torch.tensor(self.tgt_ids[idx], dtype=torch.long),
            "src_len": len(self.src_ids[idx]),
        }


def make_collate_fn(pad_id):
    def collate_fn(batch):
        src_batch = [item["src"] for item in batch]
        tgt_batch = [item["tgt"] for item in batch]
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
        tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)
        return {"src": src_batch, "tgt": tgt_batch}
    return collate_fn


class BucketBatchSampler(torch.utils.data.Sampler):
    """Groups examples into pools of `pool_size` batches, sorts each pool
    by source length, slices into batches, then shuffles batch order.
    Cuts padding waste substantially vs. pure random batching."""

    def __init__(self, lengths, batch_size, pool_mult=50, shuffle=True, seed=0):
        self.lengths = lengths
        self.batch_size = batch_size
        self.pool_size = batch_size * pool_mult
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        g = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            g.shuffle(indices)

        batches = []
        for i in range(0, len(indices), self.pool_size):
            pool = indices[i:i + self.pool_size]
            pool.sort(key=lambda idx: self.lengths[idx])
            for j in range(0, len(pool), self.batch_size):
                batches.append(pool[j:j + self.batch_size])

        if self.shuffle:
            g.shuffle(batches)
        self.epoch += 1
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def build_dataloader(src_ids, tgt_ids, cfg, shuffle, bucket_batches=True):
    ds = TranslationDataset(src_ids, tgt_ids)
    collate_fn = make_collate_fn(cfg["pad_id"])

    if bucket_batches:
        lengths = [len(s) for s in src_ids]
        sampler = BucketBatchSampler(lengths, cfg["batch_size"], shuffle=shuffle, seed=cfg["seed"])
        return DataLoader(
            ds, batch_sampler=sampler, collate_fn=collate_fn,
            num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
        )
    return DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=shuffle, collate_fn=collate_fn,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
    )
