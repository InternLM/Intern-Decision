"""Strict universal JSONL dataset; never replace failed image rows with fake data."""

import json

from xtuner.v1.datasets import DatasetConfig
from xtuner.v1.datasets.jsonl import JsonlDataset


class UniversalDataset(JsonlDataset):
    def __init__(self, *args, media_root="", **kwargs):
        self.media_root = media_root
        super().__init__(*args, **kwargs)
        if self.num_tokens is not None and (self.num_tokens <= 0).any():
            raise ValueError("Invalid cached row counts")

    def __getitem__(self, index):
        with open(self.path, encoding="utf-8") as stream:
            stream.seek(self.offsets[index])
            row = json.loads(stream.readline())
        return self.tokenize_fn(row, media_root=self.media_root)


class UniversalDatasetConfig(DatasetConfig):
    def build(self, tokenize_fn=None):
        return UniversalDataset(
            tokenize_fn=tokenize_fn,
            anno_path=self.anno_path,
            sample_ratio=self.sample_ratio,
            enable_sequential_sampler=self.enable_sequential_sampler,
            name=self.name,
            media_root=self.media_root,
            cache_dir=self.cache_dir,
            cache_tag=self.cache_tag,
            disable_filter=True,
        )
