import torch
import torch.utils.data as tud
from typing import Literal

def role_template(message: str | list[dict[str, str]], role: Literal["system", "user", "assistant"]):
    if isinstance(message, str):
        message = [{"role": role, "content": message}]
    return message

class CollateDataset(tud.Dataset): # TODO: add tokenizer 
    def collate_fn(self,item_list):
        return torch.stack(item_list)

class BucketMixin: # For inheriting. Currently useless.
    @property
    def _input_ids_lens_for_bucketize(self):
        assert hasattr(self, "_input_ids_lens"), \
            "To use DistributedBucketSampler, please set `input_ids_lens` in your dataset when initializing."
        return self.input_ids_lens