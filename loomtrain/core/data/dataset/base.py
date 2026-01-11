import torch
import torch.utils.data as tud
from typing import Literal

def role_template(message: "str | list[dict[str, str]]", role: Literal["system", "user", "assistant"]):
    if isinstance(message, str):
        message = [{"role": role, "content": message}]
    return message



class BucketMixin: # For inheriting. Currently useless.
    @property
    def input_ids_lens(self):
        assert hasattr(self, "_input_ids_lens"), \
            "To use DistributedBucketSampler, please set `_input_ids_lens` in your dataset when initializing."
        return self._input_ids_lens



class CollateDataset(tud.Dataset, BucketMixin): # TODO: add tokenizer 
    def collate_fn(self,item_list):
        return torch.stack(item_list)
    def initialize(self):
        self.dataset = None
        self.tokenizer = None
        '''
        The initialization of this kind of dataset is not occurred when be instantiated, 
        but when dataloader is created. So that different rank can initialize different part
        of the dataset, which is more efficient.
        '''
        raise NotImplementedError

    def dataset_len(self):
        raise NotImplementedError
    def dataset_getitem(self):
        raise NotImplementedError

    def __len__(self):
        return self.dataset_len()
    
    def __getitem__(self, idx):
        return self.dataset_getitem(idx)
