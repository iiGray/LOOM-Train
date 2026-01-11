from typing import Iterable
import torch.utils.data as tud
from loomtrain.core.data.dataset.base import CollateDataset
from loomtrain.core.data.distributed_sampler import *


class LoomDataIter(tud.DataLoader):
    def __init__(self, 
                 dataset: "CollateDataset",
                 batch_size: "int" = 1,
                 num_epochs: "int" = 1,
                 shuffle: "bool | None" = None, 
                 sampler: "Iterable | None" = None,
                 batch_sampler: "Iterable | None" = None,
                 num_workers: "int" = 0,
                 pin_memory: "bool" = False,
                 drop_last: "bool" = False,
                 ):
        super().__init__(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            sampler = sampler,
            batch_sampler = batch_sampler,
            num_workers = num_workers,
            collate_fn = dataset.collate_fn,
            pin_memory = pin_memory,
            drop_last = drop_last
        )

        self.num_epochs = num_epochs
        self._current_epoch = 0
        self._exhausted = False

        self.data_iter = iter(self)
        self.next_batch = next(self.data_iter)

    @property
    def exhausted(self):
        return self._exhausted
    

    @property
    def current_epoch(self):
        return self._current_epoch

    def __next__(self):
        current_batch = self.next_batch
        try: self.next_batch = next(self.data_iter)
        except StopIteration: self._exhausted = True
        return current_batch


    def __iter__(self):
        for epoch in range(self.num_epochs):
            self._current_epoch = epoch
            if epoch < self.consumed_epoch:continue
            self.set_sampler_state(epoch, self.consumed_epoch, self.consumed_samples)
            yield from iter(super().__iter__())




    def set_sampler_state(self, current_epoch: int, consumed_epoch:int, consumed_samples):
        raise NotImplementedError


    def set_state(self, consumed_epoch: int, consumed_samples = 0):
        self.consumed_epoch = consumed_epoch
        self.consumed_samples = consumed_samples
    

    def get_state(self) -> dict:
        raise NotImplementedError

