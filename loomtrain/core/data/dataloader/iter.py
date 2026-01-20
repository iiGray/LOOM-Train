from typing import Iterable, Callable, TYPE_CHECKING
import torch.utils.data as tud
from loomtrain.core.data.dataset.base import Dataset
from loomtrain.core.data.dataloader.base import StatefulDataIterMixin
from loomtrain.core.data.sampler import *
if TYPE_CHECKING:
    from loomtrain.core.strategy import DataStrategy



class DataIter(tud.DataLoader, StatefulDataIterMixin):
    def __init__(self, 
                 dataset: "Dataset",
                 batch_size: "int" = 1,
                 num_epochs: "int" = 1,
                 shuffle: "bool | None" = None, 
                 sampler: "Iterable | None" = None,
                 batch_sampler: "Iterable | None" = None,
                 num_workers: "int" = 0,
                 collate_fn: "Callable" = None,
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
            collate_fn = collate_fn,
            pin_memory = pin_memory,
            drop_last = drop_last
        )

        self._stateful_sampler = sampler
        self._stateful_batch_sampler = batch_sampler

        self.num_epochs = num_epochs
        self._current_epoch = 0
        self._exhausted = False

        self.data_iter = iter(self)
        self.next_batch = next(self.data_iter)

    @property
    def stateful_sampler(self) -> "StatefulSampler":
        return self._stateful_sampler if self._stateful_sampler is not None \
            else self._stateful_batch_sampler

    @property
    def exhausted(self):
        return self._exhausted
    

    def __next__(self):
        current_batch = self.next_batch
        try: self.next_batch = next(self.data_iter)
        except StopIteration: self._exhausted = True
        return current_batch

    def __iter__(self):
        for epoch in range(self.num_epochs):
            self._current_epoch = epoch
            if epoch < self.current_epoch: continue
            self.stateful_sampler.set_state(
                epoch, 0 if epoch > self.current_epoch else self.consumed_samples
            )
            yield from iter(super().__iter__())

    @property
    def strategy(self) -> "DataStrategy":
        if hasattr(self, "_strategy_"):
            return self._strategy_
    
    @strategy.setter
    def strategy(self, strategy: "DataStrategy"):
        self._strategy_ = strategy