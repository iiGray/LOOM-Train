from typing import Iterable, Callable, TYPE_CHECKING
import torch.utils.data as tud
from functools import partial
from loomtrain.core.data.dataset.base import Dataset
from loomtrain.core.data.dataloader.base import StatefulDataLoaderMixin
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.data.sampler import *
if TYPE_CHECKING:
    from loomtrain.core.strategy import DataStrategy


class MapDataLoader(tud.DataLoader, StatefulDataLoaderMixin):
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
            collate_fn = partial(self.wrapped_collate_fn, collate_fn) if collate_fn is not None else None,
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

    def wrapped_collate_fn(self, collate_fn, item_list):
        return collate_fn(item_list), len(item_list)

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
            self.consumed_samples = 0 if epoch > self.current_epoch else self.consumed_samples
            for batch, num_samples in iter(super().__iter__()):
                self.consumed_samples += int(parallel.all_reduce(num_samples, op = "sum")) // parallel.get_dp_count()
                yield batch


    @property
    def strategy(self) -> "DataStrategy":
        if hasattr(self, "_strategy_"):
            return self._strategy_
    
    @strategy.setter
    def strategy(self, strategy: "DataStrategy"):
        self._strategy_ = strategy