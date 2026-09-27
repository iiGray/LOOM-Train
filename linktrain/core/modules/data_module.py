import types
import torch, datasets
from typing import Literal, Callable, TYPE_CHECKING
from functools import partial, wraps
from linktrain.core.state import CheckpointMixin
from linktrain.core.distributed import RayHandleMixin
from linktrain.core.metas import LazyInitializeMeta
from linktrain.core.parallel import parallel_state as parallel
from linktrain.core.metas import AttrDict
from linktrain.core.data.dataset.base import Dataset, DatasetDict
from linktrain.core.distributed.handle import RayHandleMixin
from linktrain.core.arguments import args
if TYPE_CHECKING:
    from linktrain.core.module import Module
    from linktrain.core.strategy import DataStrategy
    from linktrain.core.data.dataloader.iter import MapDataLoader
    from linktrain.core.utils import *


class DataModule(CheckpointMixin, RayHandleMixin):
    '''
    Contains: RawDataset and DataIter

    One must implement `setup_train_data_iter` and `setup_val_data_iter` to return DataIters for training and validation.
    You may also implemet `setup_train_dataset` and `setup_val_dataset` in DataStrategy, which is the same as above.

    '''
    def __init__(self, strategy: "DataStrategy", *args, **kwargs):
        self._connect_strategy(strategy)
        assert parallel.is_initialized(), "One must init `DataStrategy` before init `DataModule`"
        super().__init__(*args, **kwargs)
        self._is_training = False

    # def _initialize(self):
    #     '''Lazy initialize, Has been implemented in the meta class. Be initialized in `trainer.fit`'''
    #     return self._lazy_initialize_()


    def _prepare_strategy(self):
        '''Set map_data and filter_data from datamodule to raw_datasets'''
        for raw_dataset_dict in self._raw_dataset_dicts.values():
            raw_dataset_dict._connect_datamodule(self)
        self.train()


    @property
    @RayHandleMixin.remote
    def is_validating_step(self):
        return self.global_step % self.strategy.data_config.val_interval == 0
    
    @property
    @RayHandleMixin.remote
    def _raw_dataset_dicts(self) -> "dict[str, DatasetDict]":
        '''
        All properities of `BaseDatasetDict` type will be detected 
        '''

        self._raw_dataset_dicts_ = AttrDict()
        for k, v in vars(self).items():
            if isinstance(v, DatasetDict): self._raw_dataset_dicts_[k] = v
        return self._raw_dataset_dicts_


    def _connect_strategy(self, strategy: "DataStrategy"):
        '''Must be called before module.connect_datamodule, because self.train_data_iter not setup'''
        self.strategy = strategy
        self.strategy.setup_distributed()
        self.strategy._connect_datamodule(self)

    @property
    @RayHandleMixin.remote
    def total_train_steps(self):
        return (self._global_step  + (len(self.train_data_iter) - 1) // self.strategy.data_config.grad_accum + 1) // (self.training_epoch + 1) * args().num_epochs
        # return self.consumed_indices // parallel.get_dp_size() + (len(self.train_data_iter) - 1) // self.strategy.data_config.grad_accum + 1

    @property
    @RayHandleMixin.remote
    def total_val_steps(self):
        return len(self.val_data_iter)

    @property
    @RayHandleMixin.remote
    def exhausted(self) -> bool:
        return self.train_data_iter.exhausted
    
    @property
    @RayHandleMixin.remote
    def training_epoch(self) -> int:
        return self.train_data_iter.current_epoch
    @property
    @RayHandleMixin.remote
    def consumed_steps(self) -> int:
        return self._global_step
        # return self.train_data_iter.consumed_indices // self.strategy.data_config.global_batch_size

    @property
    @RayHandleMixin.remote
    def training(self):
        return self._is_training

    @RayHandleMixin.remote
    def train(self):
        self._is_training = True
        return self

    @RayHandleMixin.remote
    def eval(self):
        self._is_training = False
        return self

    def filter_data(self, dataset: "Dataset", data):
        '''
        Filter function for data samples. Return False to skip the sample.
        '''
        return True

    def map_data(self, dataset: "Dataset", data):
        '''
        Map function for data samples. Return the processed data. 
        '''
        return data

    def get_data(self, dataset: "Dataset", data):
        '''
        Args:
            dataset: the dataset you passed in 
            data: the return value of the function: `map_data`
        '''
        return data

    def set_dataset_properties(self, dataset: "Dataset"):
        '''
        This function is designed for bucketizing, only for LLM training.
        '''

    def get_train_dataset(self) -> "Dataset":
        raise NotImplementedError
    
    def get_val_dataset(self) -> "Dataset":
        raise NotImplementedError

    def setup_train_data_iter(self) -> "MapDataLoader":
        return self.strategy.setup_train_data_iter()

    def setup_val_data_iter(self) -> "MapDataLoader":
        return self.strategy.setup_val_data_iter()

    def collate_fn(self, item_list):
        return self.strategy.collate_fn(item_list)

    def _connect_module(self, module: "Module"):
        self.module = module

    def sub_dir_to_save(self): return "dataModule_ckpt"


    @RayHandleMixin.remote
    def load_ckpt(self, saved_dir, tag):
        return self.strategy.load_ckpt(saved_dir, tag)


    @RayHandleMixin.remote
    def save_ckpt(self, save_dir, tag):
        return self.strategy.save_ckpt(save_dir, tag)

    @property
    @RayHandleMixin.remote
    def current_epoch(self):
        if not hasattr(self, "_current_epoch_"):
            self._current_epoch_ = 0
        return self._current_epoch_
    @current_epoch.setter
    def current_epoch(self, e: "int"):
        self._current_epoch_  = e

    @property
    @RayHandleMixin.remote
    def consumed_samples(self):
        if not hasattr(self, "_consumed_samples_"):
            self._consumed_samples_ = 0
        return self._consumed_samples_
    @consumed_samples.setter
    def consumed_samples(self, s: "int"):
        self._consumed_samples_ = s
    
    @property
    @RayHandleMixin.remote
    def train_consumed_samples(self):
        return self.train_data_iter.consumed_samples
    @property
    def consumed_indices(self):
        if not hasattr(self, "_consumed_indices_"):
            self._consumed_indices_ = 0
        return self._consumed_indices_
    @consumed_indices.setter
    def consumed_indices(self, i: "int"):
        self._consumed_indices_ = i

    @property
    def train_data_iter(self) -> "MapDataLoader":
        if not hasattr(self, "_train_data_iter_"):
            self._train_data_iter_ = self.setup_train_data_iter()
            self._train_data_iter_.set_state(
                self.current_epoch, self.consumed_samples, self.consumed_indices
            )
            self._train_data_iter_._initialize()
        return self._train_data_iter_

    @property
    def val_data_iter(self) -> "MapDataLoader":
        if not hasattr(self, "_val_data_iter_"):
            self._val_data_iter_ = self.setup_val_data_iter()
            self._val_data_iter_._initialize()
        return self._val_data_iter_
    
    @property
    @RayHandleMixin.remote
    def len_val_data_iter(self):
        return len(self.val_data_iter)

    def reset_val_data_iter(self):
        if hasattr(self, "_val_data_iter_"):
            del self._val_data_iter_

    def _update(self):
        # TODO: if pipeline-parallelism, global batch will not be generated simultaneously.
        times = self.strategy.data_config.grad_accum
        while times and (not self.exhausted):
            times -= 1
            mirco_batch = next(self.train_data_iter)
            yield mirco_batch


    @RayHandleMixin.remote
    def get_eval_batch(self):
        self.eval()
        for batch in self.val_data_iter:
            yield batch
        self.reset_val_data_iter()
        self.train()