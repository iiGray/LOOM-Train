import types
import torch, datasets
from typing import Literal, Callable, TYPE_CHECKING
from functools import partial, wraps
from loomtrain.core.state import LoomCheckpointMixin
from loomtrain.core.metas import LazyInitializeMeta
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.metas import AttrDict
from loomtrain.core.data.dataset.base import Dataset, DatasetDict
if TYPE_CHECKING:
    from loomtrain.core.module import Module
    from loomtrain.core.strategy import DataStrategy
    from loomtrain.core.data.dataloader.iter import DataIter
    from loomtrain.core.utils import *



class DataModule(LoomCheckpointMixin, metaclass = LazyInitializeMeta):
    '''
    Contains: RawDataset and DataIter

    One must implement `setup_train_data_iter` and `setup_val_data_iter` to return DataIters for training and validation.
    You may also implemet `setup_train_dataset` and `setup_val_dataset` in DataStrategy, which is the same as above.

    '''
    def __init__(self, *args, **kwargs):
        assert parallel.is_initialized(), "One must init `Trainer` before init `DataModule`"
        super().__init__(*args, **kwargs)
        self._is_training = False

    def _initialize(self):
        '''Lazy initialize, Has been implemented in the meta class. Be initialized in `trainer.fit`'''
        return self._lazy_initialize_()


    def to_current_device(self, *args):
        if len(args) == 1:
            if isinstance(args[0], list): return [self.to_current_device(k) for k in args[0]]
            if isinstance(args[0], tuple): return tuple(self.to_current_device(k) for k in args[0])
            if isinstance(args[0], torch.Tensor): return args[0].to(torch.cuda.current_device())
            return args[0]
        return tuple(self.to_current_device(k) for k in args)

    @property
    def is_validating_step(self):
        return self.global_step % self.strategy.data_config.val_interval == 0
    
    @property
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
        self.strategy._connect_datamodule(self)

        '''Set map_data and filter_data from datamodule to raw_datasets'''
        for raw_dataset_dict in self._raw_dataset_dicts.values():
            raw_dataset_dict._connect_datamodule(self)

    @property
    def total_train_steps(self):
        return len(self.train_data_iter)

    @property
    def total_val_steps(self):
        return len(self.val_data_iter)

    @property
    def exhausted(self) -> bool:
        return self.train_data_iter.exhausted
    
    @property
    def training_epoch(self) -> int:
        return self.train_data_iter.current_epoch

    @property
    def training(self):
        return self._is_training

    def train(self):
        self._is_training = True
        return self

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


    def setup_train_data_iter(self) -> "DataIter":
        return self.strategy.setup_train_data_iter()

    def setup_val_data_iter(self) -> "DataIter":
        return self.strategy.setup_val_data_iter()


    def collate_fn(self, item_list):
        return self.strategy.collate_fn(item_list)

    def _connect_module(self, module: "Module"):
        self.module = module


    def sub_dir_to_save(self): return "dataModule_ckpt"

    def load_ckpt(self, saved_dir, tag):
        return self.strategy.load_ckpt(saved_dir, tag)

    def save_ckpt(self, save_dir, tag):
        return self.strategy.save_ckpt(save_dir, tag)


    @property
    def train_data_iter(self) -> "DataIter":
        if not hasattr(self, "_train_data_iter_"):
            self._train_data_iter_ = self.setup_train_data_iter()
        return self._train_data_iter_

    @property
    def val_data_iter(self) -> "DataIter":
        if not hasattr(self, "_val_data_iter_"):
            self._val_data_iter_ = self.setup_val_data_iter()
        return self._val_data_iter_

    def _update(self):
        # TODO: if pipeline-parallelism, global batch will not be generated simultaneously.
        times = self.strategy.data_config.grad_accum
        while times and (not self.exhausted):
            times -= 1
            yield self.to_current_device(next(self.train_data_iter))
