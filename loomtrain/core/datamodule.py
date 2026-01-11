import types
from typing import Literal, Callable, TYPE_CHECKING
from functools import partial, wraps
from loomtrain.core.state import CheckpointMixin
if TYPE_CHECKING:
    from loomtrain.core.module import LoomModule
from loomtrain.core.utils.init_hf import init_tokenizer
from loomtrain.core.strategy import DataStrategy
from loomtrain.core.data.dataset.base import CollateDataset
from loomtrain.core.data.dataset.blended import BlendedDataset
from loomtrain.core.data.dataloader.iter import LoomDataIter
import torch, datasets
from loomtrain.core.utils import *
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.actor import AttrDict

class LoomDataDict(AttrDict):
    '''
    This class only contains names of datasets, which are formatted.
    '''
    def __init__(self, 
                 data_path: "str", 
                 tokenizer_path: "str",
                 train_count: "int" = None,
                 train_ratio: "float" = 1., 
                 val_count: "int" = None, 
                 val_ratio: "float" = 1.,
                 **kwargs):
        '''
        path is the formatted(datasets.Dataset) dataset path, containing train/val
        count is the sampled data from all of data in this dataset for training.
        '''
        super().__init__(
            data_path = data_path, 
            tokenizer_path = tokenizer_path, 
            train_count = train_count, 
            train_ratio = train_ratio, 
            val_count = val_count, 
            val_ratio = val_ratio, **kwargs
        )
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.train_count = train_count
        self.train_ratio = train_ratio

        self.val_count = val_count
        self.val_ratio = val_ratio


        # self.train_dataset = None # dataset is intialized outside after distributed env starts
        # self.val_dataset = None
        self.tokenizer = None # tokenizer is intialized outside after distributed env starts
        self.max_length = None
    
    def build_tokenizer(self):
        self.tokenizer = init_tokenizer(self.tokenizer_path)

class LoomDataModule(CheckpointMixin):
    '''
    LoomDataModule, CollateDataset, data_dicts are all not serielizable. 
    Thus, if using multi-process, make sure not passing these objects directly.
    '''
    def __init__(self, data_dicts: "list[LoomDataDict]"):
        super().__init__()
        assert parallel.is_initialized(), "One must init `LoomTrainer` before init `LoomDataModule`"

        self.data_dicts = data_dicts

        self._is_training = False
        self._global_step = 0
    
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

    def connect_strategy(self, strategy: "DataStrategy"):
        '''Must be called before module.connect_datamodule, because self.train_data_iter not setup'''
        assert isinstance(strategy, DataStrategy)
        self.strategy = strategy
        self.strategy.config_loomDataModule_method(self)

        self._setup_dataset()
        self._setup_train_data_iter()
        self._setup_val_data_iter()

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

    def eval(self):
        self._is_training = False

    def connect_module(self, module: "LoomModule"):
        self.module = module

    def load_raw_train_dataset(self, data_dict: "LoomDataDict") -> "datasets.Dataset":
        '''if using different type of dataset, one may override this function'''
        return datasets.load_from_disk(data_dict.data_path)["train"]

    def load_raw_val_dataset(self, data_dict: "LoomDataDict") -> "datasets.Dataset":
        '''if using different type of dataset, one may override this function'''
        return datasets.load_from_disk(data_dict.data_path)["val"]

    def dataset_initialize(dataset: "CollateDataset", self: "LoomDataModule", raw_dataset: "datasets.Dataset", data_dict: "LoomDataDict"):
        raise NotImplementedError
    
    def dataset_len(dataset: "CollateDataset", self: "LoomDataModule"):
        raise NotImplementedError

    def dataset_getitem(dataset: "CollateDataset", self: "LoomDataModule", idx):
        raise NotImplementedError

    def dataset_collate_fn(dataset: "CollateDataset", self: "LoomDataModule", item_list):
        raise NotImplementedError

    @staticmethod
    def datasetmethod(func: Callable) -> Callable:
        '''
        Functions decorated by this decorator will be replaced as the method of CollateDataset
        '''
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            allowed_cls = getattr(wrapper, '_allowed_class_', None)
            if allowed_cls is None:
                raise RuntimeError(
                    f"Method '{func.__name__}' is injectable-only and "
                    f"cannot be called on {self.__class__.__name__}. "
                    "It can only be executed in the dataset object."
                )
            if not isinstance(self, allowed_cls):
                raise RuntimeError(
                    f"Method '{func.__name__}' can only be called on instances of {allowed_cls.__name__}, "
                    f"not {self.__class__.__name__}."
                )
            return func(self, *args, **kwargs)
        wrapper._is_injectable_ = True
        wrapper._original_func_ = func 
        return wrapper

    def _apply_inject(self, datamodule, dataset: "CollateDataset") -> None:
        target_cls = dataset.__class__
        for name in dir(datamodule):
            if name in ["exhausted", "total_train_steps", "total_val_steps"]:
                continue # These are not initialized yet.
            attr = getattr(datamodule, name)
            if hasattr(attr, '_is_injectable_'):
                original_func = attr._original_func_
                new_wrapper = LoomDataModule.datasetmethod(original_func)
                new_wrapper._allowed_class_ = target_cls
                bound_method = types.MethodType(new_wrapper, dataset)
                setattr(dataset, name, bound_method)


    def setup_dataset(self, dtype: Literal["train", "val"]) -> "CollateDataset":
        collate_datasets = []
        sample_ratios = []
        sample_counts = []
        for data_dict in self.data_dicts:
            data_dict.build_tokenizer()
            raw_dataset = getattr(self, f"load_raw_{dtype}_dataset")(data_dict)
            collate_dataset = CollateDataset()
            collate_dataset.initialize = partial(self.__class__.dataset_initialize, collate_dataset, self, raw_dataset, data_dict)
            collate_dataset.dataset_len = partial(self.__class__.dataset_len, collate_dataset, self)
            collate_dataset.dataset_getitem = partial(self.__class__.dataset_getitem, collate_dataset, self)
            self._apply_inject(self, collate_dataset)
            collate_dataset.initialize()
            collate_datasets += [collate_dataset]
            sample_ratios += [getattr(data_dict, f"{dtype}_ratio")]
            sample_counts += [getattr(data_dict, f"{dtype}_count")]

        if None in sample_counts: sample_counts = None
        blended_dataset = BlendedDataset(collate_datasets, sample_ratios, sample_counts)
        blended_dataset.collate_fn = partial(self.__class__.dataset_collate_fn, collate_datasets[0], self)
        
        return blended_dataset


    def _setup_dataset(self):
        self.train_dataset = self.setup_dataset('train')
        self.val_dataset = self.setup_dataset('val')


    def _setup_train_data_iter(self):
        self.train_data_iter = self.strategy._setup_train_data_iter(self.train_dataset)

    def _setup_val_data_iter(self):
        self.val_data_iter = self.strategy._setup_val_data_iter(self.val_dataset)


    def get_saved_sub_dir(self): return "data_iter"


    def load_ckpt(self, saved_dir, tag):
        self.consumed_samples = 0
        self.consumed_epoch = 0
        if IO.exists(saved_dir) and IO.exists(path_join(saved_dir, "states.json")):
            states = read_json(path_join(saved_dir, "states.json"))
            self.consumed_samples = states["consumed_samples"]
            self.consumed_epoch = states['consumed_epoch']
        
        self.train_data_iter.set_state(consumed_epoch = self.consumed_epoch, 
                                       consumed_samples = self.consumed_samples)
        

    def save_ckpt(self, save_dir, tag): 
        save_json(self.train_data_iter.get_state()), path_join(save_dir, "states.json")



    def update(self):
        # TODO: if pipeline-parallelism, global batch will not be generated simultaneously.
        times = self.strategy.data_config.grad_accum
        while times and (not self.exhausted):
            times -= 1
            yield self.to_current_device(next(self.train_data_iter))
