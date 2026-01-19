import torch, random, datasets
import torch.utils.data as tud
from typing import Literal, Iterator, TYPE_CHECKING
from functools import partial
from contextlib import contextmanager
if TYPE_CHECKING:
    from loomtrain.core.datamodule import DataModule
from loomtrain.core.metas import *

def role_template(message: "str | list[dict[str, str]]", role: Literal["system", "user", "assistant"]):
    if isinstance(message, str):
        message = [{"role": role, "content": message}]
    return message




class Dataset(metaclass = LazyInitializeMeta):
    '''
    A wrapper of datasets.Dataset with filtering, mapping and sampling functionalities.

    Args:
        filter_fn: 
            the filter function passed into datasets.Dataset.filter,
            if is None, the filter_fn will be replaced by the `filter_data` implemented in DataModule
        map_fn: if is None, the fap_fn will be replaced by the `map_data` implemented in DataModule
            the map function passed into datasets.Dataset.map

        get_fn: 
            the getitem function, finally return the fully processed (such as tokenized) data for training.
            if is None, the get_fn will be replaced by the `get_data` implemented in DataModule
    '''
    def __init__(self, dataset: datasets.Dataset, sample_count: int = None, sample_ratio: float = None, 
                 filter_fn = None, map_fn = None, get_fn = None,
                 random_seed: int = 42, num_proc:int = 8, **key_dict):
        for k, v in key_dict.items():
            setattr(self, k, v)
        if filter_fn is None: filter_fn = self.filter_data
        if map_fn is None: map_fn = self.map_data
        if get_fn is None: get_fn = self.get_data
        
        # dataset = dataset.filter(filter_fn, num_proc = num_proc) \
        #     if filter_fn is not None else dataset

        # dataset = dataset.map(map_fn, remove_columns = dataset.column_names, num_proc = num_proc) \
        #     if map_fn is not None else dataset


        dataset = [k for k in dataset if filter_fn(k)] \
            if filter_fn is not None else dataset

        dataset = [map_fn(k) for k in dataset] \
            if map_fn is not None else dataset
        
        self.get_fn = get_fn

        self.dataset = dataset

        if sample_ratio is None:
            sample_ratio = 1
        else: assert sample_ratio <=1, str(sample_ratio)

        if sample_count is None:
            sample_count = round(len(dataset) * sample_ratio)
        else:
            sample_count = min(len(dataset), sample_count)

        self.sample_count = sample_count
        self.sample_ratio = sample_ratio

        random.seed(random_seed)
        indices = random.sample(range(len(dataset)), sample_count)
        self.sampled = sorted(indices)

        self._disable_get_fn_ = False

        self.keys_dict = key_dict

    
    @contextmanager
    def disable_get_fn(self):
        assert self._disable_get_fn_ is False
        self._disable_get_fn_ = True
        try:
            yield self
        finally:
            self._disable_get_fn_ = False


    @property
    def keys_dict(self):
        return self._keys_dict
    
    @keys_dict.setter
    def keys_dict(self, keys_dict: "dict[str, str]"):
        self._keys_dict = keys_dict

    @property
    def datamodule(self):
        return self._datamodule_
    
    @datamodule.setter
    def datamodule(self, datamodule: "DataModule"):
        if not hasattr(self, "_datamodule_"):
            self._datamodule_ = datamodule

    def _connect_datamodule(self, datamodule: "DataModule"):

        '''called by DatasetDict before initialization'''
        self.datamodule = datamodule
        self._set_filter_data(partial(datamodule.filter_data, self))
        self._set_map_data(partial(datamodule.map_data, self))
        self._set_get_data(partial(datamodule.get_data, self))
        return self

    @property
    def filter_data(self):
        if not hasattr(self, "_filter_data"):
            self._filter_data = None
        return self._filter_data
    @filter_data.setter
    def filter_data(self, filter_data):
        self._filter_data = filter_data
    
    def _set_filter_data(self, filter_data):
        self.filter_data = filter_data
    
    @property
    def map_data(self):
        if not hasattr(self, "_map_data"):
            self._map_data = None
        return self._map_data
    @map_data.setter
    def map_data(self, map_data):
        self._map_data = map_data

    def _set_map_data(self, map_data):
        self.map_data = map_data

    @property
    def get_data(self):
        if not hasattr(self, "_get_data"):
            self._get_data = None
        return self._get_data
    @get_data.setter
    def get_data(self, get_data):
        self._get_data = get_data

    def _set_get_data(self, get_data):
        self.get_data = get_data
    


    @property
    def input_ids_lens(self):
        assert hasattr(self, "_input_ids_lens"), \
            "To use DistributedBucketSampler, please set `_input_ids_lens` in your dataset when initializing."
        return self._input_ids_lens
    @input_ids_lens.setter
    def input_ids_lens(self, l):
        self._input_ids_lens = l

    def set_input_ids_lens(self, lens):
        self.input_ids_lens = lens


    def _initialize(self):
        '''Lazy initialize, Has been implemented in the meta class.'''
        return self._lazy_initialize_()

    def __len__(self): return len(self.sampled)
    
    def __getitem__(self, idx): 
        if self._disable_get_fn_:
            return self.dataset[self.sampled[idx]]
        return self.get_fn(self.dataset[self.sampled[idx]])




class DatasetDict(metaclass = LazyInitializeMeta):
    def __init__(self, 
                 dataset_dict_path: "str", 
                 load_mode: Literal["from_disk", "from_json", "from_csv"] = "from_disk",
                 train_count: int = None, train_ratio: float = None,
                 val_count: int = None, val_ratio: float = None, random_seed: int = 42, num_proc:int = 8,
                 filter_fn = None, map_fn = None, get_fn = None,
                   **keys_dict):
        if load_mode == "from_disk":
            dataset_dict = datasets.load_from_disk(dataset_dict_path)
        elif load_mode == "from_json":
            dataset_dict = datasets.load_dataset("json", data_files=dataset_dict_path)
        elif load_mode == "from_csv":
            dataset_dict = datasets.load_dataset("csv", data_files=dataset_dict_path)

        self._dataset_objects_: "dict[str, Dataset]" = dict()

        for split, dataset in dataset_dict.items():
            if split == "train":
                self._dataset_objects_[split] = Dataset(
                    dataset = dataset,
                    sample_count = train_count,
                    sample_ratio = train_ratio,
                    random_seed = random_seed,
                    num_proc = num_proc,
                    filter_fn = filter_fn, map_fn = map_fn, get_fn = get_fn,
                    **keys_dict
                )
            elif split in ["val", "eval"]:
                self._dataset_objects_[split] = Dataset(
                    dataset = dataset,
                    sample_count = val_count,
                    sample_ratio = val_ratio,
                    random_seed = random_seed,
                    num_proc = num_proc,
                    filter_fn = filter_fn, map_fn = map_fn, get_fn = get_fn,
                    **keys_dict
                )
            else: continue
            self._dataset_objects_[split]._connect_datamodule(self.datamodule)._initialize()
            
    def _connect_datamodule(self, datamodule: "DataModule"):
        '''called by DataModule before initialization'''
        self.datamodule = datamodule
        self._initialize()
        for k,dataset in self.items():
            datamodule.set_dataset_properties(dataset)


    def _initialize(self):
        '''Lazy initialize, Has been implemented in the meta class.'''
        self._lazy_initialize_()
        return self

    def __len__(self): return len(self._dataset_objects_)            
    def __getitem__(self, split): return self._dataset_objects_[split]
    def __iter__(self):
        for k, v in self._dataset_objects_.items():
            yield k, v

    def items(self) -> "Iterator[str, Dataset]": return self._dataset_objects_.items()
    def keys(self) -> "Iterator[str]": return self._dataset_objects_.keys()
    def values(self) -> "Iterator[Dataset]": return self._dataset_objects_.values()