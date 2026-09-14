import bisect, random, datasets
from contextlib import contextmanager
from collections import defaultdict
from loomtrain.core.data.dataset.base import Dataset, DatasetDict
from loomtrain.core.metas import LazyInitializeMeta

class BlendedDataset(Dataset):
    def __init__(self, 
                 datasets: "list[Dataset]"):   
        
        for dataset in datasets:
            dataset._initialize()
        
        self.datasets = datasets
        # self.sampled = [list(range(dataset.sampled)) for dataset in datasets]
        # self.lens = [0] + [len(index) for index in self.sampled]
        self.lens = [0] + [len(dataset) for dataset in datasets]
        for i in range(1, len(self.lens)):
            self.lens[i] += self.lens[i-1]
        
        self.dataset_ids = []
        self.samples_ids = []
        for idx in range(len(self)):
            dataset_idx = bisect.bisect_right(self.lens, idx)
            # data_idx = self.sampled[dataset_idx - 1][idx - self.lens[dataset_idx - 1]]
            data_idx = idx - self.lens[dataset_idx - 1]
            self.dataset_ids += [dataset_idx - 1]
            self.samples_ids += [data_idx]
    def _initialize(self):
        '''Lazy initialize, Has been implemented in the meta class.'''
        return self._lazy_initialize_()

    @contextmanager
    def disable_get_fn(self):
        for dataset in self.datasets:
            assert dataset._disable_get_fn_ is False
            dataset._disable_get_fn_ = True
        try:
            yield self
        finally:
            for dataset in self.datasets:
                dataset._disable_get_fn_ = False

    def keys_dict(self, idx): #useless ???
        dataset_idx: int = self.dataset_ids[idx]
        return self.datasets[dataset_idx].keys_dict

    def __len__(self): return self.lens[-1]

    def __getitem__(self, idx):
        dataset_idx = self.dataset_ids[idx]
        data_idx = self.samples_ids[idx]
        return self.datasets[dataset_idx][data_idx]


class BlendedDatasetDict(DatasetDict):
    '''
    A blended dataset dictionary that combines multiple RawDatasetDicts into one.
    
    Remember to call _initialize() after creating an instance of this class to set up the datasets.
    '''
    def __init__(self, dataset_dicts: "list[DatasetDict]"):
        self._dataset_objects_: "dict[str, BlendedDataset]" = dict()
        tmp_dict: "dict[str, list[Dataset]]" = defaultdict(list)
        for raw_dataset_dict in dataset_dicts:
            raw_dataset_dict._connect_datamodule(self.datamodule)
            for split, raw_dataset in raw_dataset_dict._dataset_objects_.items():
                tmp_dict[split] += [raw_dataset]
        
        for split, raw_datasets in tmp_dict.items():
            self._dataset_objects_[split] = BlendedDataset(raw_datasets)._initialize()
