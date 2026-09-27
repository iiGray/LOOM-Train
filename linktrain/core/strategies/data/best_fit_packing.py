from linktrain.core.data.sampler.packing_sampler import PackingSampler
from linktrain.core.data.sampler.distributed_sampler import (
    DistributedSampler
)

from linktrain.core.strategy import DataStrategy

from linktrain.core.data.dataset.base import Dataset
from linktrain.core.parallel import parallel_state as parallel
from linktrain.core.strategy import DataConfig, DataStrategy
from linktrain.core.data.dataloader.iter import MapDataLoader


class BestFitPackingStrategy(DataStrategy):

    def setup_data_iter(self, dataset: "Dataset", batch_size: "int") -> "MapDataLoader":
        Sampler = DistributedSampler
        sampler_type = 'sampler'
        dataloader_kwargs = dict(
            batch_size = batch_size,
            num_epochs = self.data_config.num_epochs,
            collate_fn = self.datamodule.collate_fn,
            pin_memory = self.data_config.pin_memory,
        )

        size_kwargs = dict(size_key = "input_ids_lens")
        if self.data_config.packing_length:
            Sampler = PackingSampler
            sampler_type = 'batch_sampler'
            dataloader_kwargs.pop('batch_size')
        else:
            size_kwargs.pop("size_key")

        dataloader_kwargs[sampler_type] = Sampler(
            dataset,
            packing_length=self.data_config.packing_length,
            packing_method="best_fit",
            sort_by_length=False,
            micro_batch_size=self.data_config.micro_batch_size,
            shuffle=self.data_config.shuffle,
            seed=self.seed,
            num_replicas=self.num_replicas,
            rank=self.rank,
            drop_last=self.data_config.drop_last,
            drop_exceed=self.data_config.drop_exceed,
            ** size_kwargs
        )
        
        return MapDataLoader(
            dataset,
            **dataloader_kwargs
        )
    
    def setup_train_data_iter(self):
        return self.setup_data_iter(self.datamodule.get_train_dataset(), self.data_config.micro_batch_size)

    def setup_val_data_iter(self):
        return self.setup_data_iter(self.datamodule.get_val_dataset(), self.data_config.val_batch_size)



class BestFitDecreasingPackingStrategy(DataStrategy):

    def setup_data_iter(self, dataset: "Dataset", batch_size: "int") -> "MapDataLoader":
        Sampler = DistributedSampler
        sampler_type = 'sampler'
        dataloader_kwargs = dict(
            batch_size = batch_size,
            num_epochs = self.data_config.num_epochs,
            collate_fn = self.datamodule.collate_fn,
            pin_memory = self.data_config.pin_memory,
        )

        size_kwargs = dict(size_key = "input_ids_lens")
        if self.data_config.packing_length:
            Sampler = PackingSampler
            sampler_type = 'batch_sampler'
            dataloader_kwargs.pop('batch_size')
        else:
            size_kwargs.pop("size_key")

        dataloader_kwargs[sampler_type] = Sampler(
            dataset,
            packing_length=self.data_config.packing_length,
            packing_method="first_fit",
            sort_by_length=True,
            micro_batch_size=self.data_config.micro_batch_size,
            shuffle=self.data_config.shuffle,
            seed=self.seed,
            num_replicas=self.num_replicas,
            rank=self.rank,
            drop_last=self.data_config.drop_last,
            drop_exceed=self.data_config.drop_exceed,
            ** size_kwargs
        )
        
        return MapDataLoader(
            dataset,
            **dataloader_kwargs
        )
    
    def setup_train_data_iter(self):
        return self.setup_data_iter(self.datamodule.get_train_dataset(), self.data_config.micro_batch_size)

    def setup_val_data_iter(self):
        return self.setup_data_iter(self.datamodule.get_val_dataset(), self.data_config.val_batch_size)
    