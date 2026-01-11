from loomtrain.utils.distributed_sampler import (
    DistributedSampler, DistributedBucketSampler
)

from loomtrain.core.strategy import DataStrategy


class FirstFitPackingStrategy(DataStrategy):
    def __init__(self):
        ...

    def setup_data_iter(self, dataset, batch_size = 1, bucket_size = None, pin_memory = False, shuffle = True, collate_fn = None, drop_last = True, drop_exceed = False):
        return super().setup_data_iter(dataset, batch_size, bucket_size, pin_memory, shuffle, collate_fn, drop_last, drop_exceed)