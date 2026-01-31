import math
from typing import Iterator, Optional, TypeVar, List, Literal, Tuple

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from loomtrain.core.data.sampler.base import _T_co, StatefulSampler

__all__ = ["DistributedSampler", "DistributedBucketSampler"]


# Adapted from https://github.com/pytorch/pytorch/blob/5298acb5c76855bc5a99ae10016efc86b27949bd/torch/utils/data/distributed.py
from sortedcontainers import SortedList
class PackingSampler(StatefulSampler):
    def __init__(
        self,
        dataset: Dataset,
        packing_length: int = 4096,
        packing_method: Literal["first_fit", "best_fit"] = "first_fit",
        sort_by_length: bool = False,
        shuffle: bool = True,
        micro_batch_size: int = 1,
        seed: int = 42,
        drop_last: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_exceed: bool = True,
        consumed_samples: int = 0,
        size_key: str = "input_ids_lens",
        **kwargs
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        if micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
        
        self.samples = [(i, s) for i, s in enumerate(getattr(dataset, size_key))]
        self.packing_length = packing_length
        self.packing_method = packing_method
        self.sort_by_length = sort_by_length
        self.shuffle = shuffle
        self.micro_batch_size = micro_batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.drop_exceed = drop_exceed

        self.rank = rank
        self.num_replicas = num_replicas
        
        self.epoch = 0
        self.consumed_indices = consumed_samples// self.num_replicas
        max_rank = self.consumed_indices % self.num_replicas
        if self.rank < max_rank:
            self.consumed_indices += 1

        self._cache_batches = {}  # LRU cache
        self._max_cache_size = 3  # max cache size
        

    def _build_batches(self):
        
        if self.epoch in self._cache_batches:
            print("cache hit!")
            self.all_batches = self._cache_batches[self.epoch]
            return

        # 1. sample order
        samples = self.samples
        if not samples:
            self.all_batches = []
            return
        
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(len(samples), generator=g).tolist()
        samples = [samples[i] for i in perm]
        
        if self.sort_by_length:
            samples = sorted(samples, key=lambda x: x[1], reverse=True)
        # print(samples)

        # 2. packing
        if self.packing_method == "first_fit":
            batches = self._first_fit(samples)
        elif self.packing_method == "best_fit":
            batches = self._best_fit(samples)
        else:
            raise ValueError(f"Unknown packing_method: {self.packing_method}")

        # 3. align batch size
        total_batches = len(batches)
        step_size = self.micro_batch_size * self.num_replicas

        if total_batches % step_size != 0:
            if self.drop_last:
                # drop
                num_batches = total_batches - total_batches % step_size
                batches = batches[:num_batches]
            else:
                # padding
                padding_batches = step_size - total_batches % step_size
                batches = batches + batches[:padding_batches]

        # 4. shuffle batch order (NOT sample order)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + 1)
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]

        # 5. DDP slicing (batch-level)
        assert len(batches) % step_size == 0
        self.all_batches = batches[self.rank :: self.num_replicas]

        self._cache_batches[self.epoch] = self.all_batches
        if len(self._cache_batches) > self._max_cache_size:
            # Delete the oldest epoch data
            oldest = min(self._cache_batches.keys())
            del self._cache_batches[oldest]

    def _first_fit(self, items):
        bins = []
        bin_space = []
        for idx, size in items:
            if size > self.packing_length:
                continue
                # if self.drop_exceed: continue else TODO
            placed = False
            for b, used in enumerate(bin_space):
                if used + size <= self.packing_length:
                    bins[b].append(idx)
                    bin_space[b] += size
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_space.append(size)
        return bins
    
    def _best_fit(self, samples: List[Tuple[int, int]]) -> List[List[int]]:
        batches: List[List[int]] = []
        batch_spaces = SortedList()  # [(remaining_space, batch_idx), ...]
        
        for idx, length in samples:
            if length > self.packing_length:
                continue
                # if self.drop_exceed: continue else TODO
            
            # Binary search for the first remaining space that is >= length
            pos = batch_spaces.bisect_left((length, -1))
            if pos < len(batch_spaces):
                old_space, batch_idx = batch_spaces.pop(pos)
                batches[batch_idx].append(idx)
                new_space = old_space - length
                batch_spaces.add((new_space, batch_idx))
            else:
                batch_idx = len(batches)
                batches.append([idx])
                batch_spaces.add((self.packing_length - length, batch_idx))
        
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        # skip consumed batches
        self.batches = self.all_batches[self.consumed_indices:]
        
        if self.micro_batch_size == 1:
            for batch in self.batches:
                yield batch
        else:
            for i in range(0, len(self.batches), self.micro_batch_size):
                batch = self.batches[i : i + self.micro_batch_size]
                micro_batch = []
                for meta_batch in batch:
                    # 展平
                    # micro_batch.extend(meta_batch)
                    micro_batch.append(meta_batch)
                yield micro_batch

    def __len__(self) -> int:
        consumed_indices_per_rank = self.consumed_indices // self.num_replicas
        num_batches = len(self.all_batches) - consumed_indices_per_rank
        if self.micro_batch_size == 1:
            return max(0, num_batches)
        else:
            assert max(0, num_batches) % self.micro_batch_size == 0
            return max(0, num_batches) // self.micro_batch_size

    def set_state(self, current_epoch: int, consumed_samples = 0) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = current_epoch
        self.consumed_indices = consumed_samples// self.num_replicas
        max_rank = self.consumed_indices % self.num_replicas
        if self.rank < max_rank:
            self.consumed_indices += 1

        self._build_batches()