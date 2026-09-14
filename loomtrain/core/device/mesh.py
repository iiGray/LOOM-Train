from typing import Literal, Tuple, Union
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from loomtrain.core.parallel.parallel_state import ParallelConfig


class DeviceMesh:
    def __init__(
            self,
            parallel_config: "ParallelConfig" = ParallelConfig()
    ):
        assert torch.distributed.is_initialized(), \
            "You should initialize the distributed env before instantiate this class !"
        self.parallel_config = parallel_config
        self.parallel_type2rank = OrderedDict()


        devices = torch.arange(self.world_size).reshape(
            *(t for t in reversed(self.shape))
        )
        shifts_order = torch.arange(devices.ndim).roll(shifts = 1).tolist()

        for parallel_type in parallel_config.order:
            self.parallel_type2rank[parallel_type] = devices.transpose(-1, 0).flatten(start_dim = 1).transpose(0, 1).tolist()
            devices = devices.permute(*shifts_order)

    @property
    def world_size(self):
        return self.parallel_config.expected_world_size
    @property
    def shape(self):
        return self.parallel_config.shape

    def __getitem__(self, parallel_type: Literal["tp", "sp", "cp", "pp", "dp"]) -> list:
        global_rank = dist.get_rank()
        current_rank_lists = self.parallel_type2rank[parallel_type]
        for rank_list in current_rank_lists:
            if global_rank in rank_list:
                return rank_list
        raise RuntimeError("Rank Error!")



