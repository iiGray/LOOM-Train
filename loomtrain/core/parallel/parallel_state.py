from typing import Literal, List, Any
from dataclasses import dataclass, field 
import torch
import torch.distributed as dist

@dataclass
class ParallelConfig:
    nnodes: int = 1
    devices_per_node: int = 8
    cp_size:int = 8
    pp_size:int = 1
    sp_size:int = 1
    tp_size:int = 1
    cp_type: Literal["ring"] = "ring"
    cp_args: "dict" = field(default_factory = lambda: dict(head_stride = 1))
    order = ("tp", "sp", "cp", "pp", "dp")

    @property
    def shape(self):
        return tuple([getattr(self, o + "_size") for o in self.order])

    @property
    def expected_order(self):
        return self.order

    @property
    def expected_world_size(self):
        return self.nnodes * self.devices_per_node

    @property
    def size_expect_dp(self):
        return self.cp_size * self.pp_size * self.sp_size * self.tp_size
    
    @property
    def dp_size(self):
        if not hasattr(self, "_dp"):
            self._dp = self.expected_world_size // self.size_expect_dp
        return self._dp

    @property
    def tp(self): return self.tp_size
    @property
    def sp(self): return self.sp_size
    @property
    def cp(self): return self.cp_size
    @property
    def pp(self): return self.pp_size
    @property
    def dp(self): return self.dp_size
    

    def __post_init__(self):
        assert len(self.order) == 5, str(self.order)
        for p in ("cp", "dp", "pp", "sp", "tp"):
            assert p in self.order, str(self.order)
        

        assert self.expected_world_size % self.size_expect_dp == 0, \
            f"The Nodes({self.nnodes}) * Devices({self.devices_per_node}) you provide doesn't match the parallel config:({self.shape})"



_CP_GROUP_: dist.ProcessGroup = None
_DP_GROUP_: dist.ProcessGroup = None
_SP_GROUP_: dist.ProcessGroup = None
_TP_GROUP_: dist.ProcessGroup = None
_PP_GROUP_: dist.ProcessGroup = None

_CP_RANKS_: list = None
_DP_RANKS_: list = None
_SP_RANKS_: list = None
_TP_RANKS_: list = None
_PP_RANKS_: list = None

_CP_RANK_: int = None
_DP_RANK_: int = None
_SP_RANK_: int = None
_TP_RANK_: int = None
_PP_RANK_: int = None

_CP_WORLD_SIZE_: int = None
_DP_WORLD_SIZE_: int = None
_SP_WORLD_SIZE_: int = None
_TP_WORLD_SIZE_: int = None
_PP_WORLD_SIZE_: int = None

_RANK_: int = None
_WORLD_SIZE_: int = None

_IS_INITIALIZED_: bool = False

_ALLOW_DISTRIBUTED_: bool = True

def get_cp_group():
    assert _CP_GROUP_ is not None
    return _CP_GROUP_

def get_dp_group():
    assert _DP_GROUP_ is not None
    return _DP_GROUP_

def get_sp_group():
    assert _SP_GROUP_ is not None
    return _SP_GROUP_

def get_tp_group():
    assert _TP_GROUP_ is not None
    return _TP_GROUP_

def get_pp_group():
    assert _PP_GROUP_ is not None
    return _PP_GROUP_

def set_cp_group(group: dist.ProcessGroup):
    global _CP_GROUP_
    _CP_GROUP_ = group

def set_dp_group(group: dist.ProcessGroup):
    global _DP_GROUP_
    _DP_GROUP_ = group

def set_sp_group(group: dist.ProcessGroup):
    global _SP_GROUP_
    _SP_GROUP_ = group

def set_tp_group(group: dist.ProcessGroup):
    global _TP_GROUP_
    _TP_GROUP_ = group

def set_pp_group(group: dist.ProcessGroup):
    global _PP_GROUP_
    _PP_GROUP_ = group


def get_cp_ranks():
    assert _CP_RANKS_ is not None
    return _CP_RANKS_
    
def get_dp_ranks():
    assert _DP_RANKS_ is not None
    return _DP_RANKS_
    
def get_sp_ranks():
    assert _SP_RANKS_ is not None
    return _SP_RANKS_
    
def get_tp_ranks():
    assert _TP_RANKS_ is not None
    return _TP_RANKS_
    
def get_pp_ranks():
    assert _PP_RANKS_ is not None
    return _PP_RANKS_


def set_cp_ranks(ranks: List[int]):
    global _CP_RANKS_ 
    _CP_RANKS_ = ranks

def set_dp_ranks(ranks: List[int]):
    global _DP_RANKS_ 
    _DP_RANKS_ = ranks

def set_sp_ranks(ranks: List[int]):
    global _SP_RANKS_ 
    _SP_RANKS_ = ranks

def set_tp_ranks(ranks: List[int]):
    global _TP_RANKS_ 
    _TP_RANKS_ = ranks

def set_pp_ranks(ranks: List[int]):
    global _PP_RANKS_ 
    _PP_RANKS_ = ranks


def get_cp_rank():
    assert _CP_RANK_ is not None
    return _CP_RANK_
    
def get_dp_rank():
    assert _DP_RANK_ is not None
    return _DP_RANK_
    
def get_sp_rank():
    assert _SP_RANK_ is not None
    return _SP_RANK_
    
def get_tp_rank():
    assert _TP_RANK_ is not None
    return _TP_RANK_
    
def get_pp_rank():
    assert _PP_RANK_ is not None
    return _PP_RANK_


def set_cp_rank(rank: int):
    global _CP_RANK_ 
    _CP_RANK_ = rank

def set_dp_rank(rank: int):
    global _DP_RANK_ 
    _DP_RANK_ = rank

def set_sp_rank(rank: int):
    global _SP_RANK_ 
    _SP_RANK_ = rank

def set_tp_rank(rank: int):
    global _TP_RANK_ 
    _TP_RANK_ = rank

def set_pp_rank(rank: int):
    global _PP_RANK_ 
    _PP_RANK_ = rank


def get_cp_size():
    assert _CP_WORLD_SIZE_ is not None
    return _CP_WORLD_SIZE_
    
def get_dp_size():
    assert _DP_WORLD_SIZE_ is not None
    return _DP_WORLD_SIZE_
    
def get_sp_size():
    assert _SP_WORLD_SIZE_ is not None
    return _SP_WORLD_SIZE_
    
def get_tp_size():
    assert _TP_WORLD_SIZE_ is not None
    return _TP_WORLD_SIZE_
    
def get_pp_size():
    assert _PP_WORLD_SIZE_ is not None
    return _PP_WORLD_SIZE_


def set_cp_size(size: int):
    global _CP_WORLD_SIZE_ 
    _CP_WORLD_SIZE_ = size

def set_dp_size(size: int):
    global _DP_WORLD_SIZE_ 
    _DP_WORLD_SIZE_ = size

def set_sp_size(size: int):
    global _SP_WORLD_SIZE_ 
    _SP_WORLD_SIZE_ = size

def set_tp_size(size: int):
    global _TP_WORLD_SIZE_ 
    _TP_WORLD_SIZE_ = size

def set_pp_size(size: int):
    global _PP_WORLD_SIZE_ 
    _PP_WORLD_SIZE_ = size



def get_cp_count():
    return get_world_size() // get_cp_size()
    
def get_dp_count():
    return get_world_size() // get_dp_size()
    
def get_sp_count():
    return get_world_size() // get_sp_size()
    
def get_tp_count():
    return get_world_size() // get_tp_size()
    
def get_pp_count():
    return get_world_size() // get_pp_size()



def get_parallel_group(type: Literal["tp", "sp", "cp", "pp", "dp"]):
    return globals()[f"get_{type}_group"]()

def set_parallel_group(group: dist.ProcessGroup, type: Literal["tp", "sp", "cp", "pp", "dp"]):
    return globals()[f"set_{type}_group"](group)    


def get_parallel_ranks(type: Literal["tp", "sp", "cp", "pp", "dp"]):
    return globals()[f"get_{type}_ranks"]()

def set_parallel_ranks(ranks: List[int], type: Literal["tp", "sp", "cp", "pp", "dp"]):
    return globals()[f"set_{type}_ranks"](ranks)    


def set_parallel_rank(rank: int, type: Literal["tp", "sp", "cp", "pp", "dp"]):
    return globals()[f"set_{type}_rank"](rank)


def set_parallel_size(size: int, type: Literal["tp", "sp", "cp", "pp", "dp"]):
    return globals()[f"set_{type}_size"](size)    

def get_rank():
    assert _RANK_ is not None
    return _RANK_

def is_rank0():
    assert _RANK_ is not None
    return _RANK_ == 0

def set_rank(rank: int = None):
    global _RANK_
    if rank is None: rank = dist.get_rank()
    _RANK_ = rank

def get_world_size():
    assert _WORLD_SIZE_ is not None
    return _WORLD_SIZE_

def set_world_size(size: int = None):
    global _WORLD_SIZE_
    if size is None: size = dist.get_world_size()
    _WORLD_SIZE_ = size

def barrier(group: "dist.ProcessGroup | None" = dist.GroupMember.WORLD, async_op: "bool" = False, device_ids: "Any | None" = None):
    dist.barrier(group = group, async_op = async_op, device_ids = device_ids)

def get_process_size():
    '''
    The only difference between process size and world size is that
    process size will return 1 if distributed is not initialized.
    This is useful when we want to split dataset according to process size.
    '''
    global _WORLD_SIZE_
    return 1 if _WORLD_SIZE_ is None else _WORLD_SIZE_

def get_process_rank():
    '''
    The only difference between process rank and rank is that
    process rank will return 0 if distributed is not initialized.
    This is useful when we want to split dataset according to process rank.
    '''
    global _RANK_
    return 0 if _RANK_ is None else _RANK_

def process_barrier():
    global _WORLD_SIZE_
    if _WORLD_SIZE_ is not None:
        dist.barrier()



def set_initialized():
    global _IS_INITIALIZED_
    _IS_INITIALIZED_ = True

def init_distributed(backend: str = "nccl"):
    dist.init_process_group(backend = backend)


def initialize(parallel_config: "ParallelConfig"):
    from loomtrain.core.device.mesh import DeviceMesh
    device_mesh = DeviceMesh(parallel_config)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    for parallel_type in parallel_config.order:
        parallel_group_ranks = device_mesh[parallel_type]
        assert rank in parallel_group_ranks \
            and len(parallel_group_ranks) == getattr(device_mesh.parallel_config, parallel_type), f"Group:{parallel_group_ranks}, Rank:{rank}, Parallel: {parallel_type} = {getattr(device_mesh.parallel_config, parallel_type)}"
        
        #TODO: the argument 'backend' should be configurable
        group = dist.new_group(ranks = parallel_group_ranks, backend = "nccl")

        set_parallel_group(group, parallel_type)
        set_parallel_ranks(parallel_group_ranks, parallel_type)
        set_parallel_size(len(parallel_group_ranks), parallel_type)
        set_parallel_rank(parallel_group_ranks.index(rank), parallel_type)
        set_rank(rank)
        set_world_size(world_size)
    
    _init_parallel_plugins(parallel_config)
    
    set_initialized()

def _init_parallel_plugins(parallel_config: "ParallelConfig"):
    if parallel_config.cp_type == "ring":
        from loomtrain.core.parallel.context_parallel.ring import RingFlashAttnPlugin
        RingFlashAttnPlugin().initialize(** parallel_config.cp_args)
    ... #TODO other parallel type

def is_initialized() -> bool:
    return _IS_INITIALIZED_

def enable_distributed(allow: bool = True):
    global _ALLOW_DISTRIBUTED_
    _ALLOW_DISTRIBUTED_ = allow

def is_distributed_allowed():
    global _ALLOW_DISTRIBUTED_
    return _ALLOW_DISTRIBUTED_


class ParallelPlugin:
    '''
    For different specific parallel method to inherit.
    '''
    def initialize(self):
        raise NotImplementedError



def prepare_cp_input(*args, **kwargs):
    '''
    Will soon be replaced by specific context parallel plugin after training process starts. 
    It can be sure that the replacing process is executed before it's being called.
    '''
    raise NotImplementedError
    








REDUCE_OP = dict(
    mean = dist.ReduceOp.AVG,
    max = dist.ReduceOp.MAX,
    sum = dist.ReduceOp.SUM
)

def all_reduce(data, op: Literal["mean", "max", "sum"] = "mean", group: dist.ProcessGroup = None):
    if isinstance(data, dict):
        ret = {}
        for k,v in data.items():
            ret[k] = all_reduce(v, op)

        return ret

    is_tensor = isinstance(data, torch.Tensor)
    if not is_tensor:
        data = torch.Tensor([data])
    is_cpu_tensor = data.device.type == "cpu"

    if is_cpu_tensor:
        data = data.to(torch.cuda.current_device())
    
    dist.all_reduce(data, op = REDUCE_OP[op], group = group)

    if is_cpu_tensor:
        data = data.cpu()
    
    return data if is_tensor else data.item()

