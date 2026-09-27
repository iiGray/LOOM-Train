import functools
from typing import Literal
import torch
import torch.distributed as dist

def rank0only_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if dist.get_rank() == 0:
            return func(*args, **kwargs)
    return wrapper

@rank0only_decorator
def rank0print(*values: object, sep: str | None = " ", end: str | None = "\n", file = None, flush: Literal[False] = False) -> None:
    print(*values, sep, end, file, flush)

@rank0only_decorator
def rank0breakpoint():
    breakpoint()

def rankibreakpoint(rank: int):
    if dist.get_rank() == rank:
        breakpoint()





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



# def all_gather(self, data):
#     if isinstance(data, dict):
#         ret = {}
#         for k, v in data.items():
#             ret[k] = self.all_gather(v)
#         return ret
#     else:
#         if not isinstance(data, torch.Tensor):
#             data = torch.Tensor([data])
#         is_cpu_tensor = data.device.type == "cpu"

#         ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
#         dist.all_gather(ret, data.to(torch.cuda.current_device()))
#         return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)