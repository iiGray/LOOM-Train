import torch.distributed as dist
from linktrain.core.strategy import TrainStrategy
from linktrain.core.parallel import parallel_state as parallel

class ContextParallel:
    def prepare_input(self, *args, **kwargs):
        return parallel.prepare_cp_input(*args, **kwargs)