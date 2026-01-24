import os, torch, transformers
from typing import Callable, Literal, TYPE_CHECKING
from collections import defaultdict
from torch import nn
import torch.utils.data as tud
import torch.distributed as dist
from datetime import timedelta
from loomtrain.dataset.base import CollateDataset
from loomtrain.core.data.dataloader.base import DataLoaderStateDict
from loomtrain.core.data.dataloader.iter import MapDataLoader

# from loomtrain.core.device.mesh import DeviceMes

from loomtrain.core.utils import *
from loomtrain.core.arguments import add_extra_arguments_by, args
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.datamodule import DataModule

if TYPE_CHECKING:
    from loomtrain.core.module import Module
    from loomtrain.core.modeling.actor import Actor
    from loomtrain.core.data.sampler import *

from dataclasses import dataclass
from functools import partial


@dataclass
class DataConfig:
    collate_type: Literal["packing", "padding"]
    packing_length: "int" = None # work while collate_type is packing
    
    global_batch_size: int = 1
    micro_batch_size: int = 1
    val_batch_size: int = 1
    val_interval: "int" = None
    num_epochs: "int" = 1
    pin_memory: "bool" = False
    shuffle: "bool" = True
    drop_last: "bool" = True
    drop_exceed: "bool" = False

    @property
    def grad_accum(self):
        return self.global_batch_size * parallel.get_cp_size() // self.micro_batch_size // parallel.get_world_size()

class DataStrategy: 
    '''
    prepare dataloader (This class is mainly designed for different data packing algorithms)
    '''
    def __init__(self,
                 parallel_config: "parallel.ParallelConfig" = None,
                 data_config: "DataConfig" = None,
                 full_determinism: "bool" = False,
                 seed:int = 42):
        if parallel_config is None:
            parallel_config = parallel.ParallelConfig(
                nnodes = args().nnodes,
                devices_per_node = args().devices_per_node,
                cp_size = args().cp_size,
                cp_type = args().cp_type,
                cp_args = args().cp_args
            )
        if data_config is None:
            data_config = DataConfig(
                collate_type = args().collate_type,
                packing_length = args().packing_length,
                global_batch_size = args().global_batch_size,
                micro_batch_size = args().micro_batch_size,
                val_batch_size = args().val_batch_size,
                val_interval = args().val_interval,
                num_epochs = args().num_epochs,
                pin_memory = args().pin_memory,
                shuffle = args().shuffle,
                drop_last = args().drop_last,
                drop_exceed = args().drop_exceed
            )
        if full_determinism is None:
            full_determinism = args().full_determinism
        if seed is None:
            seed = args().seed

        self.parallel_config = parallel_config
        self.data_config = data_config
        self.full_determinism = full_determinism
        self.seed = seed

        self.dp_size = self.parallel_config.dp_size
        self.num_replicas = self.parallel_config.dp_size

        self._rank = None
    
    @property
    def rank(self):
        if self._rank is None:
            assert parallel.is_initialized()
            self._rank = parallel.get_dp_rank()
        return self._rank

    @property
    def train_data_iter(self) -> "MapDataLoader":
        return self.datamodule.train_data_iter

    @property
    def val_data_iter(self) -> "MapDataLoader":
        return self.datamodule.val_data_iter


    def _connect_datamodule(self, datamodule: "DataModule"):
        assert isinstance(datamodule, DataModule)
        self.datamodule = datamodule


    def setup_train_data_iter(self):
        raise NotImplementedError
        
    def setup_val_data_iter(self):
        raise NotImplementedError

    def collate_fn(self, item_list):
        raise NotImplementedError

    def load_sampler_fn(self, sampler: "DistributedSampler | DistributedBucketSampler", consumed_epoch: int , consumed_samples: int):
        raise NotImplementedError

    def save_ckpt(self, save_dir: str, tag: str = None):
        save_json(self.train_data_iter.get_state(), path_join(save_dir, "dataIter_states.json"))
    
    def load_ckpt(self, saved_dir: str, tag: str = None):
        self.current_epoch = 0
        self.consumed_samples = 0
        if IO.exists(saved_dir) and IO.exists(path_join(saved_dir, "dataIter_states.json")):
            states = DataLoaderStateDict(** read_json(path_join(saved_dir, "dataIter_states.json")))
            self.current_epoch = states.current_epoch
            self.consumed_samples = states.consumed_samples
        
        self.train_data_iter.set_state(current_epoch = self.current_epoch, 
                                       consumed_samples = self.consumed_samples)
        
    def setup_data_iter(self, 
                        dataset: "tud.Dataset") -> "MapDataLoader":
        raise NotImplementedError


@dataclass
class OptimConfig:
    lr   : "float" = 1e-5
    min_lr: "float" = None
    betas: "tuple" = (0.9, 0.95)
    L2_weight_decay: float = 0.0
    lr_type: Literal["linear",
                    "cosine",
                    "cosine_with_restarts",
                    "polynomial",
                    "constant",
                    "constant_with_warmup",
                    "inverse_sqrt",
                    "reduce_lr_on_plateau",
                    "cosine_with_min_lr",
                    "warmup_stable_decay"] = "cosine_with_min_lr"
    warmup_ratio: "int" = 0.03
    total_steps = None
    @property
    def num_warmup_steps(self):
        if not hasattr(self, "_num_warmup_steps"):
            assert self.total_steps, "Module should connect DataModule first."
            self._num_warmup_steps = round(self.total_steps * self.warmup_ratio)

        return self._num_warmup_steps

# TBD
class TrainStrategy:
    '''
    Prepare model, optimizer, scheduler
    '''

    def __init__(self,
                 parallel_config: "parallel.ParallelConfig" = None,
                 data_config: "DataConfig" = None,
                 full_determinism: "bool" = None,
                 seed: "int" = None):
        if parallel_config is None:
            parallel_config = parallel.ParallelConfig(
                nnodes = args().nnodes,
                devices_per_node = args().devices_per_node,
                cp_size = args().cp_size,
                cp_type = args().cp_type,
                cp_args = args().cp_args
            )
        if data_config is None:
            data_config = DataConfig(
                collate_type = args().collate_type,
                packing_length = args().packing_length,
                global_batch_size = args().global_batch_size,
                micro_batch_size = args().micro_batch_size,
                val_batch_size = args().val_batch_size,
                val_interval = args().val_interval,
                num_epochs = args().num_epochs,
                pin_memory = args().pin_memory,
                shuffle = args().shuffle,
                drop_last = args().drop_last,
                drop_exceed = args().drop_exceed
            )
        
        self.parallel_config = parallel_config
        self.data_config = data_config

        self._optim_configs_ = None

        if full_determinism is None:
            full_determinism = args().full_determinism
        if seed is None:
            seed = args().seed
        
        self.full_determinism = full_determinism
        self.seed = seed
            
        self.global_batch_size = self.data_config.global_batch_size
        self.micro_batch_size = self.data_config.micro_batch_size


    def _connect_module(self, module: "Module"):
        self.module = module
        self.optim_configs = module.optim_configs

    def _connect_datamodule(self, datamodule: "DataModule"):
        self.datamodule = datamodule
        for optim_group in self.optim_configs.values():
            optim_group.total_steps = datamodule.total_train_steps

    @property
    def opt_groups(self): return self.module.actors

    @property
    def optim_configs(self):
        if not self._optim_configs_:
            raise ValueError("optim_configs has not been set yet. please connect strategy to module first.")
        return self._optim_configs_
    
    @optim_configs.setter
    def optim_configs(self, optim_configs: "dict[str, OptimConfig]"):
        self._optim_configs_ = optim_configs
    
    def get_submodule(self, name_path: str):
        if name_path.startswith("."): 
            name_path = "module" + name_path
        elif not name_path.startswith("module"):
            name_path = "module." + name_path
        name_list = name_path.split(".")
        module = self
        for name in name_list:
            module = getattr(module, name)
        return module
    def split_submodules_by_actors(self) -> "dict[str, dict[str, OptimConfig]]":
        dicts = defaultdict(dict)
        if len(self.optim_configs) == 1 and "module" in self.optim_configs:
            for actor_name in self.opt_groups.keys():
                dicts[actor_name] = {f"module.{actor_name}": self.optim_configs["module"]}
            return dicts
        
        for name_path, cfg in self.optim_configs.items():
            if name_path.startswith("."): name_path = "module" + name_path
            dicts[name_path.split(".")[1]][name_path] = cfg
        return dicts   
        
    def setup_distributed(self):
        self.set_seed()
        self.set_device()
        self.init_distributed()
        self.init_parallel()
    

    def init_distributed(self):
        raise NotImplementedError

    def config_module(self): ...


    def save_ckpt(self, save_dir: "str", tag: "str"):
        raise NotImplementedError

    def load_ckpt(self, saved_dir: "str", tag: "str"):
        raise NotImplementedError

    def save_module(self, save_dir: "str", tag: "str"):
        raise NotImplementedError

    def backward(self, loss: "torch.Tensor", actor_of_the_loss: "Actor" = None):
        loss.sum().backward()

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError


    def micro_batch_forward_backward(self, batch):
        raise NotImplementedError
    
    def micro_batch_validate_forward(self, batch):
        raise NotImplementedError

    def non_accum_logs_per_step(self):
        return dict()

    def init_parallel(self):
        parallel.initialize(self.parallel_config)
        self.grad_accum = self.data_config.grad_accum

    @property
    def world_size(self):
        if not hasattr(self, "_world_size"):
            self._world_size = dist.get_world_size()
        return self._world_size

    @property
    def rank(self):
        if not hasattr(self, "_rank"):
            self._rank = dist.get_rank()
        return self._rank

    def get_local_rank(self):
        return int(os.environ["LOCAL_RANK"])
    
    def get_local_world_size(self):
        return int(os.environ["LOCAL_WORLD_SIZE"])

    def set_seed(self):
        if self.full_determinism:
            transformers.enable_full_determinism(self.seed)
            transformers.modeling_flash_attention_utils.deterministic_g = True
        else:
            transformers.set_seed(self.seed)
    
    def set_device(self):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))



    def set_submodule(self, 
                      module: torch.nn.Module, 
                      submodule_name: str,
                      submodule: torch.nn.Module):
        submodules = module.__dict__.get("_modules")
        submodules[submodule_name] = submodule
        module.__dict__[submodule_name] = submodule



    # def backward(self, loss: torch.Tensor, model: "LoomModule"):
    #     model.model.backward(loss)
    
    # def optimizer_step(self, model: "LoomModule"):
    #     model.model.step()


    # def zero_grad(self, model: "LoomModule"):
    #     engine = model.model
        
    #     if engine.bfloat16_enabled():
    #         # TODO: Temporary until bf16_optimizer and zero_optimizer are integrated
    #         if engine.zero_optimization() and hasattr(engine.optimizer, "zero_grad"):
    #             engine.optimizer.zero_grad()
    #         else:
    #             pass
    #     elif engine.zero_optimization() or engine.fp16_enabled() or engine.amp_enabled():
    #         engine.optimizer.zero_grad()
    #     else:
    #         engine.zero_grad()
