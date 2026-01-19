import os, wandb
from typing import Literal
from dataclasses import dataclass
import torch
from torch.utils.tensorboard import SummaryWriter
from loomtrain.core.utils import basename, dirname
from loomtrain.core.state import LoomCheckpointMixin
from loomtrain.core.utils import (
    IO, rank0only_decorator, rank0print,
)


@dataclass
class WandbConfig:
    api_key: str
    entity : str
    project: str
    group  : str
    name   : str
    config : dict
    reinit : bool = True


@dataclass
class TensorboardConfig:
    log_dir: str
    name : str

class NoneVisualization(LoomCheckpointMixin):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @rank0only_decorator
    def _save_ckpt(self, *args, **kwargs): ...
    @rank0only_decorator
    def _load_ckpt(self, *args, **kwargs): ...
    @rank0only_decorator
    def _update_(self, *args, **kwargs): ...
    @rank0only_decorator
    def release(self, *args, **kwargs): ...

class VisualizationModule(LoomCheckpointMixin):
    '''Default using Tensorboard config'''
    def __init__(self,
                 logtype: Literal["tensorboard","wandb"] = "tensorboard",
                 wandb_api: str = None,
                 wandb_entity: str = None,
                 wandb_project: str = None,
                 wandb_group: str = None,
                 wandb_name: str = None,
                 wandb_cfg: dict = None,
                 wandb_reinit: bool = True):
        super().__init__()

        self.logtype = logtype
        self.wandb_api = wandb_api
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        self.wandb_name = wandb_name
        self.wandb_cfg = wandb_cfg
        self.wandb_reinit = wandb_reinit

        self.is_initialized = False

    @rank0only_decorator
    def _init_tensorboard(self, tensorboard_config: TensorboardConfig):
        self.tensorboard_config = tensorboard_config
        if tensorboard_config:
            IO.mkdir(tensorboard_config.log_dir)
            log_dir = os.path.join(tensorboard_config.log_dir, tensorboard_config.name)
            self._tensorboard = SummaryWriter(log_dir = log_dir)
    
    @rank0only_decorator
    def _update_tensorboard(self, logs_dict:dict, global_step: int, logging_steps:int = 1):
        if self.tensorboard_config and global_step % logging_steps == 0:
            for k, v in logs_dict.items():
                self._tensorboard.add_scalar(k, v, global_step)
            

    @rank0only_decorator
    def _release_tensorboard(self):
        if self.tensorboard_config:
            self._tensorboard.close()

    @rank0only_decorator
    def _init_wandb(self, wandb_config: WandbConfig):
        self.wandb_config = wandb_config
        if wandb_config:
            wandb.login(key = wandb_config.api_key)
            wandb.init(
                entity = wandb_config.entity,
                project = wandb_config.project,
                group = wandb_config.group,
                name = wandb_config.name,
                config = wandb_config.config,
                reinit = wandb_config.reinit
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", 
                                step_metric = "train/global_step",
                                step_sync = True)
            
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", 
                                step_metric = "eval/global_step",
                                step_sync = True)
    
    @rank0only_decorator
    def _update_wandb(self, logs_dict:dict, global_step:int, logging_steps:int = 1):
        if self.wandb_config and global_step % logging_steps == 0:
            wandb.log(logs_dict, step = global_step)


    @rank0only_decorator
    def _release_wandb(self):
        if self.wandb_config:
            wandb.finish()


    @rank0only_decorator
    def _update(self, logs_dict:dict):
        global_step = self.global_step
        logging_steps = self.logging_steps
        self._update_tensorboard(logs_dict, global_step, logging_steps)
        self._update_wandb(logs_dict, global_step, logging_steps)


    def sub_dir_to_save(self): 
        return self.logtype

    @rank0only_decorator
    def save_ckpt(self, save_dir, tag):
        #TODO: save in save_dir/tensorboard
        return
        return super().save_ckpt(save_dir, tag)

    @rank0only_decorator
    def load_ckpt(self, saved_dir, tag):
        self.logging_steps = self.checkpoint_config.visulization_interval
        if self.logtype == 'tensorboard':
            self._init_tensorboard(
                TensorboardConfig(
                    log_dir = dirname(saved_dir),
                    name = basename(saved_dir)
                )
            )
            self.wandb_config = None
        elif self.logtype == "wandb":
            self._init_wandb(
                WandbConfig(
                    api_key = self.wandb_api,
                    entity = self.wandb_entity,
                    project = self.wandb_project,
                    group = self.wandb_group,
                    name = self.wandb_name,
                    config = self.wandb_cfg,
                    reinit = self.wandb_reinit

                )
            )
            
            self.tensorboard_config = None
        self.is_initialized = True

    @rank0only_decorator
    def release(self):
        self._release_tensorboard()
        self._release_wandb()



class Accumulator:
    def __init__(self, value = 0, total = 0):
        self.value = value
        self.total = total
    def __iadd__(self,other: "Accumulator"):
        self.value += other.value
        self.total += other.total
        return self

    def __add__(self, other: "Accumulator"):
        return Accumulator(self.value + other.value,
                           self.total + other.total)

    def reset(self):
        self.value = 0
        self.total = 0

    def get_value(self):
        if self.total == 0: return None
        value = self.value
        if isinstance(value, torch.Tensor):
            value = value.item()
        return value/self.total
    
