import os, wandb
from typing import Any, Literal
from dataclasses import dataclass
from collections import UserDict
import torch
from torch.utils.tensorboard import SummaryWriter
from loomtrain.core.state import CheckpointMixin
from loomtrain.core.utils import (
    IO, rank0only_decorator, dirname, path_join, save_pkl, read_pkl
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

class NoneVisualization(CheckpointMixin):
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

class VisualizationModule(CheckpointMixin):
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
            log_dir = os.path.join(tensorboard_config.log_dir, self.sub_dir_to_save())
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
    def _update(self, logs_dict: "dict[str, Accum | object]"):
        self.logs_dict = logs_dict
        logs_dict = {k: v.get_value() if isinstance(v, Accum) else v for k, v in logs_dict.items()}

        global_step = self.global_step
        logging_steps = self.logging_steps
        self._update_tensorboard(logs_dict, global_step, logging_steps)
        self._update_wandb(logs_dict, global_step, logging_steps)



    def sub_dir_to_save(self): 
        return self.logtype

    @rank0only_decorator
    def save_ckpt(self, save_dir, tag):
        #TODO: save in save_dir/tensorboard
        visualized_path = path_join(dirname(save_dir), "visualized_stats.pkl")
        save_pkl(self.logs_dict, visualized_path)
        return
        return super().save_ckpt(save_dir, tag)

    @rank0only_decorator
    def load_ckpt(self, saved_dir, tag):
        self.logging_steps = self.checkpoint_config.visualization_interval
        if self.logtype == 'tensorboard':
            self._init_tensorboard(
                TensorboardConfig(
                    log_dir = saved_dir,
                    name = "tensorboard"
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
        
        visualized_path = os.path.join(saved_dir, "visualized_stats.pkl")
        if os.path.exists(visualized_path):
            return read_pkl(visualized_path)


    @rank0only_decorator
    def release(self):
        self._release_tensorboard()
        self._release_wandb()



class Accum:
    '''
    total: Only work when dtype == 'mean'. When dtype == 'sum', total is useless, keep it '0' is OK.
    is_global: whether refresh during different batch
    '''
    def __init__(self, value: "Any" = 0, total: "int" = 0, dtype: "Literal['sum', 'mean']" = "mean", is_global: "bool" = False):
        self.value = value
        self.total = total
        self.dtype = dtype
        self.is_global = is_global
    def __iadd__(self, other: "Accum"):
        assert self.dtype == other.dtype, \
            f"Only Accums with the same dtype can be summed up, but you provide :{self} and {other}"
        self.value += other.value
        self.total += other.total
        return self

    def __add__(self, other: "Accum"):
        assert self.dtype == other.dtype, \
            f"Only Accums with the same dtype can be summed up, but you provide :{self} and {other}"
        return Accum(self.value + other.value,
                           self.total + other.total)
    def __repr__(self):
        return f"Accum(value = {self.value}, total = {self.total}, dtype = {self.dtype})"
    
    def reset(self):
        self.value = 0
        self.total = 0
        return self
    
    def set_total(self, total: "int"):
        self.total = total
        return self
    
    def get_value(self):
        if self.total == 0 and self.dtype == "mean": return None
        value = self.value
        if isinstance(value, torch.Tensor):
            value = value.item()
        return value / self.total if self.dtype == "mean" else value
    
class AccumLogDict(UserDict[str, Accum]):
    """
    A specialized dictionary that enforces strict type constraints for keys and values.

    This dictionary implementation ensures that all keys are strings and all values
    are instances of the `Accum` class. It is designed to maintain data integrity
    for logging or aggregation tasks.

    Constraints:
        - Keys: Must be of type `str`.
        - Values: Must be of type `Accum`.

    Raises:
        TypeError: If a key is not a `str` or a value is not an `Accum` instance
            during item assignment or update operations.

    Example:
        >>> log = AccumLogDict(tokens = Accum(0, dtype = "sum"))
        >>> log['loss'] = Accum(0.5)  # OK
        >>> log[100] = Accum(0.5)     # Raises TypeError (Key must be str)
        >>> log['acc'] = 0.95         # Raises TypeError (Value must be Accum)
    """
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError(f"Key must be type `str`, but you provide: {type(key).__name__}")
        if not isinstance(value, Accum):
            raise TypeError(f"Value must be type `Accum`, but you provide: {type(value).__name__}")
        super().__setitem__(key, value)


class LogDict(UserDict[str, Accum]):
    """
    A specialized dictionary that enforces strict type constraints for keys and values.

    This dictionary implementation ensures that all keys are strings and all values
    are instances of the `Accum` class. It is designed to maintain data integrity
    for logging or aggregation tasks.

    Constraints:
        - Keys: Must be of type `str`.

    Raises:
        TypeError: If a key is not a `str`
            during item assignment or update operations.

    Example:
        >>> log = LogDict(lr = 5e-6)
        >>> log['loss'] = Accum(0.5)  # OK
        >>> log[100] = Accum(0.5)     # Raises TypeError (Key must be str)
    """
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError(f"Key must be type `str`, but you provide: {type(key).__name__}")
        super().__setitem__(key, value)