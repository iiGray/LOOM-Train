import os, wandb
from typing import Any, Literal
from dataclasses import dataclass
from collections import UserDict
import torch
from torch.utils.tensorboard import SummaryWriter
from linktrain.core.arguments import args
from linktrain.core.state import CheckpointMixin
from linktrain.core.parallel import parallel_state as parallel
from linktrain.core.utils import (
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

    # @rank0only_decorator
    def _save_ckpt(self, *args, **kwargs): ...

    # @rank0only_decorator
    def _load_ckpt(self, *args, **kwargs): ...

    # @rank0only_decorator
    def _update_(self, *args, **kwargs): ...
    
    # @rank0only_decorator
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
        
        self.start_logs_dict = dict()

    # @rank0only_decorator
    def _init_tensorboard(self, tensorboard_config: TensorboardConfig):
        self.tensorboard_config = tensorboard_config
        if tensorboard_config:
            IO.mkdir(tensorboard_config.log_dir)
            log_dir = os.path.join(tensorboard_config.log_dir, self.sub_dir_to_save())
            self._tensorboard = SummaryWriter(log_dir = log_dir)
    
    # @rank0only_decorator
    def _update_tensorboard(self, logs_dict:dict, global_step: int, logging_steps:int = 1):
        if self.tensorboard_config and global_step % logging_steps == 0:
            for k, v in logs_dict.items():
                self._tensorboard.add_scalar(k, v, global_step)
            

    # @rank0only_decorator
    def _release_tensorboard(self):
        if self.tensorboard_config:
            self._tensorboard.close()

    # @rank0only_decorator
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
    
    # @rank0only_decorator
    def _update_wandb(self, logs_dict:dict, global_step:int, logging_steps:int = 1):
        if self.wandb_config and global_step % logging_steps == 0:
            wandb.log(logs_dict, step = global_step)


    # @rank0only_decorator
    def _release_wandb(self):
        if self.wandb_config:
            wandb.finish()


    # @rank0only_decorator
    def _update(self, logs_dict: "dict[str, Accum | object]"):
        self.logs_dict = logs_dict
        logs_dict = {k: v.get_value() if isinstance(v, Accum) else v for k, v in logs_dict.items()}

        global_step = self.global_step
        logging_steps = self.logging_steps
        self._update_tensorboard(logs_dict, global_step, logging_steps)
        self._update_wandb(logs_dict, global_step, logging_steps)



    def sub_dir_to_save(self): 
        return self.logtype

    # @rank0only_decorator
    def save_ckpt(self, save_dir, tag):
        #TODO: save in save_dir/tensorboard
        visualized_path = path_join(dirname(save_dir), "visualized_stats.pkl")
        save_pkl(self.logs_dict, visualized_path)
        return
        return super().save_ckpt(save_dir, tag)

    # @rank0only_decorator
    def load_ckpt(self, saved_dir, tag, do_resume):
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
        if os.path.exists(visualized_path) and do_resume:
            self.start_logs_dict = read_pkl(visualized_path)
        
            return self.start_logs_dict

    def _save_ckpt(self, checkpoint_config: "CheckpointConfig", inplace: "bool" = False, save_interval: "int" = None, update_tag: "bool" = False, finished: "bool" = False):
        if save_interval is None: save_interval = self._get_saving_interval(checkpoint_config)
        if (self.global_step % save_interval) and (not finished): return
        which = f"global_step{self.global_step}"
        tag = self.sub_dir_to_save()         
        max_ckpts = checkpoint_config.max_ckpts
        max_ckpt_GB = checkpoint_config.max_ckpts_GB

        save_dir = os.path.join(checkpoint_config.save_dir, which)

        if (not inplace) and update_tag: # None means no need to save seperately
            os.makedirs(checkpoint_config.save_dir, exist_ok = True)
            MAX_SIZE = max_ckpt_GB * 1024**3
            subdirs = sorted([self._get_ckpt_path_from_dir(k) for k in IO.read_path(checkpoint_config.save_dir) if os.path.isdir(k)],
                            key = lambda x: self._get_mtime_from_ckpt_path(x))
            subdirs = [k for k in subdirs if k is not None and "global_step" in k]

            while True:
                total_size = sum(
                    os.path.getsize(os.path.join(dir_path, file_name))
                    for subdir in subdirs
                    for dir_path, folder_names, file_names in os.walk(subdir)
                    for file_name in file_names
                )

                if len(subdirs) < max_ckpts and total_size <= MAX_SIZE:
                    break

                removed = subdirs.pop(0)
                removed_dir = dirname(removed)
                IO.remove(removed)
                if not [k for k in IO.read_path(removed_dir) if IO.isdir(k)]:
                    IO.remove(dirname(removed_dir))

        if inplace:
            print(f"{self.__class__.__name__} Checkpoint: Inplace saving to {checkpoint_config.save_dir}/ ...")
        self.save_ckpt(save_dir, tag = tag)
        
        if (not inplace) and update_tag:
            with open(os.path.join(checkpoint_config.save_dir, "latest"), "w") as f:
                f.write(which)    

        if (not inplace):
            print(f"{self.__class__.__name__} Checkpoint: {save_dir}/ is ready !!!")

    def _load_ckpt(self, checkpoint_config: "CheckpointConfig", inplace: bool = False):
        self.checkpoint_config = checkpoint_config
        saved_dir = checkpoint_config.save_dir
        latest_path = os.path.join(saved_dir, "latest")
        tag = self.sub_dir_to_save()
        
        if not inplace:
            if not os.path.exists(saved_dir) or (not os.path.exists(latest_path)):
                print(f"Make sure that this is the first training process,"
                            f" because ckpt path:`{saved_dir}` doesn't exist .")
                return
            
        if os.path.exists(latest_path): 
            with open(latest_path, "r") as f:
                which = f.read().strip()
            self._global_step = int(which.replace("global_step",""))
            saved_dir = os.path.join(saved_dir, which)
        try:
            load_result = self.load_ckpt(saved_dir, tag, checkpoint_config.do_resume) 
            if not os.path.exists(saved_dir) or inplace: return load_result
            print(f"Successfully load {self.__class__.__name__} Checkpoint from: {saved_dir}/ !!!")
            return load_result

        except Exception as e:
            print(f"Fail to load {self.__class__.__name__} Checkpoint from: {saved_dir}/:",e)



    # @rank0only_decorator
    def release(self):
        self._release_tensorboard()
        self._release_wandb()



class Accum:
    '''
    Args:
        total: 
            Only work when dtype == 'mean'. 
            When dtype == 'sum', total is useless, keep it '0' is OK.
            when total is None, total will be automatically set as the total samples in a batch
        is_global: whether refresh during different batch
    '''
    def __init__(self, value: "Any" = 0, total: "int" = None, dtype: "Literal['sum', 'mean']" = "mean", all_reduce: "None | Literal['sum', 'mean']" = None, is_global: "bool" = False):
        if all_reduce:
            value = parallel.all_reduce(value, op = all_reduce)
        self.value = value
        self.total = total
        self.dtype = dtype
        self.is_global = is_global


        self._user_set_total = total != None

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
        if self._user_set_total: return self
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