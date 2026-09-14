from dataclasses import dataclass
import torch.distributed as dist
from loomtrain.core.utils.common.iotools import IO
from loomtrain.core.utils import rank0print
from loomtrain.core.strategy import *


@dataclass
class CheckpointConfig:
    save_dir: str
    load_dir: str = None
    
    do_resume: bool = True
    ckpt_interval: int = 10
    weight_interval: int = 10
    visualization_interval: int = 1
    max_ckpts: int = 2
    max_ckpts_GB: int = 1024

    def __post_init__(self):
        if self.load_dir is None:
            self.load_dir = self.save_dir
        
        if self.visualization_interval is None:
            self.visualization_interval = min(self.weight_interval, self.ckpt_interval)



class CheckpointMixin:
    '''
    This class automatically save training status
    '''
    def __init__(self):
        self._global_step = 0
    def __repr__(self):
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"
    
    @property
    def global_step(self): return self._global_step


    def sub_dir_to_save(self) -> str:
        ''' sub_dir mainly for different types or checkpoint '''
        raise NotImplementedError
    

    def _update(self):
        '''update the state'''
        raise NotImplementedError

    def save_ckpt(self, save_dir: str, tag: str):
        raise NotImplementedError
    
    def load_ckpt(self, saved_dir: str, tag: str):
        raise NotImplementedError
    
    def _get_saving_interval(self, checkpoint_config: "CheckpointConfig") -> bool:
        '''extract saving interval from checkpoint_config and return it'''
        return checkpoint_config.ckpt_interval

    def _update_(self, *args, **kwargs):
        self._global_step += 1
        return self._update(*args, **kwargs)

    def _get_mtime_from_ckpt_path(self, path: "str"):
        mtime = float("inf")
        if not path or IO.isfile(path): return mtime        
        for sub_path in IO.read_path(path):
            if self.sub_dir_to_save() in sub_path:
                mtime = min(mtime, os.path.getmtime(sub_path))
            else: mtime = min(mtime, self._get_mtime_from_ckpt_path(sub_path))
            if mtime is not float("inf"): break
        return mtime
    def _get_ckpt_path_from_dir(self, path: "str"):
        if (path is None) or IO.isfile(path): return
        if self.sub_dir_to_save() in path: return path
        for sub_path in IO.read_path(path):
            ret = self._get_ckpt_path_from_dir(sub_path)
            if ret: return ret
            

    def _save_ckpt(self, checkpoint_config: "CheckpointConfig", inplace: "bool" = False, save_interval: "int" = None, update_tag: "bool" = False, finished: "bool" = False):
        if save_interval is None: save_interval = self._get_saving_interval(checkpoint_config)
        if (self.global_step % save_interval) and (not finished): return
        which = f"global_step{self.global_step}"
        tag = self.sub_dir_to_save()         
        max_ckpts = checkpoint_config.max_ckpts
        max_ckpt_GB = checkpoint_config.max_ckpts_GB

        save_dir = os.path.join(checkpoint_config.save_dir, which)
        
        dist.barrier()

        if (not inplace) and update_tag: # None means no need to save seperately
            if dist.get_rank() == 0:
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

            dist.barrier()
        if dist.get_rank() == 0 and inplace:
            print(f"{self.__class__.__name__} Checkpoint: Inplace saving to {checkpoint_config.save_dir}/ ...")
        self.save_ckpt(save_dir, tag = tag)
        
        if (not inplace) and update_tag:
            with open(os.path.join(checkpoint_config.save_dir, "latest"), "w") as f:
                f.write(which)    

        if (dist.get_rank() == 0) and (not inplace):
            print(f"{self.__class__.__name__} Checkpoint: {save_dir}/ is ready !!!")

    

    def _load_ckpt(self, checkpoint_config: "CheckpointConfig", inplace: bool = False):
        self.checkpoint_config = checkpoint_config
        saved_dir = checkpoint_config.save_dir
        latest_path = os.path.join(saved_dir, "latest")
        tag = self.sub_dir_to_save()
        
        if not inplace:
            if not os.path.exists(saved_dir) or (not os.path.exists(latest_path)):
                rank0print(f"Make sure that this is the first training process,"
                            f" because ckpt path:`{saved_dir}` doesn't exist .")
                return
            
        if os.path.exists(latest_path): 
            with open(latest_path, "r") as f:
                which = f.read().strip()
            self._global_step = int(which.replace("global_step",""))
            saved_dir = os.path.join(saved_dir, which)
        try:
            load_result = self.load_ckpt(saved_dir, tag) 
            if not os.path.exists(saved_dir) or inplace: return 
            if dist.get_rank() == 0:
                print(f"Successfully load {self.__class__.__name__} Checkpoint from: {saved_dir}/ !!!")
            return load_result

        except Exception as e:
            if dist.get_rank() == 0:
                print(f"Fail to load {self.__class__.__name__} Checkpoint from: {saved_dir}/:",e)
