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
        

    def _save_ckpt(self, checkpoint_config: "CheckpointConfig", inplace: bool = False, update_tag: "bool" = False):
        if self.global_step % self._get_saving_interval(checkpoint_config): return
        which = f"global_step{self.global_step}"
        tag = self.sub_dir_to_save()         
        max_ckpts = checkpoint_config.max_ckpts
        max_ckpt_GB = checkpoint_config.max_ckpts_GB

        save_dir = os.path.join(checkpoint_config.save_dir, which)
        
        if dist.get_rank() == 0:
            os.makedirs(save_dir, exist_ok = True)
        dist.barrier()

        if (not inplace) and update_tag: # None means no need to save seperately
            if dist.get_rank() == 0:
                MAX_SIZE = max_ckpt_GB * 1024**3
                subdirs = sorted([k for k in IO.read_path(save_dir) if os.path.isdir(k)],
                                key = lambda x: os.path.getmtime(x))
                
                while True:
                    total_size = sum(
                        os.path.getsize(os.path.join(dir_path, file_name))
                        for subdir in subdirs
                        for dir_path, folder_names, file_names in os.walk(subdir)
                        for file_name in file_names
                    )

                    if len(subdirs) < max_ckpts and total_size <= MAX_SIZE:
                        break

                    IO.remove(subdirs.pop(0))

            dist.barrier()
        if inplace and dist.get_rank() == 0:
            print(f"{self.__class__.__name__} Checkpoint: Inplace saving to {save_dir}/{tag} ...")
        self.save_ckpt(save_dir, tag = tag)
        
        if (not inplace) and update_tag:
            with open(os.path.join(checkpoint_config.save_dir, "latest"), "w") as f:
                f.write(which)    

        if dist.get_rank() == 0:
            print(f"{self.__class__.__name__} Checkpoint: {save_dir}/{tag} is ready !!!")

    

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
            if not os.path.exists(os.path.join(saved_dir, tag)) or inplace: return 
            if dist.get_rank() == 0:
                print(f"Successfully load {self.__class__.__name__} Checkpoint from: {saved_dir}/{tag} !!!")
            return load_result

        except Exception as e:
            if dist.get_rank() == 0:
                print(f"Fail to load {self.__class__.__name__} Checkpoint from: {saved_dir}/{tag}:",e)
