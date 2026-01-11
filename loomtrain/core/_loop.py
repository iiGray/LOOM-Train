from loomtrain.core.trainer import LoomTrainer
from loomtrain.core.module import LoomModule
from loomtrain.core.datamodule import LoomDataModule
from loomtrain.core.state import CheckpointConfig
from loomtrain.core.distributed_sampler import *


class _Loop:
    def __init__(self, 
                 trainer: "LoomTrainer",
                 module: "LoomModule",
                 datamodule: "LoomDataModule",
                 checkpoint_config: "CheckpointConfig"):
        self.trainer = trainer
        self.module = module
        self.datamodule = datamodule
        self.checkpoint_config = checkpoint_config
    
    @property
    def done(self) -> bool:
        '''Loop Ending Condition'''
        return False

    def setup_variables(self): 
        '''
        setup the variables soon used in : 
        `on_execute_start` `execute` `on_execute_end`

        do not allow set self.trainer
        '''
        ...
    def on_execute_start(self): ...
    def on_execute_end(self): ...
    def execute(self): raise NotImplementedError

    def run(self):
        self.on_execute_start()
        while not self.done:
            try: self.execute()
            except StopIteration: break
        self.on_execute_end()




class _FitLoop(_Loop):

    def on_execute_start(self):
        if self.checkpoint_config.do_resume:
            self.module._load_ckpt(saved_dir = self.checkpoint_config.save_dir)
            self.datamodule._load_ckpt(saved_dir = self.checkpoint_config.save_dir)

        self.module.train()
        self.datamodule.train()
    
    @property
    def done(self) -> bool:
        return self.datamodule.train_data_iter.exhausted
        
    def execute(self):
        for batches in self.datamodule.train_data_iter:
            if batches is None: return
            
    
    def on_execute_end(self):
        ...

