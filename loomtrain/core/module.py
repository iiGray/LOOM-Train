import os
import torch
from typing import TYPE_CHECKING
from torch import nn
import torch.distributed as dist
from transformers import PreTrainedTokenizer
from collections import defaultdict
from tqdm import tqdm
from loomtrain.core.metas import AttrDict, LazyInitializeMeta
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.strategy import TrainStrategy, OptimConfig
from loomtrain.core.state import CheckpointConfig, LoomCheckpointMixin
from loomtrain.core.modeling.actor import Actor
if TYPE_CHECKING:
    from loomtrain.core.datamodule import DataModule
from loomtrain.core.visualization import Accumulator
from loomtrain.utils.lora import LoRAConfig, get_peft_model
from dataclasses import dataclass
import loomtrain as lt
 


class Module(LoomCheckpointMixin, metaclass = LazyInitializeMeta):
    r"""Base class for all modules to be optimized.

    Your models should also subclass this class.

    THIS `Module` IS DIFFERENT FROM `torch.nn.Module`. 
    A `loomtrain.core.Module` IMPLEMENTS TRAINING LOGIC, OPTIMIZER,
    SCHEDULER, ETC. IT IS DESIGNED TO WORK WITH:
        `loomtrain.core.DataModule`
        `loomtrain.core.strategy.TrainStrategy` 
        `loomtrain.core.strategy.DataStrategy`
    AND BE TRAINED BY:
        `loomtrain.core.trainer.fit` 
    
    Please wrapper your WHOLE model into an Actor and assign it as an attribute of this Module.
    
    ONLY THE ACTORS IN THE MODULE WILL BE TRAINED AND SAVED !

    You may implement the forward in loomtrain.core.Actor, or in loomtrain.core.Module(suggested).
    
    Different Actors can have different Optimizer and Scheduler, which is supported by the attribute `optim_configs`:
        optim_configs: loomtrain.core.OptimConfig | dict[str, loomtrain.core.OptimConfig]
            A dictionary mapping from the name of the Actor attribute to its OptimConfig.
            Each OptimConfig contains the optimizer and scheduler configurations for the corresponding Actor.
            The keys of this dictionary must form a partition of the set of all trainable Actors in the Module.
        
        if there is only one Actor in this Module, you can directly pass an OptimConfig instance to optim_configs,
        else you must pass a dictionary, whose keys are the attribute names of the Actors to be trained. For example:

        class MyModule(loomtrain.core.Module):
            def __init__(self, optim_configs):
                super().__init__(optim_configs)
                self.actor1 = loomtrain.core.Actor(torch.nn.Linear(10,10))
                self.actor2 = loomtrain.core.Actor(torch.nn.Conv2d(3,16,3))
                # Then optim_configs must be a dict like: 
                    {
                        "actor1": loomtrain.core.OptimConfig(...), also supporting keys like: actor1.submodule1, actor1.submodule2
                        "actor2": loomtrain.core.OptimConfig(...)
                    }

    """

    def __init__(self, optim_configs: "OptimConfig | dict[str, OptimConfig]", *args, **kwargs):        
        assert parallel.is_initialized() or (not parallel.is_distributed_allowed), "One must init `Trainer` before init `Module`"
        super().__init__(*args, **kwargs)
        if isinstance(optim_configs, OptimConfig):
            # module means TrainStrategy.module
            optim_configs = {"module": optim_configs} 
        self.optim_configs = optim_configs
        self._check_optim_configs()

    def _check_optim_configs(self):
        '''The keys of the attribute dictionary optim_configs 
        must form a partition of the set of all trainable models in the module.'''
        #TODO
        return
        raise NotImplementedError

    def _initialize(self):
        '''Lazy initialize, Has been implemented in the meta class. Be intialized in `trainer.fit`'''
        return self._lazy_initialize_()


    def set_actor(self, actor_name: str, actor: "Actor"):
        setattr(self, actor_name, actor)

    @property
    def actors(self) -> "dict[str, Actor]":
        # if not hasattr(self, "_actors"):
        self._actors = AttrDict()
        for k, v in vars(self).items():
            if isinstance(v, Actor): self._actors[k] = v
        return self._actors

    @property
    def training(self):
        return next(self.actors.values()).training
    def train(self):
        for actor in self.actors.values():
            actor.train()
    def eval(self):
        for actor in self.actors.values():
            actor.eval()

    def _connect_datamodule(self, datamodule: "DataModule"):
        '''must be called before connect_strategy, because total_steps unset ??????'''
        self.datamodule = datamodule

        for optim_group in self.strategy.optim_configs.values():
            optim_group.total_steps = datamodule.total_train_steps


    def _connect_strategy(self, strategy: "TrainStrategy"):
        assert parallel.is_initialized()
        assert isinstance(strategy, TrainStrategy)
        self.strategy = strategy
        self.strategy._connect_module(self)
        # self.zero_grad()


    def _save_module(self, checkpoint_config: "CheckpointConfig"):
        if self.global_step % checkpoint_config.weight_interval: return

        save_dir = os.path.join(checkpoint_config.save_dir, "models", f"global_step{self.global_step}")
        if dist.get_rank() == 0:
            os.makedirs(save_dir, exist_ok = True)
        dist.barrier()

        self.save_module(save_dir)
        
        dist.barrier()
        torch.cuda.synchronize()

        if dist.get_rank() == 0:
            print(f"Model Weight: {save_dir} is ready !!!")
    

    def save_module(self, save_dir: str, tag: str):
        '''
        save_dir is already be set different
        This Function can either be implemented manually, or be replaced by train_strategy'''
        return self.strategy.save_module(save_dir, tag)


    def config_module(self):
        '''
        Config/Setup model, optimizer, scheduler by different TrainStrategy
        This Function can either be implemented manually, or be replaced by train_strategy'''
        return self.strategy.config_module()
    
    def backward(self, loss: torch.Tensor, actor_of_the_loss: Actor = None):
        '''
        This Function should implements the backward process.
        It can either be implemented manually, or be replaced by train_strategy'''
        return self.strategy.backward(loss, actor_of_the_loss)

    def step(self):
        '''
        This Function should implements the optimizer step process.
        It can either be implemented manually, or be replaced by train_strategy'''
        return self.strategy.step()

    def zero_grad(self):        
        '''
        This Function should implements the optimizer/model zero_grad process.
        It can either be implemented manually, or be replaced by train_strategy'''
        return self.strategy.zero_grad()

    def micro_batch_forward_backward(self, batch) -> "dict[str, Accumulator]":
        '''You May implement this function, or implement `forward_backward` directly.'''
        return self.strategy.micro_batch_forward_backward(batch)

    def non_accum_logs_per_step(self) -> "dict[str, Accumulator]":
        '''
        This function returns a dict of variables for being visualized,
          (Only for those remain the same among different micro batches of a same global batch, 
           and different among different global batches, such as the value of learning rate)
        '''
        return self.strategy.non_accum_logs_per_step()

    def forward_backward(self, batches) -> "dict[str, Accumulator]":
        '''
        This Function defines a global step forward and backward process, aimed to gain the full grad of this batches. The optimizer step will be executed immediately after this function having been executed.

        [Note] batches is a global batch, a list of micro batch
        '''

        logs_dict = defaultdict(Accumulator)
        for batch in batches:
            mirco_logs_dict = self.micro_batch_forward_backward(batch)
            for k, v in mirco_logs_dict.items():
                if not isinstance(v, Accumulator): v = Accumulator(v, 1)
                logs_dict[k] += v
        return {k: v.get_value() for k, v in logs_dict.items()}


    def micro_batch_validate_forward(self, batch) -> "dict[str, Accumulator]":
        raise NotImplementedError
    
    def save_ckpt(self, save_dir, tag):
        return self.strategy.save_ckpt(save_dir, tag)
    def load_ckpt(self, saved_dir, tag):
        return self.strategy.load_ckpt(saved_dir, tag)

    def sub_dir_to_save(self): return "Module_ckpts"



    def validate(self, val_data_iter):
        '''You may implement validating process in this function  and return a result dicts for visualization'''

        logs_dict = defaultdict(Accumulator)
        
        step_bar = tqdm(
                range(len(self.eval_dataloader)),
                desc = f"Eval stage of steps {self.global_step}",
                disable = parallel.get_rank() != 0
            )
        for batches in val_data_iter:
            for batch in batches:
                mirco_logs_dict = self.validate_forward(batch)
                for k, v in mirco_logs_dict.items():
                    logs_dict[k] += v
            step_bar.update()
        logs_dict = {k: v.get_value() for k, v in logs_dict.items()}
        step_bar.set_postfix(logs_dict)
        return logs_dict

    def _validate(self, datamodule: "DataModule"):
        logs_dict = dict()
        if datamodule.is_validating_step:
            self.eval()
            datamodule.eval()
            datamodule._setup_val_data_iter()
            with torch.no_grad():
                logs_dict = self.validate(datamodule.val_data_iter)
            self.train()
            datamodule.train()
        return logs_dict

    def _update(self, batches):
        '''logic that forward/backward a whole batch then update parameters'''
        train_logs_dict = self.forward_backward(batches)
        self.step()
        self.zero_grad()
        train_logs_dict.update(self.non_accum_logs_per_step())
        
        return train_logs_dict
        





