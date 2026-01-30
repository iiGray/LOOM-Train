import os
import torch
from typing import TYPE_CHECKING, Iterable, Iterator, TypeVar
from torch import nn
import torch.distributed as dist
from transformers import PreTrainedTokenizer
from collections import defaultdict
from tqdm import tqdm
from loomtrain.core.metas import AttrDict, LazyInitializeMeta
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.strategy import TrainStrategy, OptimConfig
from loomtrain.core.state import CheckpointConfig, CheckpointMixin
from loomtrain.core.modeling.actor import Actor
from loomtrain.core.arguments import args
if TYPE_CHECKING:
    from loomtrain.core.datamodule import DataModule
    from loomtrain.core.data.dataloader.iter import MicroBatch
from loomtrain.core.visualization import LogDict, AccumLogDict, Accum
from loomtrain.utils.lora import LoRAConfig, get_peft_model
from dataclasses import dataclass
import loomtrain as lt

T = TypeVar("T")
 
class CountableIterator(Iterator[T]):
    def __init__(self, iterable: "Iterable[T]"):
        self._iterator = iter(iterable)
        self._count = 0
        self._finished = False 
    def __iter__(self):
        return self

    def __next__(self):
        if self._finished:
            raise StopIteration
        try:
            item = next(self._iterator)
            self._count += 1
            return item
        except StopIteration:
            self._finished = True
            raise 
    @property
    def length(self):
        assert self._finished, "Iteration is not complete; the property `length` is unavailable."
        return self._count


class Module(CheckpointMixin, metaclass = LazyInitializeMeta):
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

    def __init__(self, optim_configs: "OptimConfig | dict[str, OptimConfig]", stat_batch_as_unit: "bool" = False, *args, **kwargs):        
        assert parallel.is_initialized() or (not parallel.is_distributed_allowed), "One must init `Trainer` before init `Module`"
        super().__init__(*args, **kwargs)
        if isinstance(optim_configs, OptimConfig):
            # module means TrainStrategy.module
            optim_configs = {"module": optim_configs} 
        self.optim_configs = optim_configs
        self._check_optim_configs()

        self.stat_batch_as_unit = stat_batch_as_unit

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


    def _save_module(self, checkpoint_config: "CheckpointConfig", finished: "bool" = False):
        if (self.global_step % checkpoint_config.weight_interval) and (not finished): return

        save_dir = os.path.join(checkpoint_config.save_dir, f"global_step{self.global_step}")
        if dist.get_rank() == 0:
            os.makedirs(save_dir, exist_ok = True)
        dist.barrier()

        self.save_module(save_dir, "model_weights")
        
        dist.barrier()
        torch.cuda.synchronize()

        if dist.get_rank() == 0:
            print(f"Model Weight: {save_dir} is ready !!!")
    

    def save_module(self, save_dir: "str", tag: "str"):
        '''
        save_dir is already be set different
        This Function can either be implemented manually, or be replaced by train_strategy'''
        return self.strategy.save_module(save_dir, tag)


    def config_module(self):
        '''
        Config/Setup model, optimizer, scheduler by different TrainStrategy
        This Function can either be implemented manually, or be replaced by train_strategy
        '''
        return self.strategy.config_module()
    
    def backward(self, loss: "torch.Tensor", actor_of_the_loss: "Actor" = None):
        '''
        This Function should implements the backward process.
        It can either be implemented manually, or be replaced by train_strategy
        '''
        return self.strategy.backward(loss, actor_of_the_loss)

    def step(self, micro_steps: "int" = 1):
        '''
        This Function should implements the optimizer step process.
        It can either be implemented manually, or be replaced by train_strategy

        micro_steps: number of micro-batches consumed from the last step on.
        '''
        return self.strategy.step(micro_steps)

    def zero_grad(self):        
        '''
        This Function should implements the optimizer/model zero_grad process.
        It can either be implemented manually, or be replaced by train_strategy
        '''
        return self.strategy.zero_grad()

    def micro_batch_forward_backward(self, batch) -> "AccumLogDict[str, Accum]":
        '''You May implement this function, or implement `forward_backward` directly.'''
        return self.strategy.micro_batch_forward_backward(batch)

    def on_after_micro_batch_forward_backward(self):
        return self.strategy.on_after_micro_batch_forward_backward()

    def non_accum_logs_per_step(self) -> "LogDict[str, Accum]":
        '''
        This function returns a dict of variables for being visualized,
          (Only for those remain the same among different micro batches of a same global batch, 
           and different among different global batches, such as the value of learning rate)
        '''
        return self.strategy.non_accum_logs_per_step()

    def forward_backward(self, batches: "CountableIterator[MicroBatch]") -> "AccumLogDict[str, Accum]":
        '''
        This Function defines a global step forward and backward process, aimed to gain the full grad of this batches. The optimizer step will be executed immediately after this function having been executed.

        [Note] batches is a global batch, a list of micro batch
        '''

        logs_dict = defaultdict(Accum)
        for batch in tqdm(batches, desc = f"Micro Batches of Global Step {self.global_step}",
                          total = self.strategy.data_config.grad_accum,
                          position = 1, 
                          disable = parallel.get_rank() != 0 or (not args().enable_micro_bar)):
            mirco_logs_dict = self.micro_batch_forward_backward(batch.value)
            self.on_after_micro_batch_forward_backward()
            for k, v in mirco_logs_dict.items():
                v.set_total(1 if self.stat_batch_as_unit else batch.num_samples)
                try:
                    if k not in logs_dict: logs_dict[k] = v
                    else: logs_dict[k] += v
                except Exception as e:
                    print(f"Exception Occurs During handling logs_dict {k} !")
                    raise e
        return AccumLogDict( ** {k: v for k, v in logs_dict.items()})


    def batch_validate_forward(self, batch) -> "AccumLogDict[str, Accum]":
        raise NotImplementedError

    def save_ckpt(self, save_dir, tag):
        return self.strategy.save_ckpt(save_dir, tag)
    def load_ckpt(self, saved_dir, tag):
        return self.strategy.load_ckpt(saved_dir, tag)

    def sub_dir_to_save(self): return "model_ckpts"
       
    def validate(self, val_data_iter) -> "AccumLogDict[str, Accum]":
        '''You may implement validating process in this function  and return a result dicts for visualization'''

        logs_dict = defaultdict(Accum)
        
        step_bar = tqdm(
                range(len(val_data_iter)),
                desc = f"Eval Stage of Global Step {self.global_step}",
                disable = parallel.get_rank() != 0
            )
        for batch in val_data_iter:
            num_samples = batch.num_samples  # dataiter will wrap batch into MicroBatch
            batch = batch.value # dataiter will wrap batch into MicroBatch
            batch = self.datamodule.to_current_device(batch)
            mirco_logs_dict = self.batch_validate_forward(batch)
            for k, v in mirco_logs_dict.items():
                v.set_total(1 if self.stat_batch_as_unit else num_samples)
                if k in logs_dict: logs_dict[k] += v
                else: logs_dict[k] = v
            step_bar.update()
        logs_dict = AccumLogDict(** {k: v for k, v in logs_dict.items()})
        step_bar.set_postfix({k: v.get_value() for k, v in logs_dict.items()})
        return logs_dict

    def _validate(self, datamodule: "DataModule", finished: "bool") -> "AccumLogDict[str, Accum]":
        logs_dict = dict()
        if datamodule.is_validating_step or finished:
            self.eval()
            datamodule.eval()
            with torch.no_grad():
                logs_dict = self.validate(datamodule.val_data_iter)
            datamodule.reset_val_data_iter()
            self.train()
            datamodule.train()
        return logs_dict

    def _update(self, batches: "Iterable[MicroBatch]") -> "dict[str, Accum | object]":
        '''logic that forward/backward a whole batch then update parameters'''
        batches = CountableIterator(batches)
        train_logs_dict = dict()
        train_logs_dict.update(self.forward_backward(batches))
        self.step(batches.length)
        self.zero_grad()
        train_logs_dict.update(self.non_accum_logs_per_step())
        
        return train_logs_dict
        





