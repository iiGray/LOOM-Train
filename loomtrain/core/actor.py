from torch import nn
from typing import Literal
from loomtrain.utils.init_hf import init_model, init_tokenizer
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.modeling.actors import *
from loomtrain.core.modeling.loss import *

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class LoomOptDict(AttrDict):
    def __init__(
        self,
        model_name: str = None,
        model_type :Literal["causal", "classifier"] = "causal",
        collate_type: Literal["packing", "padding"] = "packing", #TODO: padding
        loss_type: Literal["sft", "simpo"] = "sft",
        tokenizer_name = None,
        lr   : "float" = 1e-5,
        min_lr: "float" = None,
        betas: "tuple" = (0.9, 0.95),
        L2_weight_decay: float = 0.0,
        lr_type: Literal["linear",
                       "cosine",
                       "cosine_with_restarts",
                       "polynomial",
                       "constant",
                       "constant_with_warmup",
                       "inverse_sqrt",
                       "reduce_lr_on_plateau",
                       "cosine_with_min_lr",
                       "warmup_stable_decay"] = "cosine_with_min_lr",
        warmup_ratios: "int" = 0.03,
        num_warmup_steps: "int" = None,
    ):


        super().__init__(
            model_name = model_name,
            model_type = model_type,
            collate_type = collate_type,
            tokenizer_name = tokenizer_name,
            lr = lr,
            min_lr = min_lr,
            lr_type = lr_type,
            warmup_ratios = warmup_ratios,
            num_warmup_steps = num_warmup_steps,
            loss_type = loss_type,
            total_steps = None, # This proprety will be set after module.connect_datamodule(datamodule)
        )

        self.model_name = model_name
        self.model_type = model_type
        self.collate_type = collate_type
        if tokenizer_name is None:
            self.tokenizer_name = model_name
        self.lr = lr
        self.min_lr = min_lr
        self.lr_type = lr_type
        self.betas = betas
        self.L2_weight_decay = L2_weight_decay
        self.warmup_ratios = warmup_ratios
        self._num_warmup_steps = num_warmup_steps
        self.loss_type = loss_type
        self.total_steps = None

    @property
    def num_warmup_steps(self):
        if not self._num_warmup_steps:
            assert self.total_steps, "LoomModule should connect LoomDataModule first."
            self._num_warmup_steps = round(self.total_steps * self.warmup_ratios)

        return self._num_warmup_steps


class LoomActorGroup(AttrDict):
    def __init__(
        self,
        model: "nn.Module",
        tokenizer,
        optimizer,
        scheduler,
        actor_type: Literal["causal", "classifier"] = "causal",
        collate_type: Literal["packing", "padding"] = "packing", #TODO: padding
        loss_type: Literal["sft", "simpo"] = "sft",
        actor: "Actor" = None,
        loss_fn = None):

        super().__init__(
            model = model, 
            tokenizer = tokenizer,
            optimizer = optimizer, 
            scheduler = scheduler, 
            actor_type = actor_type,
            loss_type = loss_type,
            actor = actor,
            loss_fn = loss_fn,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.actor_type = actor_type
        self.collate_type = collate_type
        self.loss_type = loss_type
    
    def build_actor(self):
        self.actor = get_actor_cls(self.actor_type, self.collate_type)(self.model)
        self.loss_fn = get_loss_cls(self.loss_type)()
