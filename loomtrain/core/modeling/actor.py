import torch
from torch import nn

from typing import Literal, Optional
from functools import partial
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, 
    SequenceClassifierOutputWithPast
)

from loomtrain.core.metas import AttrDict
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.modeling.customs.rm_modeling import train_forwards

from loomtrain.core.utils.init_hf import init_model, init_tokenizer


def get_actor_cls(actor_type: Literal["causal", "classifier"] = "causal", 
                  collate_type: Literal["packing", "padding"] = "packing") -> "type[Actor]":
    if (actor_type, collate_type) == ("causal", "packing"):
        return PackingGPT
    
    elif (actor_type, collate_type) == ("classifier", "packing"):
        return PackingClassifier


def init_actor(model_path, 
               model_type: Literal['causal', 'classifier'] = "causal", 
               collate_type: Literal["packing", "padding"] = "packing") -> "Actor":
    actor = get_actor_cls(model_type, collate_type)(init_model(model_path, model_type = model_type))
    actor.init_args = AttrDict(model_path = model_path, 
                               model_type = model_type, 
                               collate_type = collate_type)
    return actor

class Actor(nn.Module):
    '''
    A module with a single optimizer and learning rate scheduler
    '''
    def __init__(self, model: "nn.Module", trainable: bool = True):
        super().__init__()
        self.model = model
        self.trainable = trainable
        self._optim_objects_ = {} 

    @property
    def init_args(self):
        if not hasattr(self, "_init_args_"):
            self._init_args_ = AttrDict(model_path = None, model_type = None, collate_type = None)
        return self._init_args_
    @init_args.setter
    def init_args(self, args: "AttrDict"):
        self._init_args_ = args

    @property
    def optimizer(self) -> "torch.optim.Optimizer":
        return self._optim_objects_.get("optimizer", None)


    def set_optimizer(self, optim: "torch.optim.Optimizer"):
        assert self.trainable, "Cannot set optimizer for a non-trainable Actor"
        self.optimizer = optim
    def set_scheduler(self, sched: "torch.optim.lr_scheduler.LRScheduler"):
        assert self.trainable, "Cannot set scheduler for a non-trainable Actor"
        self.scheduler = sched

    @optimizer.setter
    def optimizer(self, optim: "torch.optim.Optimizer"):    
        self._optim_objects_["optimizer"] = optim

    @property
    def scheduler(self) -> "torch.optim.lr_scheduler.LRScheduler":
        return self._optim_objects_.get("scheduler", None)

    @scheduler.setter
    def scheduler(self, sched: "torch.optim.lr_scheduler.LRScheduler"):    
        self._optim_objects_["scheduler"] = sched

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class PackingGPT(Actor):
    def forward(self, 
                sequences: "torch.LongTensor",
                seq_lens: "Optional[list[int]]" = None,
                attention_mask: "Optional[torch.BoolTensor]" = None):
        inputs = parallel.prepare_cp_input(packed_sequences = sequences, 
        seq_lens = seq_lens, attention_mask = attention_mask)
        output = self.model(*inputs)

        output["logits"] = output["logits"].to(torch.float32)

        return output

class PackingClassifier(Actor):
    def __init__(self, model: "nn.Module"):
        model._org_forward = model.forward
        
        train_forward = getattr(train_forwards,  model.config.architectures[0],
                                self.train_forward)

        model.forward = partial(train_forward, model = model)

        super().__init__(model)



    def train_forward(self,
                      input_ids: torch.LongTensor = None, #packed_sequences
                      attention_mask: Optional[torch.BoolTensor] = None,
                      position_ids: Optional[torch.LongTensor]= None,                     
                      model: "nn.Module" = None,
                      ** kwargs,
                      ):

        output: BaseModelOutputWithPast = model.model(input_ids = input_ids,
                                                      attention_mask = attention_mask,
                                                      position_ids = position_ids)
        hidden_states = output.last_hidden_state
        logits = model.score(hidden_states)

        output = SequenceClassifierOutputWithPast(
            logits = logits
        )        

        return output



    def forward(self,
                sequences: "torch.LongTensor", #packed_sequences
                seq_lens: "Optional[list[int]]" = None,
                attention_mask: "Optional[torch.BoolTensor]" = None,
                ):

        inputs = parallel.prepare_cp_input(packed_sequences = sequences, 
        seq_lens = seq_lens, attention_mask = attention_mask)
        output = self.model(*inputs)

        output["logits"] = output["logits"].to(torch.float32)

        return output


