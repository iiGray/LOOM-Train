from __future__ import annotations
import os
import torch
import torch.distributed as dist
from copy import deepcopy
from peft import get_peft_model_state_dict, get_peft_model, PeftModel
# from loomtrain.strategy.base import Strategy

from loomtrain.core.strategy import TrainStrategy
from loomtrain.core.parallel import parallel_state as parallel

from loomtrain.core.modeling.actor import PackingGPT, PackingClassifier
from loomtrain.core.utils.init_hf import init_model
from loomtrain.core.utils.common import IO



class OrdinaryStrategy(TrainStrategy):
    '''
    This Strategy has no optimizer/grad offload, which means it may take much memory
    '''
    def __init__(
            self,
            parallel_config: "parallel.ParallelConfig",
            full_determinism: bool = False,
            seed: int = 42,
    ):
        super().__init__(
            parallel_config = parallel_config,
            full_determinism = full_determinism,
            seed = seed
        )
        

    def init_distributed(self):
        parallel.init_distributed(backend = 'nccl')



    def backward(self, actor, loss):
        loss.backward()
    
    def step(self):
        for group in self.opt_groups.values():
            group.actor.optimizer.step()
            group.actor.scheduler.step()

    def zero_grad(self):
        for group in self.opt_groups.values():
            group.actor.model.zero_grad()

    def save_ckpt(self, save_dir: str, tag: str):
        for name, group in self.opt_groups.items():
            group.model.save_checkpoint(save_dir = os.path.join(save_dir, name), 
                                        tag = tag,
                                        client_state = dict(),
                                        save_latest = True)


    def load_ckpt(self, saved_dir: str, tag: str):
        for name, group in self.opt_groups.items():
            group.model.load_checkpoint(
                load_dir = saved_dir,
                tag = tag,
                load_module_strict = True,
                load_optimizer_states = True,
                load_lr_scheduler_states = True,
                load_module_only = False
            )
    
    def save_module(self, save_dir: str):        
        for name, group in self.opt_groups.items():
            gathered_state_dict = dict()
            actor = group.actor
            model_to_save = actor.model

            csave_dir = os.path.join(save_dir, name)

            if dist.get_rank() == 0:
                state_dict = model_to_save.state_dict()
                for k, v in model_to_save.named_buffers():
                    if k in state_dict:
                        gathered_state_dict[k] = v.data.cpu()
                
                state_dict_keys = set(state_dict.keys())
                gathered_state_dict_keys = set(gathered_state_dict.keys())

                assert state_dict_keys.issubset(gathered_state_dict), \
                f"Mismatch keys: {gathered_state_dict_keys.symmetric_difference(state_dict_keys)}"

                if isinstance(model_to_save, PeftModel):
                    if isinstance(actor, PackingGPT):
                        model_to_save = deepcopy(model_to_save)
                    elif isinstance(actor, PackingClassifier):
                        base_model = init_model(model_to_save.base_model.model._load_path,
                                                model_type = "classifier")
                        cloned = get_peft_model(base_model, self.lora_config)
                        cloned.load_state_dict(model_to_save.state_dict(), strict=True)
                        model_to_save = cloned


                    if self.lora_config.save_merged:
                        model_to_save = model_to_save.merge_and_unload()
                        model_to_save.save_pretrained(csave_dir, )
                    else:
                        adapter_csave_dir = csave_dir
                        model_to_save.save_pretrained(adapter_csave_dir, )
                        if self.config.zero_stage == 3:
                            torch.save(
                                get_peft_model_state_dict(model_to_save, gathered_state_dict),
                                os.path.join(csave_dir, "adapter_model.bin"),
                            )
                        
                else:
                    model_to_save.save_pretrained(
                        save_directory = csave_dir, state_dict = gathered_state_dict, )

                model_to_save.config.to_json_file(os.path.join(csave_dir, "config.json"))

                group.tokenizer.save_pretrained(csave_dir)


                train_from_model_path = model_to_save.config._name_or_path
                if os.path.exists(train_from_model_path):
                    for file_name in IO.read_path(train_from_model_path, concat_root = False):
                        if file_name.endswith(".py"):
                            IO.copy(os.path.join(train_from_model_path, file_name),
                                    os.path.join(csave_dir, file_name))

            dist.barrier()
            torch.cuda.synchronize()