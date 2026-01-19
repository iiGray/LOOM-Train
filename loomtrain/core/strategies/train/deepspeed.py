from __future__ import annotations
import os
from copy import deepcopy
from datetime import timedelta
from typing import Union, Literal, Callable, TYPE_CHECKING
from dataclasses import dataclass
import torch, deepspeed
from transformers.trainer import get_scheduler
import torch.distributed as dist
from torch import nn


from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import get_peft_model_state_dict, get_peft_model, PeftModel
# from loomtrain.strategy.base import Strategy
from loomtrain.core.utils import basename, dirname, save_json
from loomtrain.core.strategy import TrainStrategy
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.utils.init_hf import init_model, init_tokenizer
from loomtrain.core.modeling.actor import Actor, PackingGPT, PackingRM 
from loomtrain.utils.common import IO
from loomtrain.core.arguments import args
from loomtrain.core.strategy import DataConfig, OptimConfig

@dataclass
class DeepspeedConfig:
    zero_stage: Literal[2, 3] = 2,
    enable_bf16     : bool = True
    offload         : bool = False
    adam_offload    : bool = False
    ref_offload     : bool = False # reference model offload
    grad_clip       : float = 1. 
    zpg             : int = 1
    grad_accum_dtype: Literal["fp16", "bf16", "fp32"] = None
    overlap_comm    : bool = False
    load_universal   : bool = False
    torch_compile   : bool = False #useless ?


class DeepspeedStrategy(TrainStrategy):
    def __init__(self, 
                 parallel_config: "parallel.ParallelConfig" = None,
                 data_config: "DataConfig" = None,
                 config: "DeepspeedConfig" = None,):
        super().__init__(
            parallel_config = parallel_config, 
            data_config = data_config
        )
        if config is None:
            config = DeepspeedConfig(
                zero_stage = args().zero_stage,
                enable_bf16 = args().enable_bf16,
                offload = args().offload,
                adam_offload = args().adam_offload,
                ref_offload = args().ref_offload,
                grad_clip = args().grad_clip,
                zpg = args().zpg,
                grad_accum_dtype = args().grad_accum_dtype,
                overlap_comm = args().overlap_comm,
                load_universal = args().load_universal,
            )
        self.config = config

    def init_distributed(self):
        deepspeed.init_distributed(timeout = timedelta(minutes = args().deepspeed_init_timeout))


    def config_module(self):
        '''
        deepspeed default config only one model, one optimizer and one scheduler,
        for more flexible use, one may implement `setup_module` directly by inheriting LoomModule and override it.
        '''

        for actor_name, actor_optim_cfg in self.split_submodules_by_actors().items():
            AdamOptimizer = DeepSpeedCPUAdam if self.config.adam_offload else FusedAdam

            optim_params = []
            for name_path, opt_cfg in actor_optim_cfg.items():
                optim_params += optimizer_grouped_parameters(
                    self.get_submodule(name_path),
                    lr = opt_cfg.lr,
                    betas = opt_cfg.betas,
                    weight_decay = opt_cfg.L2_weight_decay
                )

            optimizer = AdamOptimizer(optim_params)
            scheduler = get_scheduler(
                name = opt_cfg.lr_type,
                optimizer = optimizer,
                num_warmup_steps = opt_cfg.num_warmup_steps,
                num_training_steps = opt_cfg.total_steps,
                scheduler_specific_kwargs = dict(min_lr = opt_cfg.lr * 0.05)
            )
            
            actor, optimizer, scheduler = self._prepare_train(
                self.get_submodule(actor_name), optimizer, scheduler
            )

            actor.set_optimizer(optimizer)
            actor.set_scheduler(scheduler)

            self.module.set_actor(actor_name, actor)


    def backward(self, loss: "torch.Tensor", actor_of_the_loss: "Actor" = None):
        actor_of_the_loss.model.backward(loss)
    
    def step(self):
        for actor in self.opt_groups.values():
            actor.model.step()

    def zero_grad(self):
        for actor in self.opt_groups.values():
            engine = actor.model
            if engine.bfloat16_enabled():
                # TODO: Temporary until bf16_optimizer and zero_optimizer are integrated
                if engine.zero_optimization() and hasattr(engine.optimizer, "zero_grad"):
                    engine.optimizer.zero_grad()
                else:
                    pass
            elif engine.zero_optimization() or engine.fp16_enabled() or engine.amp_enabled():
                engine.optimizer.zero_grad()
            else:
                engine.zero_grad()

    def _prepare_train(self, model: "Actor", optimizer, scheduler):
        engine, optimizer, _, scheduler = deepspeed.initialize(
            model = model.model,
            optimizer = optimizer,
            lr_scheduler = scheduler,
            config = deepspeed_train_config(
                        offload = self.config.offload,
                        adam_offload = self.config.adam_offload,
                        stage = self.config.zero_stage,
                        enable_bf16 = self.config.enable_bf16,
                        # train_batch_size = self.config.train_batch_size,
                        gradient_accumulation_steps = self.data_config.grad_accum,
                        train_micro_batch_size_per_gpu = self.data_config.micro_batch_size,
                        grad_clip = self.config.grad_clip,
                        zpg = self.config.zpg,
                        grad_accum_dtype = self.config.grad_accum_dtype,
                        overlap_comm = self.config.overlap_comm,
                        load_univeral = self.config.load_universal
                    ),
            args = dict(local_rank = int(os.environ.get("LOCAL_RANK", "-1"))),
            dist_init_required = True,
        ) 

        if self.config.torch_compile: engine.compile()
        
        model.model = engine

        return model, optimizer, scheduler
    

    def save_ckpt(self, save_dir: str, tag: str):
        for name, actor in self.opt_groups.items():
            actor.model.save_checkpoint(save_dir = os.path.join(save_dir, name), 
                                        tag = tag,
                                        client_state = dict(),
                                        save_latest = True)


    def load_ckpt(self, saved_dir: str, tag: str):
        for name, group in self.opt_groups.items():
            assert isinstance(group.model, deepspeed.DeepSpeedEngine)
            group.model.load_checkpoint(
                load_dir = os.path.join(saved_dir, name),
                tag = tag,
                load_module_strict = True,
                load_optimizer_states = True,
                load_lr_scheduler_states = True,
                load_module_only = False
            )
    

    # def restore_ckpt(self,
    #                  states: dict,
    #                  train_dataloader: tud.DataLoader,
    #                  config: TrainerConfig):
    #     consumed_samples = states['consumed_samples']
    #     total_tokens = states["total_tokens"] / self.ring_groups * (10**9)
    #     loss_tokens = states["loss_tokens"] / self.ring_groups * (10**9)
    #     update_steps_per_epoch = config.update_steps_per_epoch(train_dataloader)

    #     step = consumed_samples // config.batch_size * self.accumulated_gradient + 1
    #     start_epoch = consumed_samples // config.batch_size // update_steps_per_epoch
    #     consumed_samples %= (update_steps_per_epoch * config.batch_size)

    #     return step, update_steps_per_epoch, start_epoch, consumed_samples, total_tokens, loss_tokens
    
    def save_module(self, save_dir: str):
        
        for name, group in self.opt_groups.items():
            gathered_state_dict = dict()
            actor = group.actor
            model_to_save = actor.model
            assert isinstance(model_to_save, deepspeed.DeepSpeedEngine)
            
            if hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module


            csave_dir = os.path.join(dirname(save_dir), name, basename(save_dir))

            for k, v in model_to_save.named_parameters():

                with deepspeed.zero.GatheredParameters([v], enabled = \
                                                    hasattr(v, "ds_id") and v.ds_status == ZeroParamStatus.NOT_AVAILABLE):
                    if dist.get_rank() == 0:
                        gathered_state_dict[k] = v.data.cpu()
            
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
                    elif isinstance(actor, PackingRM):
                        base_model = init_model(model_to_save.base_model.model._load_path,
                                                model_type = "classifier")
                        cloned = get_peft_model(base_model, self.lora_config)
                        cloned.load_state_dict(model_to_save.state_dict(), strict=True)
                        model_to_save = cloned

                    #TODO Lora
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




def deepspeed_train_config(
    offload: bool,
    adam_offload: bool = True,
    stage: Literal[1, 2, 3] = 2,
    enable_bf16: bool = True,
    # train_batch_size: int = 1,
    gradient_accumulation_steps: int  = 1,
    train_micro_batch_size_per_gpu: int = 1,
    grad_clip: float = 1.0,
    zpg: int = 8,
    grad_accum_dtype: Literal["fp16", "bf16", "fp32"] = None,
    overlap_comm: bool = False,
    load_univeral: bool =False,
    activation_checkpoint: bool = True,
    cpu_checkpointing: bool = False,
):
    offload = bool(stage == 3)
    device = "cpu" if offload else "none"

    zero_opt_dict = dict(
        stage = stage,
        offload_param = dict(device = device),
        offload_optimizer = dict(
            device = "cpu" if adam_offload else "none",
            pin_memory = True
        ),
        sub_group_size = "auto",
        stage3_max_live_parameters = "auto", 
        stage3_max_reuse_distance = "auto",
        stage3_param_persistence_threshold = "auto",
        stage3_prefetch_bucket_size = "auto",
        reduce_bucket_size = "auto", 
        zero_hpz_partition_size = zpg, 
        zero_quantized_weights = False, 
        zero_quantized_gradients = False, 
    )

    if overlap_comm:
        zero_opt_dict.update(dict(
            overlap_comm = True,
            contiguous_gradients = True,
        ))
    if stage == 3:
        zero_opt_dict.update(dict(reduce_scatter = True))

    config_dict = dict(
        steps_per_print = 100,
        zero_optimization = zero_opt_dict,
        bf16 = dict(enabled = enable_bf16),
        # train_batch_size = train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        train_micro_batch_size_per_gpu = train_micro_batch_size_per_gpu,
        gradient_clipping =  grad_clip,
        prescale_gradients = False,
        wall_clock_breakdown = False,
        data_types = dict(grad_accum_dtype = grad_accum_dtype),
        checkpoint = dict(load_universal = load_univeral),
        activation_checkpointing = dict(
            partition_activations = False,
            contiguous_memory_optimization = True,
            cpu_checkpointing = cpu_checkpointing
        )
    )

    if not enable_bf16:
        config_dict.update(fp16 = dict(enabled = True))

    if activation_checkpoint:
        config_dict.update(dict(
            partition_activations = False,
            contiguous_memory_optimization = True,
            cpu_checkpointing = cpu_checkpointing
        ))
    elif cpu_checkpointing:
        print("Warning: ", f"`cpu_checkpointing` will not "
              "be set because you didn't set `activation_checkpoint = True`")
    
    return config_dict


def deepspeed_eval_config(
    offload: bool,
    stage: Literal[0, 1, 2, 3] = 0, # 默认不启用
    enable_bf16: bool = True,
    train_batch_size: int = 1,
    train_micro_batch_size_per_gpu: int = 1,
):
    zero_opt_dict = dict(
        stage = stage,
        stage3_max_live_parameters = "auto",
        stage3_max_reuse_distance = "auto",
        stage3_param_persistence_threshold = "auto",
        stage3_prefetch_bucket_size = "auto",
        offload_param = dict(
            device = "cpu" if offload else "none",
            pin_memory = True
        )
    )
    return dict(
        steps_per_print = 100,
        zero_optimization = zero_opt_dict,
        bf16 = dict(enabled = enable_bf16),
        fp16 = dict(enabled = not enable_bf16),
        train_batch_size = train_batch_size,
        train_micro_batch_size_per_gpu = train_micro_batch_size_per_gpu,
        gradient_clipping = 1.0,
        prescale_gradients = False,
        wall_clock_breakdown = False,

    )


def optimizer_grouped_parameters(
    model: deepspeed.DeepSpeedEngine,
    lr: float,
    betas: "tuple[float, float]",
    weight_decay: float,
    no_decay_name_list=["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "lr": lr,
            "betas": betas,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def zero3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]



def offload_deepspeed_states(model: deepspeed.DeepSpeedEngine, 
                             pin_memory: bool = True, 
                             non_blocking: bool = True):
    zero_stage = model.zero_optimization_stage()  # config['zero_optimization']['stage']
    adam_offload = model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"

    # state offloading not required when using Adam optimizer offloading
    if adam_offload:
        return

    if zero_stage != 3:
        raise NotImplementedError("Only Zero stage 3 is currently supported")

    # if zero_stage == 3 and not adam_offload:
    import torch
    from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum

    model.optimizer.offload_states(
        include=[
            OffloadStateTypeEnum.optim_states,
            OffloadStateTypeEnum.contiguous_grad_buffer,
            OffloadStateTypeEnum.hp_params,
        ],
        device=OffloadDeviceEnum.cpu,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
    )
    model.empty_partition_cache()
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    torch.cuda.synchronize()



def reload_deepspeed_states(model: deepspeed.DeepSpeedEngine, 
                            non_blocking=True):
    zero_stage = model.zero_optimization_stage()  # config['zero_optimization']['stage']
    adam_offload = model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"

    # state offloading not required when using Adam optimizer offloading
    if adam_offload:
        return

    if zero_stage != 3:
        raise NotImplementedError("Only Zero stage 3 is currently supported")

    # if zero_stage == 3 and not adam_offload:
    import torch

    model.reload_states(non_blocking=non_blocking)
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    torch.cuda.synchronize()

