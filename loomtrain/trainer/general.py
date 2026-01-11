from typing import Union, Literal, List
from dataclasses import dataclass
import torch
import torch.distributed as dist
import torch.optim as opt
import torch.utils.data as tud
import torch.optim.lr_scheduler as tol
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
# from einops import rearrange,reduce,repeat
# from einops.layers.torch import Rearrange,Reduce
from transformers import PreTrainedTokenizer
from datasets import load_from_disk, Dataset
from loomtrain import cached, basename, path_join
from loomtrain.utils.init_hf import init_model, init_tokenizer
from loomtrain.utils.distributed.torch import all_reduce
from loomtrain.utils.wandb import WandbConfig
from loomtrain.utils.tensorboard import TensorboardConfig
from loomtrain.trainer.base import Trainer, TrainerConfig
from loomtrain.modeling.gpt import GPT, GPTCELoss
from loomtrain.modeling.rm import RM
from loomtrain.strategy import DeepspeedStrategy
from loomtrain.dataset import *
from loomtrain.trainer import *
from loomtrain.utils.distributed_sampler import (
    DistributedSampler, DistributedBucketSampler
)


@dataclass
class LoomTrainerConfig(TrainerConfig):
    mode: Literal["sft", "simpo", "simpo_disrm"] = "sft"
    extra_config: dict = dict()
    def __post_init__(self):
        for k, v in self.extra_config:
            setattr(self, k, v)
        extra_keys = set(self.extra_config.keys()) 
        required_keys = set()
        if self.mode == "sft":
            required_keys |= set([
                "prompt_key", "response_key"
            ])
        if self.mode == "simpo":
            required_keys |= set([
                "prompt_key", "chosen_key", "rejected_key", "num_rejects",
                "beta", "label_smoothing", "ipo", "memory_save", "nll_loss_weight"
            ])
        assert extra_keys.issuperset(required_keys)


    beta: float = 0.01
    label_smoothing: float = 0.
    ipo: bool = False
    memory_save: bool = False
    nll_loss_weight: float = 0.


class LoomTrainer:
    '''
    strategy will wrap the model and setup dataloader here
    '''
    def __init__(
      self,
      model_path: str,
      dataset_paths: List[str],
      dataset_cache_dir: str,
      max_data_length: int,
      train_dataset_counts: List[int],
      train_dataset_ratios: List[float],
      eval_dataset_counts: List[int],
      strategy: DeepspeedStrategy,
      config: LoomTrainerConfig,
      
      eval_dataset_paths: List[str] = None,
      tokenizer_path: str = None,
      wandb_config: WandbConfig = None,
      tensorboard_config: TensorboardConfig = None     
    ):
        self.model_path = model_path
        self.tokenzier_path = tokenizer_path
        self.dataset_paths = dataset_paths
        self.dataset_cache_dir = dataset_cache_dir
        self.max_data_length = max_data_length
        self.eval_dataset_paths = eval_dataset_paths
        self.strategy = strategy
        self.config = config

        self.wandb_config = wandb_config
        self.tensorboard_config = tensorboard_config
        
        strategy.init_distributed()
        
        model, tokenizer = self.init_hf(model_path, tokenizer_path)
        self.model = self.prepare_model(model)
        self.tokenizer = tokenizer

        self.train_dataset, self.eval_dataset = self.prepare_dataset(
            dataset_paths, eval_dataset_paths,
            train_dataset_counts, train_dataset_ratios,
            eval_dataset_counts
        )

        self.optimizer = self.prepare_optimizer()

        if config.mode == "sft":
            trainer_class = SFTTrainer
        elif config.mode == "simpo":
            trainer_class = SimPOTrainer
        elif config.mode == "simpo_disrm":
            trainer_class = SimPOBradleyTerryRMTrainer

        self.trainer = trainer_class(
            model = self.model,
            train_dataset = self.train_dataset,
            eval_dataset = self.eval_dataset,
            optimizer = self.optimizer,
            strategy = self.strategy,
            config = self.config,
            tokenizer = self.tokenizer,
            wandb_config = self.wandb_config,
            tensorboard_config = self.tensorboard_config
        )

    def init_hf(self, model_path, tokenizer_path):

        model = init_model(model_path)
        tokenizer = init_model(model_path if tokenizer_path is None else tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        return model, tokenizer

    def prepare_model(self, model):
        if self.config.mode == "simpo_disrm":
            return self.strategy.prepare_if_lora(RM(model))
        return self.strategy.prepare_if_lora(GPT(model))

    def prepare_dataset(self, dataset_paths: List[str], eval_dataset_paths: List[str],
                        train_dataset_counts: List[int], train_dataset_ratios: List[float], eval_dataset_counts: List[int]):
        datasets_dicts = [
            load_from_disk(pth) for pth in dataset_paths
        ]

        if self.config.mode == "sft":
            cache_fn = self._cache_PreferenceDataset 
            blended_class = BlendedSFTDataset
        elif self.config.mode in ("simpo", "simpo_disrm"):
            cache_fn = self._cache_PreferenceDataset
            blended_class = BlendedPreferenceDataset


        if eval_dataset_paths is not None:
            eval_datasets_dicts = [
                load_from_disk(pth) for pth in eval_dataset_paths
            ]

        # TODO: distributed preprocess
        if dist.get_rank() == 0:
            [cache_fn(datasets["train"], dataset_path, "train") \
             for datasets, dataset_path in zip(datasets_dicts, dataset_paths)]
            
            if eval_dataset_paths is None:
                [cache_fn(datasets["eval"], dataset_path, "eval") \
                for datasets, dataset_path in zip(datasets_dicts, dataset_paths)]
            else:
                [cache_fn(datasets["train"], dataset_path, "eval") \
                for datasets, dataset_path in zip(eval_datasets_dicts, eval_dataset_paths)]

        dist.barrier()

        train_datasets = [cache_fn(datasets["train"], dataset_path, "train") \
             for datasets, dataset_path in zip(datasets_dicts, dataset_paths)]
        
        if eval_dataset_paths is None:
            eval_datasets = [cache_fn(datasets["eval"], dataset_path, "eval") \
            for datasets, dataset_path in zip(datasets_dicts, dataset_paths)]
        else:
            eval_datasets = [cache_fn(datasets["train"], dataset_path, "eval") \
            for datasets, dataset_path in zip(eval_datasets_dicts, eval_dataset_paths)]
        
        for d in train_datasets + eval_datasets: 
            d.ring_attn_size = self.strategy.ring_attn_config.ring_attn_size
        

        train_dataset = blended_class(train_datasets, 
                                      sample_counts = train_dataset_counts,
                                      sample_ratios = train_dataset_ratios)
        
        eval_dataset = blended_class(eval_datasets,
                                     sample_counts = eval_dataset_counts)
        

        return train_dataset, eval_dataset

    
    def _cache_SFTDataset(self, dataset, dataset_path: str, type: Literal["train", "eval"]):
        return cached(
            SFTDataset,
            kwargs = dict(
                dataset = dataset,
                prompt_key = self.config.prompt_key, 
                response_key = self.config.response_key,
                tokenizer = self.tokenizer, max_length = self.max_data_length,
                ring_attn_size = self.strategy.ring_attn_config.ring_attn_size
            ),
            cache_dir = self.dataset_cache_dir,
            cache_name = f"{basename(self.model_path)}={basename(dataset_path)}=max-length{self.max_data_length}-{type}"
        ) 

    def _cache_PreferenceDataset(self, dataset, dataset_path: str, type: Literal["train", "eval"]):
        return cached(
            PreferenceDataset,
            kwargs = dict(
                dataset = Dataset.from_dict(dataset['eval'][:150]),
                prompt_key = self.config.prompt_key,
                chosen_key = self.config.chosen_key,
                rejected_key = self.config.rejected_key,
                num_rejects = self.config.num_rejects,
                sample_rejects = lambda x, n: x[:n],
                tokenizer = self.tokenizer, max_length = self.max_data_length,
                ring_attn_size = self.strategy.ring_attn_config.ring_attn_size,
                num_processors = 1,
            ),
            cache_dir = self.dataset_cache_dir,
            cache_name = f"{basename(self.model_path)}={basename(dataset_path)}=max-length{self.max_data_length}-{type}"
        )

    def prepare_optimizer(self):
        return self.strategy.setup_optimizer(
            self.model,
            lr = self.config.learing_rate,
            betas = self.config.adam_betas,
            weight_decay = self.config.adam_weight_decay
        )


    def fit(self, load_ckpt: bool = True):
        self.trainer.fit(load_ckpt)
