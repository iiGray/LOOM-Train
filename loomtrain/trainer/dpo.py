import torch
import torch.distributed as dist
import torch.optim as opt
import torch.utils.data as tud
import torch.optim.lr_scheduler as tol
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from dataclasses import dataclass
# from einops import rearrange,reduce,repeat
# from einops.layers.torch import Rearrange,Reduce
from transformers import PreTrainedTokenizer
from loomtrain.utils.distributed.torch import all_reduce
from flash_attn.utils.distributed import all_gather
from loomtrain.utils.wandb import WandbConfig
from loomtrain.utils.tensorboard import TensorboardConfig
from loomtrain.trainer.base import Trainer, TrainerConfig
from loomtrain.modeling.gpt import GPT
from loomtrain.modeling.rl import DPOLoss, logps_from_logits
from loomtrain.strategy import DeepspeedStrategy
from loomtrain.dataset.sft import SFTDataset
from loomtrain.utils.distributed_sampler import DistributedSampler


@dataclass
class DPOTrainerConfig(TrainerConfig):
    beta: float = 0.01
    label_smoothing: float = 0.
    ipo: bool = False
    memory_save: bool = False
    nll_loss_weight: float = 0.


class DPOTrainer(Trainer):
    '''
    strategy will wrap the model and setup dataloader here
    '''
    def __init__(
      self,
      model: GPT,
      ref_model: GPT,
      train_dataset: SFTDataset,
      eval_dataset: SFTDataset,
      optimizer: opt.Optimizer,
      strategy: DeepspeedStrategy,
      config: DPOTrainerConfig,
      tokenizer: PreTrainedTokenizer = None,
      save_hf_ckpt: bool = False,
      disable_ds_ckpt: bool = False,
      wandb_config: WandbConfig = None,
      tensorboard_config: TensorboardConfig = None     
    ):
        super().__init__(
            model = model,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            optimizer = optimizer,
            strategy = strategy,
            config = config,
            wandb_config = wandb_config,
            tensorboard_config = tensorboard_config
        )
        
        self.config = config
        self.ref_model = strategy.prepare_eval(ref_model)

        self.tokenzier = tokenizer
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt

        self.loss_fn = DPOLoss(beta = config.beta,
                               label_smoothing = config.label_smoothing,
                               ipo = config.ipo)
        


    def fit(self, load_ckpt: bool = True):
        states = self.load_ckpt(load_ckpt = load_ckpt)
        consumed_samples = states['consumed_samples']
        update_steps_per_epoch = self.config.update_steps_per_epoch(len(self.train_dataloader.dataset))
        
        assert len(self.train_dataloader.dataset) and update_steps_per_epoch, \
            f"train_dataset_len: {len(self.train_dataloader.dataset)} < batch_size: {self.config.batch_size}"
        if self.config.eval_steps < 0:
            self.config.eval_steps = update_steps_per_epoch
        

        step = consumed_samples // self.config.batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // self.config.batch_size // update_steps_per_epoch
        consumed_samples %= (update_steps_per_epoch * self.config.batch_size)

        epoch_bar = tqdm(range(start_epoch, self.config.max_epochs), 
                         desc = "Train epoch", 
                         disable = dist.get_rank() != 0)


        # scheduler_steps_per_epoch = len(self.train_dataloader.dataset) // len(self.train_dataloader.batch_size)


        loss, acc = 0, 0
        total_tokens, loss_tokens = 0, 0
        for epoch in range(start_epoch, self.config.max_epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples = 0 if epoch > start_epoch else consumed_samples 
                )
            
            step_bar = tqdm(range(update_steps_per_epoch),
                            desc = f"Train step of epoch {epoch}",
                            disable = dist.get_rank() != 0)
            
            self.model.train()
            for inputs, attention_masks, loss_masks, seq_lens in self.train_dataloader:
                inputs = inputs.to(torch.cuda.current_device())                
                attention_mask = attention_masks.to(torch.cuda.current_device())
                loss_mask = loss_masks.to(torch.cuda.current_device())


                preference_loss, nll_loss, chosen_reward, reject_reward = self._get_loss_and_reward(
                    inputs, attention_mask, loss_masks, seq_lens
                )

                final_loss = preference_loss + nll_loss
                preference_acc = (chosen_reward > reject_reward).float().mean()


                self.strategy.backward(final_loss, self.model)
                self.strategy.optimizer_step(self.model)

                loss += final_loss.item()
                acc += preference_acc.item()

                total_tokens += sum(seq_lens)
                loss_tokens += loss_mask.int().sum().item()

                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict = dict(
                        mean_loss = loss / self.strategy.accumulated_gradient,
                        preference_loss = preference_loss.item(),
                        acc = acc,
                        chosen_reward = chosen_reward.mean().item(),
                        reject_reward = reject_reward.mean().item(),
                        lr = self.scheduler.get_last_lr()[0]
                    )
                    if self.config.nll_loss_weight > 1e-8:
                        logs_dict["nll_loss"] = nll_loss.item()

                    logs_dict = all_reduce(logs_dict)

                    loss, acc = 0, 0
                    
                    step_bar.set_postfix(logs_dict)
                    step_bar.update()
                    
                    global_step = step // self.strategy.accumulated_gradient

                    tokens_dict = dict(
                        total_tokens = all_reduce(total_tokens, op = "sum") / 10**9,
                        loss_tokens = all_reduce(loss_tokens, op = "sum") / 10**9
                    )

                    visualized_dict = { **{f"train/{k}": v for k, v in \
                                           {**logs_dict, "global_step": global_step}.items()},
                                        **{f"train/{k}(B)": v for k, v in tokens_dict.items()}
                                       }

                    self.update_visualization(visualized_dict,
                                              global_step,
                                              self.config.logging_steps,
                                              step = global_step)
                    self.evaluate(global_step)
                    self.save_ckpt(global_step = global_step, 
                                   client_state = dict(consumed_samples = global_step * self.config.batch_size))
                    
                    self.save_model(global_step = global_step)

                step += 1
            epoch_bar.update()
    
        self.finish_visualization()

    def evaluate(self, global_step:int = 0):
        if global_step % self.config.eval_steps: return
        self.model.eval()
        with torch.no_grad():
            loss, acc = 0, 0
            total_tokens, loss_tokens = 0, 0
            step_bar = tqdm(
                range(len(self.eval_dataloader)),
                desc = f"Eval stage of steps {global_step}",
                disable = dist.get_rank() == 0,
            )
            for times, (inputs, attention_masks, loss_masks, seq_lens) in enumerate(self.eval_dataloader):
                inputs = inputs.to(torch.cuda.current_device())
                attention_mask = attention_masks.to(torch.cuda.current_device())
                loss_mask = loss_masks.to(torch.cuda.current_device())

                preference_loss, nll_loss, chosen_reward, reject_reward = self._get_loss_and_reward(
                    inputs, attention_mask, loss_mask, seq_lens
                )

                final_loss = preference_loss + nll_loss
                preference_acc = (chosen_reward > reject_reward).float().mean()

                self.strategy.backward(final_loss, self.model)
                self.strategy.optimizer_step(self.model)

                loss += final_loss.item()
                acc += preference_acc.item()
                
                total_tokens += sum(seq_lens)
                loss_tokens += loss_mask.int().sum().item()

                bar_dict = {"eval_loss": loss / (times + 1),
                            "eval_acc" : acc / (times + 1)}

                step_bar.update()

            logs = all_reduce(bar_dict)
            step_bar.set_postfix(logs)
            
            tokens_dict = dict(
                total_tokens = all_reduce(total_tokens, op = "sum") / 10**9,
                loss_tokens = all_reduce(loss_tokens, op = "sum") / 10**9
            )
            visualized_dict = { **{f"eval/{k}": v for k, v in {**logs, "global_step": global_step}.items()},
                                **{f"eval/{k}(B)": v for k, v in tokens_dict.items()}
                               }

            self.update_visualization(
                visualized_dict, 
                global_step, 
                self.config.logging_steps, 
                step = global_step
            )
        
        self.model.train()

    
    def _get_loss_and_reward(self, 
                             inputs: torch.LongTensor, 
                             attention_mask: torch.LongTensor, 
                             loss_mask: torch.BoolTensor, 
                             seq_lens: list[int]):
        chosen_logps, reject_logps, nll_loss = self._get_chosen_rejected_logps(
            self.model, inputs, attention_mask, loss_mask, seq_lens
        )
        with torch.no_grad():
            ref_chosen_logps, ref_reject_logps, _ = self._get_chosen_rejected_logps(
                self.ref_model, inputs, attention_mask, loss_mask, seq_lens
            )
        loss, chosen_reward, reject_reward = self.loss_fn(
            chosen_logps, reject_logps, ref_chosen_logps, ref_reject_logps
        )

        return loss, nll_loss, chosen_reward, reject_reward



    def _get_chosen_rejected_logps(self, 
                                   model: GPT,
                                   inputs: torch.LongTensor, 
                                   attention_mask: torch.LongTensor, 
                                   loss_mask: torch.BoolTensor, 
                                   seq_lens: list[int]):
        
        all_logits = self._get_forward_logits(model, inputs, attention_mask, seq_lens)

        all_logps_sums, all_logps_means = self._get_logps_from_logits(
            logits = all_logits, 
            labels = inputs, 
            masks = attention_mask.bool() & loss_mask.bool(), 
            seq_lens = seq_lens
        )

        chosen_logps = all_logps_sums[: len(seq_lens) // 2]
        reject_logps = all_logps_sums[len(seq_lens) // 2: ]

        nll_loss = -all_logps_means[: len(seq_lens) // 2].mean() * self.config.nll_loss_weight \
            if self.config.nll_loss_weight > 1e-8 else 0

        return chosen_logps, reject_logps, nll_loss



    def _get_forward_logits(self,
                            model: GPT,
                            inputs: torch.LongTensor,
                            attention_mask: torch.LongTensor,
                            seq_lens: list[int]):
        if self.config.memory_save:
            l = sum(seq_lens[:len(seq_lens)//2])

            chosen_logits = model(
                sequences = inputs[: l],
                seq_lens = seq_lens[: len(seq_lens)//2],
                attention_mask = attention_mask[: l],
                ring_attn_group = self.strategy.ring_attn_group
            )["logits"]
            
            reject_logits = model(
                sequences = inputs[l: ],
                seq_lens = seq_lens[len(seq_lens)//2: ],
                attention_mask = attention_mask[l: ],
                ring_attn_group = self.strategy.ring_attn_group
            )["logits"]

            return torch.concat([chosen_logits, reject_logits], dim = 0)

        return model(
            sequences = inputs,
            seq_lens = seq_lens,
            attention_mask = attention_mask,
            ring_attn_group = self.strategy.ring_attn_group
        )["logits"]

    def _get_logps_from_logits(self,
                   logits: torch.FloatTensor,
                   labels: torch.LongTensor,
                   masks: torch.BoolTensor,
                   seq_lens: list[int]):
        assert logits.shape[:-1] == labels.shape
        if self.strategy.ring_attn_group is None:
            logps = logps_from_logits(logits = logits[:, :-1, :], labels = labels[:, 1:])
        else:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process + 1
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)

            local_label = labels[:, start_idx: end_idx]

            if self.strategy.ring_attn_rank + 1 == self.strategy.ring_attn_size:
                local_label = F.pad(local_label, (0, 1), value = 0)
            
            
            local_logps = logps_from_logits(logits = logits, labels = local_label)

            assert local_logps.ndim == 1, f"local_logps shape: {local_logps.shape}" 

            logps = all_gather(local_logps, self.strategy.ring_attn_group).reshape(1,-1)[:, : -1]
        
        loss_masks = masks[:, 1:]
        logps_sums, logps_means = [], []
        start_idx = 0
        for seq_len in seq_lens:
            seq = logps[0, start_idx: start_idx + seq_len - 1]
            mask = loss_masks[0, start_idx: start_idx + seq_len - 1]
            logps_sums +=[(seq @ mask)]
            logps_means += [(seq @ mask)/ mask.sum()]

            start_idx += seq_len
        
        return torch.stack(logps_sums), torch.stack(logps_means)