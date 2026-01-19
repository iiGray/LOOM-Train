from typing import Optional, Union, Literal
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import deepspeed
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
from transformers import PreTrainedModel
from loomtrain.core.parallel import parallel_state as parallel


def get_loss_cls(loss_type: Literal["sft", "simpo"] = "sft"):
    if loss_type == "sft":
        return GPTCELoss
    elif loss_type == "simpo":
        return SimPOLoss

def init_loss_fn(loss_type: Literal["sft", "simpo"] = "sft", *loss_args, **loss_kwargs):
    return get_loss_cls(loss_type)(*loss_args, **loss_kwargs)


class GPTCELoss(nn.Module):
    def __init__(self, ignore_index:int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss = nn.CrossEntropyLoss(ignore_index = ignore_index)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        '''
        logits: ring local logits (including the last useless logit)
        '''
        
        total_seq_len = labels.size(-1)
        seq_len_per_process = total_seq_len // parallel.get_cp_size()
        start_idx = parallel.get_cp_rank() * seq_len_per_process + 1
        
        end_idx = min(start_idx + seq_len_per_process, total_seq_len)
        
        full_labels = labels
        labels = labels[..., start_idx: end_idx] #shift_logits is automatic between star_idx and end_idx

        if parallel.get_cp_rank() + 1 == parallel.get_cp_size():
            labels = F.pad(labels, (0, 1), value = self.loss.ignore_index)

        shift_logits = logits.contiguous()
        shift_labels = labels.contiguous()

        if torch.all(shift_labels == self.ignore_index):
            loss = shift_logits.mean() * 0
        else:
            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        calculated = (shift_labels != self.ignore_index).long().sum().item()
        all_calculated = (full_labels != self.ignore_index).long().sum().item()

        loss *= (calculated / max(all_calculated, 1))

        dist.all_reduce(loss, op = dist.ReduceOp.SUM, group = parallel.get_cp_group())

        return loss



def logps_from_logits(logits: torch.FloatTensor,
                      labels: torch.LongTensor,
                      temperature: float = 1.0,
                      ignore_index: int = -100):
    assert logits.ndim == 2 and labels.ndim == 1, \
        f"logits shape: {logits.shape}, labels shape: {labels.shape}"
    if temperature != 1.0:
        logits.div_(temperature)
    
    if logits.dtype in (torch.float32, torch.float64):
        dims = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logps, _ = cross_entropy_loss(logits.reshape(-1, last_dim), labels.reshape(-1),
                                      ignore_index = ignore_index)
        
        return -logps.view(*dims)
    safe_mask = (labels != ignore_index)
    safe_labels = labels.clone().masked_fill_(~safe_mask, 0)
    return (F.log_softmax(logits, dim = -1)\
            .gather(dim = -1, index = safe_labels.unsqueeze(-1)).squeeze(-1)) * safe_mask.float()


class DPOLoss(nn.Module):
    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo
    
    def forward(self,
                policy_chosen_logps: torch.FloatTensor,
                policy_rejected_logps: torch.FloatTensor,
                reference_chosen_logps: torch.FloatTensor,
                reference_rejected_logps: torch.FloatTensor):
        policy_delta = policy_chosen_logps - policy_rejected_logps
        reference_delta = reference_chosen_logps - reference_chosen_logps

        logits = policy_delta - reference_delta

        if self.ipo:
            loss = (logits - 1 / (2 * self.beta)) ** 2 # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            loss = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                -F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss.mean(), chosen_rewards, rejected_rewards


class SimPOLoss(nn.Module):
    def __init__(self, 
                 beta: float = 2.0, gamma: float = 0.5, 
                 label_smoothing: float = 0.0, 
                 ltype: Literal["sigmoid", "hing"] = "sigmoid"):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.gamma_beta_ratio = gamma / beta
        self.label_smoothing = label_smoothing        
        self.ltype = ltype
    def forward(self,
                policy_chosen_logps: torch.FloatTensor,
                policy_rejected_logps: torch.FloatTensor,
                beta: float = None, gamma: float = None, 
                label_smoothing: float = None, 
                ltype: Literal["sigmoid", "hing"] = None):
        '''
        policy_rejected_logps may be the mean of multiple rejected_logps 
        '''

        if beta is None: beta = self.beta
        if gamma is None: gamma = self.gamma
        if label_smoothing is None: label_smoothing = self.label_smoothing
        if ltype is None: ltype = self.ltype
        gamma_beta_ratio = gamma / beta

        policy_delta = (policy_chosen_logps - policy_rejected_logps).flatten()
        logits = policy_delta - gamma_beta_ratio


        if ltype == "hinge":
            loss = torch.relu(1 - beta * logits)
        elif ltype == "sigmoid":
            loss = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing)
                -F.logsigmoid(-beta * logits) * label_smoothing
            )
        else:
            raise ValueError(
                f"Unknown loss type: {ltype}. Should be one of ['sigmoid', 'hinge']"
            )

        
        chosen_rewards = beta * policy_chosen_logps.detach()
        rejected_rewards = beta * policy_rejected_logps.detach()

        return loss.mean(), chosen_rewards, rejected_rewards

