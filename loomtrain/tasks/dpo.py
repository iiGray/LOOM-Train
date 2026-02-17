import torch
import loomtrain.core as lt
from loomtrain.core import parallel
from torch.nn import functional as F
from loomtrain.core import args
from loomtrain.tasks.simpo import SimPODataModule

class DPODataModule(SimPODataModule): ...


class DPOModule(lt.Module):
    def __init__(self, model_path: str = None, ref_model_path: str = None, tokenizer_path: str = None, 
                 model_type: str = "causal", collate_type = "packing", 
                 optim_config: "lt.OptimConfig | dict[str, lt.OptimConfig]" = lt.OptimConfig()):
        super().__init__(optim_configs = optim_config)

        if model_path is None: model_path = args().model_path
        if ref_model_path is None: ref_model_path = args().model_path
        if tokenizer_path is None: tokenizer_path = args().tokenizer_path

        self.actor = lt.modeling.init_actor(model_path = model_path,
                                            model_type = model_type,
                                            collate_type = collate_type)
        self.ref_actor = lt.modeling.init_actor(ref_model_path, model_type,
                                                collate_type, trainable = False)
        self.loss_fn = lt.modeling.init_loss_fn(loss_type = "dpo")

        self.toknizer = lt.data.init_tokenizer(tokenizer_path if tokenizer_path else model_path)


    
    def micro_batch_forward_backward(self, batch) -> "lt.AccumLogDict[str, lt.Accum]":
        (inputs_ids, attention_masks, loss_masks, 
                 seq_lens_list, packed_seq_lens, merged_seq_lens) = batch
        
        preference_loss, nll_loss, chosen_reward, reject_reward = self._get_loss_and_reward(
            inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens
        )

        final_loss = preference_loss + nll_loss
        preference_correct = (chosen_reward > reject_reward).float().sum()

        self.backward(final_loss, self.actor)

        loss_token = loss_masks.int().sum().item()
        total_token = (sum(sum(k) for k in seq_lens_list) + loss_token * (len(seq_lens_list) - 1))\
                      /(len(seq_lens_list))

        return lt.AccumLogDict(
            mean_loss = lt.Accum(parallel.all_reduce(final_loss.item())),
            preference_loss = lt.Accum(parallel.all_reduce(preference_loss.item())),
            rewards_chosen = lt.Accum(parallel.all_reduce(chosen_reward.sum().item())),
            rewards_reject = lt.Accum(parallel.all_reduce(reject_reward.sum().item())),
            preference_acc = lt.Accum(parallel.all_reduce(preference_correct)),
 
            total_tokens = lt.Accum(parallel.all_reduce(total_token) * parallel.get_dp_count() / 10 ** 9, 
                                    dtype = "sum", is_global = True),
            loss_tokens = lt.Accum(parallel.all_reduce(loss_token) * parallel.get_dp_count() / 10 ** 9, 
                                   dtype = "sum", is_global = True),
        )

    def non_accum_logs_per_step(self):
        return lt.LogDict(
            lr = self.actor.scheduler.get_last_lr()[0]
        )

    def batch_validate_forward(self, batch):
        (inputs_ids, attention_masks, loss_masks, 
                 seq_lens_list, packed_seq_lens, merged_seq_lens) = batch
        
        preference_loss, nll_loss, chosen_reward, reject_reward = self._get_loss_and_reward(
            inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens
        )

        final_loss = preference_loss + nll_loss
        preference_correct = (chosen_reward > reject_reward).float().sum()

        loss_token = loss_masks.int().sum().item()
        total_token = (sum(sum(k) for k in seq_lens_list) + loss_token * (len(seq_lens_list) - 1))\
                /(len(seq_lens_list))


        return lt.AccumLogDict(
            loss = lt.Accum(parallel.all_reduce(final_loss.item())),
            acc = lt.Accum(parallel.all_reduce(preference_correct.item())),
            rewards_chosen = lt.Accum(parallel.all_reduce(chosen_reward.sum().item())),
            rewards_reject = lt.Accum(parallel.all_reduce(reject_reward.sum().item())),

            total_tokens = lt.Accum(parallel.all_reduce(total_token, op = "sum") / 10**9, dtype = "sum", is_global = True),
            loss_tokens = lt.Accum(parallel.all_reduce(loss_token, op = "sum") / 10**9, dtype = "sum", is_global = True)
        )        

    def _get_loss_and_reward(self, 
                             inputs_ids: torch.LongTensor, 
                             attention_masks: torch.LongTensor, 
                             loss_masks: torch.BoolTensor, 
                             seq_lens_list: list[list[int]], 
                             packed_seq_lens: list[int], 
                             merged_seq_lens: list[int]
                             ):
        '''The first element of each input belongs to chosen, others belong to rejected'''
        
        chosen_logps, reject_logps, nll_loss = self._get_logps(
            self.actor, inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens
        )
        with torch.no_grad():
            ref_chosen_logps, ref_reject_logps, *_ = self._get_logps(
                self.ref_actor, inputs_ids, attention_masks, loss_masks, seq_lens_list, packed_seq_lens, merged_seq_lens
            )
        loss, chosen_reward, reject_reward = self.loss_fn(
            chosen_logps, reject_logps, ref_chosen_logps, ref_reject_logps
        )

        return loss, nll_loss, chosen_reward, reject_reward

    def _get_logps(self, 
                   model: "lt.modeling.Actor",
                   inputs_ids: torch.LongTensor, 
                   attention_masks: torch.LongTensor, 
                   loss_masks: torch.BoolTensor, 
                   seq_lens_list: list[list[int]],
                   packed_seq_lens: list[int], 
                   merged_seq_lens: list[int]):
        
        packed_local_logits = model(
            sequences = inputs_ids,
            seq_lens = packed_seq_lens,
            attention_mask= attention_masks,
        )["logits"]

        logps_sums_list, logps_means_list = self._get_logps_from_packed_logits(
            logits = packed_local_logits, labels = inputs_ids,
            masks = attention_masks.bool() & loss_masks.bool(),
            seq_lens_list = seq_lens_list,
            merged_seq_lens = merged_seq_lens
        )

        chosen_logps = logps_means_list[0]
        reject_logps = torch.stack(logps_means_list[1:]).mean(dim = 0)

        nll_loss = -logps_means_list[0].mean() * args().nll_loss_weight \
            if args().nll_loss_weight > 1e-8 else 0

        return chosen_logps, reject_logps, nll_loss



    def _get_logps_from_packed_logits(self,
                                      logits: torch.FloatTensor,
                                      labels: torch.LongTensor,
                                      masks: torch.BoolTensor,
                                      seq_lens_list: list[int],
                                      merged_seq_lens: list[int]):
        '''logits: [1, local_seq_len, vocab_size], labels: [1, seq_len]'''
        if parallel.get_cp_group() is None: # useless ???
            full_logps = lt.modeling.logps_from_logits(logits = logits[:, :-1, :], 
                                                       labels = labels[:, 1:])
        else:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // parallel.get_cp_size()
            start_idx = parallel.get_cp_rank() * seq_len_per_process + 1
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)

            local_label = labels[:, start_idx: end_idx]

            if parallel.is_last_cp_rank():
                local_label = F.pad(local_label, (0, 1), value = 0)
            
            
            local_logps = lt.modeling.logps_from_logits(logits = logits.squeeze(0), 
                                                        labels = local_label.squeeze(0))

            assert local_logps.ndim == 1, f"local_logps shape: {local_logps.shape}" 

            full_logps = parallel.flash_attn_all_gather(local_logps, parallel.get_cp_group()).reshape(1,-1)[:, : -1]
        
        logps_list = [full_logps[:, l: r] for l, r in zip(merged_seq_lens[:-1], merged_seq_lens[1:])]
        loss_masks_list = [masks[:, l: r] for l, r in zip(merged_seq_lens[:-1], merged_seq_lens[1:])]
        logps_sums_list, logps_means_list = [], []

        for logps, mask, seq_lens in zip(logps_list, loss_masks_list, seq_lens_list):
            loss_masks = mask[:, 1:].float()
            logps_sums, logps_means = [], []
            start_idx = 0
            for seq_len in seq_lens:
                seq = logps[0, start_idx: start_idx + seq_len - 1]
                mask = loss_masks[0, start_idx: start_idx + seq_len - 1]
                logps_sums +=[(seq @ mask)]
                logps_means += [(seq @ mask)/ mask.sum()]

                start_idx += seq_len
    
            logps_sums_list += [torch.stack(logps_sums)]
            logps_means_list += [torch.stack(logps_means)]
        
        return logps_sums_list, logps_means_list