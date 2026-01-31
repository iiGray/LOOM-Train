import torch, random
import loomtrain.core as lt
from loomtrain.core import parallel
from torch.nn import functional as F
from loomtrain.core import args


class SimPODataModule(lt.DataModule):
    def __init__(self, 
                 dataset_dicts: "list[lt.data.DatasetDict]", 
                 tokenizer_path: str = None,
                 max_length: int = None):
        super().__init__()

        if tokenizer_path is None:
            tokenizer_path = args().tokenizer_path
        if max_length is None:
            max_length = args().max_data_length

        self.max_length = max_length
        self.cp_size = parallel.get_cp_size()
        self.dataset_dict = lt.data.BlendedDatasetDict(dataset_dicts)

        self.tokenizer = lt.data.init_tokenizer(tokenizer_path)

    def filter_data(self, dataset: "lt.data.Dataset", data):
        if dataset.max_length < 128000:
            prompt_template = data[dataset.prompt_key]
            chosen_template = lt.role_template(data[dataset.chosen_key], "assistant")
            tokenized = self.tokenizer.apply_chat_template(
                prompt_template + chosen_template, tokenize = True, 
                max_length = 128000,padding = False,
                truncation = True
            )
            if len(tokenized) > self.max_length: return False
        reject = data[dataset.rejected_key]
        return (1 if isinstance(reject, str) else len(reject)) >= dataset.num_rejects

    def map_data(self, dataset: "lt.data.Dataset", data):
        rejects = [data[dataset.rejected_key]] \
            if isinstance(data[dataset.rejected_key], str)\
            else data[dataset.rejected_key]
        
        #keep num rejects the same as each other
        if len(rejects) > dataset.num_rejects:
            rejects = random.sample(rejects, dataset.num_rejects) \
                if dataset.sample_rejects is None else dataset.sample_rejects(rejects, dataset.num_rejects)

        prompt_template = data[dataset.prompt_key]
        if isinstance(prompt_template, str):
            prompt_template = lt.role_template(prompt_template, "user")
        chosen_template = lt.role_template(data[dataset.chosen_key], "assistant")
        rejected_templates = [lt.role_template(r, "assistant") for r in rejects]

        prompt = self.tokenizer.apply_chat_template(
            prompt_template, tokenize = False, add_generation_prompt = True
        )
        chosen = self.tokenizer.apply_chat_template(
            prompt_template + chosen_template, tokenize = False
        )[len(prompt): ]
        rejecteds = [self.tokenizer.apply_chat_template(
            prompt_template + r_template, tokenize = False
        )[len(prompt): ] for r_template in rejected_templates]
        
        #TODO: margin: contrast loss
        margin = data.get("margin", 0) 

        prompt_token = self.tokenizer(prompt,
                                      max_length = dataset.max_length,
                                      padding = False,
                                      truncation = True,
                                      return_tensors = 'pt',
                                      add_special_tokens = False)
        prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

        reponses = [self.tokenizer(r, max_length = dataset.max_length,
                                   padding = False,truncation = True,
                                   return_tensors = 'pt', add_special_tokens = False)\
                                    for r in [chosen] + rejecteds]
        responses_ids_len = [r['attention_mask'].int().sum().item() for r in reponses]

        return dict(
            prompt = prompt,
            chosen = chosen,
            rejects = rejecteds,
            prompt_ids_len = prompt_ids_len, # for spliting prompt and chosen/reject
            input_ids_len = prompt_ids_len + max(responses_ids_len)
        )

    def get_loss_mask(self, input_ids:torch.Tensor, prompt_ids_len):
        loss_mask = torch.zeros_like(input_ids, dtype = torch.bool)
        loss_mask[0, prompt_ids_len: ] = True
        loss_mask[0, -1] = True

        return loss_mask

    def get_data(self, dataset: "lt.data.Dataset", data):
        prompt, chosen, rejects = data['prompt'], data['chosen'], data['rejects']
        prompt_ids_len = data['prompt_ids_len']

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen, max_length = self.max_length,
            padding = False,
            truncation = True,
            return_tensors = "pt",
            add_special_tokens = False
        )
        chosen_loss_mask = self.get_loss_mask(chosen_token["input_ids"], prompt_ids_len)
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True

        rejects_input_ids, rejects_attn_mask, rejects_loss_mask = [], [], []
        for reject in rejects: # multiple rejected sequences
            reject = (prompt + reject).rstrip("\n")
            if not reject.endswith(self.tokenizer.eos_token):
                reject += " " + self.tokenizer.eos_token
            reject_token = self.tokenizer(
                reject, max_length = dataset.max_length,
                padding = False,
                truncation=  True,
                return_tensors = "pt",
                add_special_tokens = False
            )
            reject_loss_mask = self.get_loss_mask(reject_token["input_ids"], prompt_ids_len)
            reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            reject_token["attention_mask"][0][-1] = True

            rejects_input_ids += [reject_token['input_ids']]
            rejects_attn_mask += [reject_token['attention_mask']]
            rejects_loss_mask += [reject_loss_mask]

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            chosen_loss_mask,
            rejects_input_ids,
            rejects_attn_mask,
            rejects_loss_mask
        )
    
    def set_dataset_properties(self, dataset: "lt.data.Dataset"):
        with dataset.disable_get_fn():
            dataset.set_input_ids_lens(
                [k['input_ids_len'] for k in dataset]
            )

    def collate_fn(self, item_list):
        '''
        returns:
            packed sequence,  packed_seq_lens
        '''
        # the first element of each is chosen, while others are rejected
        packed_input_ids_list = [[] for _ in range(args().num_rejects + 1)]
        packed_attention_masks_list = [[] for _ in range(args().num_rejects + 1)]
        packed_loss_masks_list = [[] for _ in range(args().num_rejects + 1)]
        seq_lens_list = [[] for _ in range(args().num_rejects + 1)]

        merged_seq_lens = [0 for _ in range(args().num_rejects + 2)]

        for index, (chosen_id, chosen_attention_mask, chosen_loss_mask,
                    rejects_id, rejects_attention_mask, rejects_loss_mask) in enumerate(item_list):
            packed_input_ids_list[0] += [chosen_id.flatten()]
            packed_attention_masks_list[0] += [torch.full_like(chosen_id.flatten(), index + 1)]
            packed_loss_masks_list[0] += [chosen_loss_mask.flatten()]
            seq_lens_list[0] += [len(chosen_id.flatten())]

            for r_index, (reject_id, reject_attention_mask, reject_loss_mask) in enumerate(zip(
                rejects_id[: args().num_rejects], rejects_attention_mask[: args().num_rejects], rejects_loss_mask[: args().num_rejects]
            )):

                packed_input_ids_list[r_index + 1] += [reject_id.flatten()]
                packed_attention_masks_list[r_index + 1] += [torch.full_like(reject_id.flatten(), 
                                                                             index + 1 + (r_index + 1) * len(item_list))]
                packed_loss_masks_list[r_index + 1] += [reject_loss_mask.flatten()]
                seq_lens_list[r_index + 1] += [len(reject_id.flatten())]
        

        for i_index in range(args().num_rejects + 1):
            packed_input_ids_list[i_index] = torch.concat(packed_input_ids_list[i_index]).unsqueeze(0)
            packed_attention_masks_list[i_index] = torch.concat(packed_attention_masks_list[i_index]).unsqueeze(0)
            packed_loss_masks_list[i_index] = torch.concat(packed_loss_masks_list[i_index]).unsqueeze(0)
            merged_seq_lens[i_index + 1] = sum(seq_lens_list[i_index]) + merged_seq_lens[i_index]
        
        packed_input_ids = torch.concat(packed_input_ids_list, dim = -1)
        packed_attention_masks = torch.concat(packed_attention_masks_list, dim = -1)
        packed_loss_masks = torch.concat(packed_loss_masks_list, dim = -1)
        packed_seq_lens = [sl for seq_lens in seq_lens_list for sl in seq_lens]

        if packed_input_ids.numel() % parallel.get_cp_size():
            padding_len = parallel.get_cp_size() - (packed_input_ids.numel() % parallel.get_cp_size())
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value = self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value = 0)
            packed_loss_masks = F.pad(packed_loss_masks, (0, padding_len), value = 0)
            
        


        return (packed_input_ids, packed_attention_masks, packed_loss_masks, 
                seq_lens_list, packed_seq_lens, merged_seq_lens)

    def get_train_dataset(self):
        return self.dataset_dict["train"]
    
    def get_val_dataset(self):
        return self.dataset_dict[args().val_split]


class SimPOModule(lt.Module):
    def __init__(self, model_path: str = None, tokenizer_path: str = None, model_type: str = "causal", collate_type = "packing", 
                 optim_config: "lt.OptimConfig | dict[str, lt.OptimConfig]" = lt.OptimConfig()):
        super().__init__(optim_configs = optim_config)
        if model_path is None: model_path = args().model_path
        if tokenizer_path is None: tokenizer_path = args().tokenizer_path

        self.actor = lt.modeling.init_actor(model_path = model_path,
                                            model_type = model_type,
                                            collate_type = collate_type)
        
        self.loss_fn = lt.modeling.init_loss_fn(loss_type = "simpo")

        self.tokenizer = lt.modeling.init_tokenizer(tokenizer_path if tokenizer_path else model_path)
    
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
            rewards_rect = lt.Accum(parallel.all_reduce(reject_reward.sum().item())),

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
        loss, chosen_reward, reject_reward = self.loss_fn(
            chosen_logps, reject_logps
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
        loss_masks_list = [masks[:, l: r] for l, r in zip(merged_seq_lens[:-1],merged_seq_lens[1:])]
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