import torch
from loomtrain.core.actor import LoomOptDict
from loomtrain.core.module import LoomModule
from loomtrain.core.datamodule import LoomDataModule, LoomDataDict
from loomtrain.core.data import role_template, CollateDataset, BucketMixin, BlendedDataset
from loomtrain.core.parallel import parallel_state as parallel
from torch.nn import functional as F


class LoomSFTModule(LoomModule):
    def __init__(self, model_name: str):
        opt_dicts = dict(
            group0 = LoomOptDict(
                model_name = model_name,
                model_type = 'causal',
                loss_type = "sft",
                collate_type = "packing" # will be overrided if strategy also sets this argument.
            )
        )

        super().__init__(opt_dicts)
    
    def setup_self_module(self):
        self.actor = self.opt_groups['group0'].actor
        self.toknizer = self.opt_groups['group0'].tokenizer
        self.optimizer = self.opt_groups['group0'].optimizer
        self.scheduler = self.opt_groups['group0'].scheduler
        self.loss_fn = self.opt_groups['group0'].loss_fn

    def micro_batch_forward_backward(self, batch) -> "dict[str, object]":
        inputs, attention_mask, loss_mask, seq_lens = batch
        output = self.actor(sequences = inputs, attention_mask = attention_mask, seq_lens = seq_lens)
        labels = torch.where(attention_mask.bool() & loss_mask.bool(), inputs, self.loss_fn.ignore_index)

        gpt_loss = self.loss_fn(output.logits, labels)

        self.backward(self.actor, gpt_loss)

        return dict(
            loss = gpt_loss.item(),
            total_tokens = parallel.all_reduce(sum(seq_lens)) * parallel.get_dp_count() / 10 ** 9,
            loss_tokens = parallel.all_reduce(loss_mask.int().sum().item()) * parallel.get_dp_count() / 10 ** 9
        )

    def micro_batch_validate_forward(self, batch):
        inputs, attention_masks, loss_masks, seq_lens = batch
        output = self.actor(sequences = inputs, attention_masks = attention_masks,seq_lens = seq_lens)
        labels = torch.where(attention_masks.bool() & loss_masks.bool(), inputs, self.loss_fn.ignore_index)

        gpt_loss = self.loss_fn(output.logits, labels)

        return dict(
            loss = gpt_loss.item(),
            total_tokens = parallel.all_reduce(sum(seq_lens)) * parallel.get_dp_count() / 10 ** 9,
            loss_tokens = parallel.all_reduce(loss_masks.int().sum().item()) * parallel.get_dp_count() / 10 ** 9
        )
    
    def non_accum_logs_per_step(self):
        return dict(lr = self.scheduler.get_last_lr()[0])



class LoomSFTData(LoomDataModule):
    def __init__(self, 
                 data_dicts: "list[LoomDataDict]", 
                 max_length: int,
                 num_proc: "int" = 8):
        super().__init__(data_dicts)
        self.max_length = max_length
        self.cp_size = parallel.get_cp_size()
        self.num_proc = num_proc
        for data_dict in data_dicts:
            data_dict.max_length = max_length
            assert "prompt_key" in data_dict
            assert  "response_key" in data_dict

    @LoomDataModule.datasetmethod
    def get_loss_mask(dataset, input_ids, idx):
        loss_mask = torch.zeros_like(input_ids, dtype = torch.bool)
        prompt_ids_len = dataset.prompt_ids_lens[idx]
        # TODO: multi-turn
        loss_mask[0, prompt_ids_len: ] = True

        return loss_mask

    def dataset_initialize(dataset, self: "LoomSFTData", raw_dataset, data_dict):

        tokenizer = dataset.tokenizer = data_dict.tokenizer
        prompt_key = dataset.prompt_key = data_dict["prompt_key"]
        response_key = dataset.response_key = data_dict["response_key"]
        max_length = dataset.max_length = self.max_length

        def filter_data(data: "dict"):
            if max_length < 128000:
                prompt_template = data[prompt_key]
                response_template = role_template(data[response_key], "assistant")
                tokenized = tokenizer.apply_chat_template(
                    prompt_template + response_template, tokenize = True, 
                    max_length = 128000, padding = False,
                    truncation = True, return_tensors = 'pt'
                )
                if tokenized.numel() > max_length: return False
            return True

        def process_data(data):
            prompt_template = data[prompt_key]
            response_text = data[response_key]
            if isinstance(response_text, str):
                response_text = [{"role":"assistant", "content": response_text}]
            prompt = tokenizer.apply_chat_template(
                prompt_template, tokenize = False, add_generation_prompt = True
            )
            response = tokenizer.apply_chat_template(
                prompt_template + response_text, tokenize = False
            )[len(prompt): ]


            prompt_token = tokenizer(prompt, max_length = max_length,
                                        padding = False,
                                        truncation = True,
                                        return_tensors = 'pt',
                                        add_special_tokens = False)
            response_token = tokenizer(response, max_length = max_length,
                                            padding = False,
                                            truncation = True,
                                            return_tensors = 'pt',
                                            add_special_tokens = False)
            
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            input_ids_len = prompt_ids_len + response_token["attention_mask"].int().sum().item()

            return dict(
                prompt = prompt,
                response = response,
                prompt_ids_len = prompt_ids_len,
                input_ids_len = input_ids_len,
                response_ranges = None # not multiturn
            )

        raw_dataset = raw_dataset.filter(filter_data, num_proc = self.num_proc)
        processed_dataset = raw_dataset.map(process_data,
                                            remove_columns = raw_dataset.column_names,
                                            num_proc = self.num_proc)
        
        dataset.prompts = processed_dataset["prompt"]
        dataset.responses = processed_dataset["response"]
        dataset.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        # This attribute makes itself possible to be packed.
        dataset._input_ids_lens = processed_dataset["prompt_ids_len"]


    
    def dataset_len(dataset, self: "LoomSFTData"):
        return len(dataset.prompts)

    def dataset_getitem(dataset, self: "LoomSFTData", idx):
        prompt = dataset.prompts[idx]
        response = dataset.responses[idx]
        prompt_ids_len = dataset.prompt_ids_lens[idx]

        text = (prompt + response).rstrip("\n")
        if not text.endswith(dataset.tokenizer.eos_token):
            text += " " + dataset.tokenizer.eos_token
        
        input_token = dataset.tokenizer(
            text, max_length = self.max_length,
            padding = False,
            truncation = True,
            return_tensors = "pt",
            add_special_tokens = False
        )

        loss_mask = dataset.get_loss_mask(input_token["input_ids"], idx)

        input_token["input_ids"][0][-1] = dataset.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True


        return input_token["input_ids"], input_token["attention_mask"], loss_mask

    def dataset_collate_fn(dataset, self: "LoomSFTData", item_list):
        packed_input_ids = []
        packed_attention_masks = []
        packed_loss_masks = []
        seq_lens = []
        for index, (input_id, attention_mask, loss_mask) in enumerate(item_list):
            packed_input_ids += [input_id.flatten()]
            packed_attention_masks += [torch.full_like(input_id.flatten(), index + 1)]
            packed_loss_masks += [loss_mask.flatten()]
            seq_lens += [attention_mask.int().sum().item()]

        packed_input_ids = torch.concat(packed_input_ids).unsqueeze(0)
        packed_attention_masks = torch.concat(packed_attention_masks).unsqueeze(0)
        packed_loss_masks = torch.concat(packed_loss_masks).unsqueeze(0)

        if packed_input_ids.numel() % self.cp_size:
            padding_len = self.cp_size - (packed_input_ids.numel() % self.cp_size)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value = dataset.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value = 0)
            packed_loss_masks = F.pad(packed_loss_masks, (0, padding_len), value = 0)

        
        return packed_input_ids, packed_attention_masks, packed_loss_masks, seq_lens