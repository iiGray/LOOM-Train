import torch
import loomtrain.core as lt
from loomtrain.core import parallel
from torch.nn import functional as F
from loomtrain.core import args

class SFTModule(lt.Module):
    def __init__(self, model_path: str = None, tokenizer_path: str = None, model_type: str = "causal", collate_type = "packing", 
                 optim_config: "lt.OptimConfig | dict[str, lt.OptimConfig]" = lt.OptimConfig()):
        super().__init__(optim_configs = optim_config)

        if model_path is None: model_path = args().model_path
        if tokenizer_path is None: tokenizer_path = args().tokenizer_path

        self.actor = lt.modeling.init_actor(model_path = model_path,
                                            model_type = model_type,
                                            collate_type = collate_type)
        self.loss_fn = lt.modeling.init_loss_fn(loss_type = "ce")

        self.toknizer = lt.data.init_tokenizer(tokenizer_path if tokenizer_path else model_path)

    def micro_batch_forward_backward(self, batch) -> "lt.AccumLogDict[str, lt.Accum]":
        inputs, attention_mask, loss_mask, seq_lens = batch
        output = self.actor(sequences = inputs, attention_mask = attention_mask, seq_lens = seq_lens)
        labels = torch.where(attention_mask.bool() & loss_mask.bool(), inputs, self.loss_fn.ignore_index)

        gpt_loss = self.loss_fn(output.logits, labels)

        self.backward(gpt_loss, actor_of_the_loss = self.actor)

        return lt.AccumLogDict(
            loss = lt.Accum(gpt_loss.item(), dtype = "mean"),
            total_tokens = lt.Accum(parallel.all_reduce(sum(seq_lens)) * parallel.get_dp_count() / 10 ** 9, 
                                    dtype = "sum", is_global = True),
            loss_tokens = lt.Accum(parallel.all_reduce(loss_mask.int().sum().item()) * parallel.get_dp_count() / 10 ** 9, 
                                   dtype = "sum", is_global = True),
        )

    def batch_validate_forward(self, batch) -> "lt.AccumLogDict[str, lt.Accum]":
        inputs, attention_masks, loss_masks, seq_lens = batch
        output = self.actor(sequences = inputs, attention_mask = attention_masks,seq_lens = seq_lens)
        labels = torch.where(attention_masks.bool() & loss_masks.bool(), inputs, self.loss_fn.ignore_index)

        gpt_loss = self.loss_fn(output.logits, labels)

        return lt.AccumLogDict(
            loss = lt.Accum(gpt_loss.item()),
            total_tokens = lt.Accum(parallel.all_reduce(sum(seq_lens)) * parallel.get_dp_count() / 10 ** 9, 
                                    dtype = "sum"),
            loss_tokens = lt.Accum(parallel.all_reduce(loss_masks.int().sum().item()) * parallel.get_dp_count() / 10 ** 9, 
                                   dtype = "sum")
        )
    
    def non_accum_logs_per_step(self) -> "lt.LogDict[str, object]":
        return lt.LogDict(lr = self.actor.scheduler.get_last_lr()[0])


class SFTDataModule(lt.DataModule):
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
            response_template = lt.role_template(data[dataset.response_key], "assistant")
            tokenized = self.tokenizer.apply_chat_template(
                prompt_template + response_template, tokenize = True, 
                max_length = 128000, padding = False,
                truncation = True, return_tensors = 'pt'
            )
            if tokenized.numel() > dataset.max_length: return False
        return True

    def map_data(self, dataset: "lt.data.Dataset", data):
        prompt_template = data[dataset.prompt_key]
        response_text = data[dataset.response_key]
        if isinstance(response_text, str):
            response_text = [{"role":"assistant", "content": response_text}]
        prompt = self.tokenizer.apply_chat_template(
            prompt_template, tokenize = False, add_generation_prompt = True
        )
        response = self.tokenizer.apply_chat_template(
            prompt_template + response_text, tokenize = False
        )[len(prompt): ]


        prompt_token = self.tokenizer(prompt, max_length = dataset.max_length,
                                      padding = False,
                                      truncation = True,
                                      return_tensors = 'pt',
                                      add_special_tokens = False)
        response_token = self.tokenizer(response, max_length = dataset.max_length,
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
    def set_dataset_properties(self, dataset: "lt.data.Dataset"):
        with dataset.disable_get_fn():
            dataset.set_input_ids_lens(
                [k['input_ids_len'] for k in dataset]
            )

    def get_loss_mask(self, prompt_ids_len, input_ids):
        loss_mask = torch.zeros_like(input_ids, dtype = torch.bool)
        
        # TODO: multi-turn
        loss_mask[0, prompt_ids_len: ] = True

        return loss_mask

    def get_data(self, dataset: "lt.data.Dataset", data):
        prompt = data['prompt']
        response = data['response']
        prompt_ids_len = data['prompt_ids_len']

        text = (prompt + response).rstrip("\n")
        if not text.endswith(self.tokenizer.eos_token):
            text += " " + self.tokenizer.eos_token
        
        input_token = self.tokenizer(
            text, max_length = dataset.max_length,
            padding = False,
            truncation = True,
            return_tensors = "pt",
            add_special_tokens = False
        )

        loss_mask = self.get_loss_mask(prompt_ids_len, input_token["input_ids"])

        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True


        return input_token["input_ids"], input_token["attention_mask"], loss_mask


    def collate_fn(self, item_list):

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
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value = self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value = 0)
            packed_loss_masks = F.pad(packed_loss_masks, (0, padding_len), value = 0)

        
        return packed_input_ids, packed_attention_masks, packed_loss_masks, seq_lens


    def get_train_dataset(self):
        return self.dataset_dict["train"]
    
    def get_val_dataset(self):
        return self.dataset_dict[args().val_split]
