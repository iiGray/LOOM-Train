import torch
import torch.nn.functional as F
from loomtrain.dataset.base import CollateDataset
from loomtrain.utils.sequence import pad_sequences
from transformers import PreTrainedTokenizer

import datasets

    
class LMDataset(CollateDataset):
    def __init__(
            self,
            dataset: datasets.Dataset, # raw_dataset
            text_key: str,
            tokenizer: PreTrainedTokenizer,
            max_length: int, # max_length for tokenizer
            num_processors: int = 8,
            ring_attn_size: int = None, 
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ring_attn_size = ring_attn_size

        self.text_key = text_key
        self.dataset = dataset#.filter(lambda x: x['ctx_length'] in ('128k',))
        processed_dataset = dataset.map(self.process_data,
                                        remove_columns = dataset.column_names,
                                        num_proc = num_processors)

        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.input_ids_lens = processed_dataset["input_ids_len"]
    
    def process_data(self, data):
        text = data[self.text_key]

        prompt_token = self.tokenizer(text, max_length = self.max_length,
                                      padding = False,
                                      truncation = True,
                                      return_tensors = 'pt',
                                      add_special_tokens = False)
        
        prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

        return dict(
            prompt = text,
            input_ids_len = prompt_ids_len,
        )

    def __len__(self): return len(self.prompts)

    def __getitem__(self, idx):
        text = self.prompts[idx]
        if not text.endswith(self.tokenizer.eos_token):
            text += " " + self.tokenizer.eos_token
        
        input_token = self.tokenizer(
            text, max_length = self.max_length,
            padding = False,
            truncation = True,
            return_tensors = "pt",
            add_special_tokens = False
        )

        loss_mask = self.get_loss_mask(input_token["input_ids"], idx)

        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True


        return input_token["input_ids"], input_token["attention_mask"], loss_mask


    def get_loss_mask(self, input_ids, idx):
        loss_mask = torch.zeros_like(input_ids, dtype = torch.bool)
        prompt_ids_len = self.prompt_ids_lens[idx]
        # TODO: multi-turn
        loss_mask[0, prompt_ids_len: ] = True

        return loss_mask


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

        if packed_input_ids.numel() % self.ring_attn_size:
            padding_len = self.ring_attn_size - (packed_input_ids.numel() % self.ring_attn_size)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value = self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value = 0)
            packed_loss_masks = F.pad(packed_loss_masks, (0, padding_len), value = 0)

        
        return packed_input_ids, packed_attention_masks, packed_loss_masks, seq_lens
        


        