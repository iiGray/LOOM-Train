import torch
import loomtrain.core as lt
from loomtrain.core import parallel
from torch.nn import functional as F
from loomtrain.core import args
from loomtrain.tasks.sft import SFTModule

# pretrain
class PTModule(SFTModule): ...

class PTDataModule(lt.DataModule):
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


    def map_data(self, dataset: "lt.data.Dataset", data):
        text = data[dataset.text_key]
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
        text = data['prompt']
        if not text.endswith(self.tokenizer.eos_token):
            text += " " + self.tokenizer.eos_token
        
        input_token = self.tokenizer(
            text, max_length = self.max_length,
            padding = False,
            truncation = True,
            return_tensors = "pt",
            add_special_tokens = False
        )

        loss_mask = torch.ones_like(input_token["input_ids"], dtype = torch.bool)

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
