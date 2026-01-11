from loomtrain.utils.common.iotools import read_json, save_json, path_join
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model,
    PeftModel, 
    PeftModelForCausalLM, 
    PeftConfig
)
from copy import deepcopy
from safetensors import safe_open
import torch

from dataclasses import dataclass

@dataclass
class LoRAConfig(LoraConfig):
    task_type: str = TaskType.CAUSAL_LM
    save_merged: bool = True
    target_modules: list | str = "all-linear"
    def __post_init__(self):
        super().__post_init__()
        self.task_type = TaskType.CAUSAL_LM
        self.bias = "none"



def merge_peft_model(model_path: str,
                     adapter_path: str = None,
                     save_path: str = None):
    # check config, remove lora modules that actually has no attr.
    adapter_config_path = path_join(adapter_path, "adapter_config.json")
    adapter_config = read_json(adapter_config_path)
    target_modules = [k for k in adapter_config["target_modules"]]
    remove_modules = set()
    with safe_open(path_join(adapter_path, "adapter_model.safetensors"), 
                   framework = "pt", device = "cpu") as f:
        for k in f.keys():
            if f.get_tensor(k).numel() == 0:
                for target_module in target_modules:
                    if target_module in k: remove_modules.add(target_module)

    if remove_modules:
        for k in remove_modules: target_modules.remove(k)
        tmp_adapter_config = read_json(adapter_config_path)
        tmp_adapter_config["target_modules"] = target_modules
        save_json(tmp_adapter_config, 
                  adapter_config_path, indent = 2)


    if save_path == None:
        save_path = path_join(adapter_path, "merged")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2="flash_attention_2",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code = True
        )

        if adapter_path is not None:
            adapter_model = PeftModel.from_pretrained(model, adapter_path, ignore_mismatched_sizes=False)
            print("Loading PEFT adapter from:", adapter_path)
            print("Merging model and PEFT adapter weights...")
            model = adapter_model.merge_and_unload()
        
        print(f"Saving merged model to {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    except Exception as e: raise e
    finally:
        if remove_modules:
            save_json(adapter_config, 
                      adapter_config_path, indent = 2)
    print("Merged Model saved successfully!")

