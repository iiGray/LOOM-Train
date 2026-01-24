# 🏗️ Loom-Train  
*A Simple & Efficient Training Framework for Long-Context LLMs/Agents*
<div class="image-container h-64 md:h-80">
  <img src="assets/LOOM-Train-LOGO.png" 
       alt="LOOM-Scope Logo" 
       title="LOOM-Scope Logo" 
       class="logo-image">
</div>


<p align="center">
  <a href="https://iigray.github.io/LOOM-Train/">
    <img src="https://img.shields.io/badge/Documentation-docs-orange?style=plastic&logo=readthedocs" alt="Documentation">
  </a>
</p>

> Optimized for scalability, memory efficiency, and seamless integration — built to unlock the full potential of long-context large language models.

---

## 📅 Update Log

- **📅 2026-01-20** — 🚀 **Initial Release**: Loom-Train is now live!  
  ✅ Native support for CUSTOM training tasks   
  ✅ Optimized attention with [🌀 Ring-Flash-Attention](https://github.com/zhuzilin/ring-flash-attention)  
  ✅ Lightweight, plug-and-play design for long-sequence training (128K+ tokens)

---

## ✨ Key Features

- 🔧 **Plug-and-Play**: Drop-in replacement for HF Trainer — no major code changes needed.  
- 🚀 **Memory-Efficient**: Leverages Ring-Flash-Attention to reduce GPU memory footprint by up to 50%.  
- 📈 **Scalable**: Seamlessly scales to 100K+ context lengths without sacrificing speed.  
- ⚡ **Fast Setup**: Minimal dependencies, easy installation via `pip install loom-train`.

---

## 💻 Environment & Installation

To install the`loomtrain` package from the gitee repository, run:

```bash
git clone https://github.com/LCM-Lab/LOOM-Train.git
conda create -n loom_train python=3.10 -y
conda activate loom_train
cd LOOM-Train/loomtrain
pip install -e .
```
To install flash attention, run the command below to obtain the required `flah-attn` version:
```bash
loomtrain-required-flash-attn
```

Download the suitable version of flash_attn from https://github.com/Dao-AILab/flash-attention/releases
```bash
pip install <path_to_flash_attn_whl_file>
pip install ring_flash_attn
```

---

## 🛠️ Getting Started
```python
from loomtrain.tasks import (
    SFTModule,
    SFTDataModule,

)
from loomtrain import core as lt
def train():
    args = lt.args()
     # You may also define your own training tasks by inheriting lt.Module / lt.DataModule, see the docs for more details
    module = SFTModule()
    datamodule = SFTDataModule(
        dataset_dicts = [
            lt.data.DatasetDict(pth, train_count = tc, val_count = vc) \
                for pth, tc, vc in zip(args.dataset_paths, args.train_samples, args.val_samples)
        ], max_length = args.max_data_length)
        
    lt.fit(
        module = module,
        datamodule = datamodule,
        train_strategy = lt.train_strategy.DeepspeedStrategy(),
        data_strategy = lt.data_strategy.SortPackingStrategy(),
    )
def sft_args(parser: "lt.ArgumentParser"):
    group = parser.add_argument_group("SFT Arguments")
    group.add_argument(
        "--model-path", type = str, required = True
    )
    group.add_argument(
        "--dataset-paths", type = str, nargs = "+", required = True
    )
    group.add_argument(
        "--train-samples", type = int, nargs = "+", required = True
    )
    group.add_argument(
        "--val-samples", type = int, nargs = "+", required = True
    )
    group.add_argument(
        "--prompt-key", type = str, default = "prompt"
    )
    group.add_argument(
        "--response-key", type = str, default = "response"
    )
if __name__ == "__main__":
    lt.add_extra_arguments_by(sft_args)
    train()
```

---

## 🤝 Contributing

We welcome contributions! Whether it’s bug fixes, new features, or documentation improvements — feel free to open an issue or PR.  
Let’s build the future of long-context training, together. 💪

---

## 📬 Contact

Questions? Suggestions? Reach out at: `iiiigray19@gmail.com`
