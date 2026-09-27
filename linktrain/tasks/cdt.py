import torch
import linktrain.core as lt
from linktrain.core import parallel
from torch.nn import functional as F
from linktrain.core import args
from linktrain.tasks.sft import SFTDataModule

class CDTDataModule(SFTDataModule): ...