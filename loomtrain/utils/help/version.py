from tabulate import tabulate
import torch.version


def check_python():
    import sys
    return sys.version

def check_torch():
    import torch
    return torch.__version__

def check_cuda():
    import torch
    return torch.version.cuda

def check_nccl():
    import torch
    if torch.cuda.is_available():
        return ".".join(map(str,
                            torch.cuda.nccl.version()))
def check_cxx11abi():
    import torch
    return torch._C._GLIBCXX_USE_CXX11_ABI
    
def check_transformers():
    import transformers
    return transformers.__version__

def check_vllm():
    import vllm
    return vllm.__version__

def check_triton():
    import triton
    return triton.__version__



from typing import Literal
class Version:
    checkers = {
        'python'      : check_python,
        'torch'       : check_torch,
        "cuda"        : check_cuda,
        'nccl'        : check_nccl,
        'transformers': check_transformers,
        'vllm'        : check_vllm,
        'triton'      : check_triton,
        'cxx11abi'    : check_cxx11abi
    }

    @classmethod
    def check(cls, module_name: Literal["torch",
                                        "nccl",
                                        "cxx11abi",
                                        "transformers",
                                        "vllm",
                                        
                                        "triton"]):
        assert module_name in Version.checkers,\
        f"`{module_name}` is currently not supported."
        return Version.checkers[module_name]()
    
    @classmethod
    def check_all(cls):
        data = [(k, v()) for k,v in Version.checkers.items()]
        return tabulate(data, headers = ["Package", 
                                         "Version"])

    @classmethod
    def required_flash_attn(cls):
        torch_version = check_torch()[:3]
        cuda_version = check_cuda().split(".")[0]
        cxx11abi_version = str(check_cxx11abi()).upper()
        python_version = check_python().split()[0].split(".")[:2]
        python_version = "".join([str(k) for k in python_version])

        return (f"flash_attn-x.x.x+cu{cuda_version}"
                f"torch{torch_version}"
                f"cxx11abi{cxx11abi_version}"
                f"-cp{python_version}"
                "-linux_x86_64.whl")
        return torch_version,cuda_version,cxx11abi_version,python_version