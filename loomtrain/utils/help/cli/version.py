import os, sys, argparse
from pathlib import Path


sys.path.append(
    str(Path(__file__).absolute().parents[3])
)

from loomtrain.utils.help.version import Version

def required_flash_attn():
    return Version.required_flash_attn()
