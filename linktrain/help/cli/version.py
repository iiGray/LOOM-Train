import os, sys
from pathlib import Path


sys.path.append(
    str(Path(__file__).absolute().parents[3])
)

from linktrain.utils.common.args import AutoParser, Arg
from linktrain.help.version import Version

def main():
    args = AutoParser([
        Arg(name = "package", short = "p", type = str),
        Arg(name = "show", short = "s", action = 'store_true')
    ]).parse_args()

    if args.show:
        print("\n".join(Version.checkers.keys()))
    else:
        print(Version.check(args.package))

def required_flash_attn():
    return Version.required_flash_attn()
