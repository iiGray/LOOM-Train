import argparse, json, ast
from typing import Callable

_ARGUMENT_PARSER_ = None
_ARGUMENTS_ = None
_EXTRA_ARGUMENTS_: "list[Callable]" = []


def parse_tuple(s):
    try:
        ret = ast.literal_eval(s)
        if not isinstance(ret, tuple):
            raise ValueError
        return ret
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Must be legal: '(1, 2)'")


def add_extra_arguments_by(add_func: "Callable | list[Callable]"):
    '''
    add_func must be like this:
    >>> def add_extra_args(parser):
    >>>     group = parser.add_group("xxx")
    >>>     group.add_arguments(xxx)
    
    '''

    global _EXTRA_ARGUMENTS_, _ARGUMENTS_
    assert _ARGUMENTS_ is None, "Please add extra arguments before calling args()"
    if isinstance(add_func, list):
        _EXTRA_ARGUMENTS_.extend(add_func)
    else:
        _EXTRA_ARGUMENTS_.append(add_func)


def args() -> "argparse.Namespace":
    global _ARGUMENTS_, _EXTRA_ARGUMENTS_
    if _ARGUMENTS_ is None:
        _ARGUMENTS_ = _init_args([
            _add_data_config_arguments,
            _add_parallel_config_arguments,
            _add_checkpoint_config_arguments,
            _add_visualization_config_arguments,
            _add_base_strategy_arguments,
            _add_deepspeed_strategy_arguments,
            *_EXTRA_ARGUMENTS_
        ])
    post_set_args()
    return _ARGUMENTS_

def get_arg(name):
    global _ARGUMENTS_
    return getattr(_ARGUMENTS_, name, None)

def set_arg(name, value):
    global _ARGUMENTS_
    setattr(_ARGUMENTS_, name, value)

def post_set_args():
    
    global _ARGUMENTS_
    assert _ARGUMENTS_ is not None
    if not get_arg("tokenizer_path"):
        set_arg(
            "tokenizer_path",
            get_arg("model_path")
        )


def _init_parser():
    global _ARGUMENT_PARSER_
    _ARGUMENT_PARSER_ = argparse.ArgumentParser(description='LOOM-Train Arguments', 
                                                conflict_handler='resolve',
                                                allow_abbrev=False)
    return _ARGUMENT_PARSER_
def get_parser() -> "argparse.ArgumentParser":
    global _ARGUMENT_PARSER_
    return _ARGUMENT_PARSER_


def _init_args(add_arguments: "list[Callable]" = None):
    global _ARGUMENTS_
    if _ARGUMENTS_ is None:
        parser = get_parser()
        if parser is None: 
            parser = _init_parser()
        if add_arguments is not None:
            for func in add_arguments:
                func(parser)
        _ARGUMENTS_ = parser.parse_args()
    return _ARGUMENTS_

def add_argument(*args, **kwargs):
    parser = get_parser()
    if parser is None: 
        parser = _init_parser()
    parser.add_argument(*args, **kwargs)


def _add_data_config_arguments(parser: "argparse.ArgumentParser"):
    group = parser.add_argument_group(title = "Data Config Arguments")
    group.add_argument(
        "--data-cache-dir", type = str , default = ".cache/loomtrain/processed_datasets",
        help = "The cache directory for processed datasets"
    )
    group.add_argument(
        "--val-split", type = str , default = 'val',
        help = "The split name of validation dataset"
    )
    group.add_argument(
        "--collate-type", type = str, choices = ['packing', 'padding'] , default = 'packing',
        help = "The collate type of DataLoader, if packing, datas will be concatenated from head to tail"
    )
    group.add_argument(
        "--packing-length", type = int , default = 0,
        help = 'works only if --collate-type is packing'
    )
    parser.add_argument(
        "--max-data-length", type = int, default = 128000
    )

    group.add_argument(
        "--global-batch-size", type = int, default = 64,
        help = "The global batch size for training"
    )
    group.add_argument(
        "--micro-batch-size", type = int, default = 64,
        help = "The micro batch size for training"
    )
    group.add_argument(
        "--val-batch-size", type = int, default = 1,
        help = "The batch size for validation"
    )
    group.add_argument(
        "--val-interval", type = int, default = 20,
        help = "The interval (in steps) to run validation"
    )
    group.add_argument(
        "--num-epochs", type = int, default = 1,
        help = "The number of epochs to train"
    )
    group.add_argument(
        "--pin-memory", type = bool, default = True,
        help = "Whether to pin memory in DataLoader"
    )
    group.add_argument(
        "--shuffle", type = bool, default = True,
        help = "Whether to shuffle the dataset each epoch"
    )
    group.add_argument(
        "--drop-last", type = bool, default = False,
        help = "Whether to drop the last incomplete batch"
    )
    group.add_argument(
        "--drop-exceed", type = bool, default = False,
        help = "Whether to drop samples that exceed the max length"
    )


def _add_parallel_config_arguments(parser: "argparse.ArgumentParser"):
    group = parser.add_argument_group(title = "Parallel Config Arguments")
    parser.add_argument(
        "--local_rank", type = int , default = -1,
        help = "No need to pass this argument manually"
    )
    group.add_argument(
        "--nnodes", type = int, default = 1,
        help = "The number of nodes for distributed training"
    )
    group.add_argument(
        "--devices-per-node", type = int, default = 8,
        help = "The number of devices per node"
    )

    group.add_argument(
        "--cp-size", type = int, default = 8,
        help = "The size of context parallelism"
    )
    group.add_argument(
        "--cp-type", type = str, default = "ring",
        help = "The type of context parallelism"
    )
    group.add_argument(
        "--cp-args", type = json.loads, default = '{"head_stride": 1}',
        help = "The args of context parallelism in json format"
    )

def _add_checkpoint_config_arguments(parser: "argparse.ArgumentParser"):
    group = parser.add_argument_group(title = "Checkpoint Config Arguments")

    group.add_argument(
        "--save-dir", type = str, required = True,
        help = "The directory to save checkpoints"
    )

    group.add_argument(
        "--do-resume", type = bool, default = True,
        help = "Whether to resume training from the latest checkpoint"
    )
    group.add_argument(
        "--ckpt-interval", type = int, default = 20,
        help = "The interval (in steps) to save checkpoints"
    )
    group.add_argument(
        "--weight-interval", type = int, default = 20,
        help = "The interval (in steps) to save model weights"
    )
    group.add_argument(
        "--visualization-interval", type = int, default = 1,
        help = "The interval (in steps) to save visualizations"
    )
    group.add_argument(
        "--max-ckpts", type = int, default = 2,
        help = "The maximum number of checkpoints to keep"
    )   
    group.add_argument(
        "--max-ckpts-GB", type = int, default = 1024,
        help = "The maximum size (in GB) of checkpoints to keep"
    )


def _add_visualization_config_arguments(parser: "argparse.ArgumentParser"):
    group = parser.add_argument_group(title = "Visulization Config Arguments")

    group.add_argument(
        "--logtype", type = str, choices = ['tensorboard', 'wandb', ''], default = 'tensorboard',
        help = "The type of visualization tool"
    )
    group.add_argument(
        "--wandb-api", type = str, default = None,
        help = "The wandb API key"
    )
    group.add_argument(
        "--wandb-entity", type = str, default = None,
        help = "The wandb entity (team) name"
    )
    group.add_argument(
        "--wandb-project", type = str, default = 'loom-train',
        help = "The wandb project name"
    )
    group.add_argument(
        "--wandb-group", type = str, default = None,
        help = "The wandb group name"
    )
    group.add_argument(
        "--wandb-name", type = str, default = None,
        help = "The wandb run name"
    )
    group.add_argument(
        "--terminal-logtype", type = str, choices = ["tqdm", "rich"], default = "rich",
        help = "Whether to enable training progress bar"
    )
    group.add_argument(
        "--enable-micro-bar", type = bool, default = False,
        help = "Whether to enable micro batch progress bar"
    )


def _add_base_strategy_arguments(parser: "argparse.ArgumentParser"):
    group = parser.add_argument_group(title = "Base Strategy Arguments")

    group.add_argument(
        "--full-determinism", type = bool, default = False,
        help = "Whether to enable full determinism"
    )
    group.add_argument(
        "--seed", type = int, default = 42,
        help = "The random seed"
    )

    group.add_argument(
        "--lr", type = float, default = 1e-4,
        help = "The learning rate"
    )
    group.add_argument(
        "--warmup-ratio", type = float, default = 0.03,
        help = "The warmup ratio"
    )

def _add_deepspeed_strategy_arguments(parser: "argparse.ArgumentParser"):
    group = parser.add_argument_group(title = "Deepspeed Strategy Arguments")
    
    group.add_argument(
        "--deepspeed-init-timeout", type = int , default = 60,
        help = "The timeout (in minutes) for deepspeed initialization"
    )
    group.add_argument(
        "--zero-stage", type = int, default = 3,
        help = "The ZeRO optimization stage"
    )
    group.add_argument(
        "--enable-bf16", type = bool, default = True,
        help = "Whether to enable bf16 training"
    )
    group.add_argument(
        "--offload", type = bool, default = False,
        help = "Whether to offload parameters to CPU"
    )
    group.add_argument(
        "--adam-offload", type = bool, default = True,
        help = "Whether to offload Adam optimizer states to CPU"
    )
    group.add_argument(
        "--ref-offload", type = bool, default = False,
        help = "Whether to offload reference model to CPU"
    )
    group.add_argument(
        "--grad-clip", type = float, default = 1.0,
        help = "The gradient clipping value"
    )
    group.add_argument(
        "--zpg", type = int , default = 1,
    )
    group.add_argument(
        "--grad-accum-dtype", type = str , default = None,
        choices = [None, 'fp16', 'bf16', 'fp32'],
        help = "The data type for gradient accumulation"
    )
    group.add_argument(
        "--overlap-comm", type = bool , default = False,
        help = "Whether to overlap communication with computation"
    )
    group.add_argument(
        "--load-universal", type = bool , default = False,
        help = "Whether to load universal model"
    )

