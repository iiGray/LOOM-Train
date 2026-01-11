import argparse, os
from loomtrain.utils.common.iotools import dirname

def add_saving_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title = "Saving Config")
    group.add_argument(
        "--data-cache-dir", type = str, default = f"{dirname(dirname(dirname(os.path.abspath(__file__))))}/examples/datasetfiles",
        help = "Data cache directory."
    )
    group.add_argument(
        "--save-dir", type = str, default = f"{dirname(dirname(dirname(os.path.abspath(__file__))))}/examples/",
        help = "The ckpt and weights saving directory."
    )
    group.add_argument(
        "--save-name", type = str, default = None, required = True,
        help = "The ckpt and weights saving name."
    )
    return parser


def add_model_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title = "Model Config")
    group.add_argument(
        "--model-path", type = str, required = True,
        help = "Huggingface Model Name"
    )
    return parser    


def add_data_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title = "Data Config")
    group.add_argument(
        "--dataset-paths", type = str, nargs = "+", required = True,
        help = "A series formatted Dataset paths. For more detailed about the format, see xxx."
    )
    group.add_argument(
        "--sample-ratios", type = float, nargs = "+", default = None,
        help = "xxx."
    )
    group.add_argument(
        "--sample-counts", type = int, nargs = "+", required = True,
        help = "xxx."
    )
    group.add_argument(
        "--sample-counts-eval", type = int, nargs = "+", required = True,
        help = "xxx."
    )
    group.add_argument(
        "--max-data-length", type = int, default = 128000,
        help = "Upper Bound of the length of the training data, those excess this will be filterd out."
    )
    group.add_argument(
        "--max-packing-length", type = int, default = 128000,
        help = "Upper Bound of the packing length."
    )
    return parser    



def add_training_arguments(parser: argparse.ArgumentParser):

    group = parser.add_argument_group(title = "Training Config")
    
    group.add_argument(
        "--seed", type = int, default = 42
    )
    group.add_argument(
        "--do-resume", type = bool, default = True
    )
    group.add_argument(
        "--max-epochs", type = int, default = 1,
        help = "Maximum Training Epochs"
    )
    group.add_argument(
        "--learning-rate", type = float, default = 2e-6
    )

    group.add_argument(
        "--ckpt-save-interval", type = int, default = 20,
        help = "Checkpoint saving interval, used for resuming."
    )
    group.add_argument(
        "--weights-save-interval", type = int, default = 20,
        help = "Weight saving interval, used for evaluating."
    )
    group.add_argument(
        "--max-ckpts", type = int, default = 2,
        help = "Maximum Checkpoints retained. Delete the oldest one upon excessing."
    )

    group.add_argument(
        "--max-weights", type = int, default = 10,
        help = "Maximum Weights retained. Delete the oldest one upon excessing."
    )
    group.add_argument(
        "--evaluate-interval", type = int, default = 20,
        help = "Checkpoint saving interval, used for resuming."
    )
    group.add_argument(
        "--lr-warmup-ratio", type = float, default = 0.03,
        help = "Default lr scheduler is cosine annealing. This argument refers to the proportion of the entire training steps during which the LR linear growth reaches its peak."    
    )
    group.add_argument(
        "--adam-betas", type = float, nargs = 2, default = (0.9, 0.95)
    )

    group.add_argument(
        "--L2-weight-decay", type = float, default = 0.0
    )

    group = add_strategy_arguemnts_group(group)
    group = add_visualization_arguments_group(group)

    return parser





def add_strategy_arguemnts_group(group: argparse._ArgumentGroup):
    subgroup = group.add_argument_group(title = "Strategy Config")

    subgroup.add_argument(
        "--micro-batch-size", type = int, default = 1,
        help = "Micro Batch Size."
    )
    subgroup.add_argument(
        "--global-batch-size", type = int, default = 16,
        help = "Global Batch Size. Grad Accumulation will be automatically computed by `gloabl//micro`."
    )

    subgroup.add_argument(
        "--zero-stage", type = int, default = 2, choices = [2, 3],
        help = "DeepSpeed Zero Stage."
    )
    subgroup.add_argument(
        "--enable-bf16", type = bool, default = True,
        help = "Whether to use bf16. Otherwise fp16."
    )    
    subgroup.add_argument(
        "--grad-clip", type = float, default = 1.,
        help = "Grad Clipping after backward."
    )
    subgroup.add_argument(
        "--zpg", type = int, default = 1,
        help = "."
    )
    subgroup.add_argument(
        "--overlap-comm", type = bool, default = False,
    )

    subgroup = group.add_argument_group(title = "Context Parallel Config")
    subgroup.add_argument(
        "--ring-attn-size", type = int, default = 8
    )
    subgroup.add_argument(
        "--ring-head-stride", type = int, default = 4
    )
    return group


def add_visualization_arguments_group(group: argparse._ArgumentGroup):
    subgroup = group.add_argument_group(title = "Visulization Config")
    subgroup.add_argument(
        "--tensorboard-logdir", type = str, required = True
    )
    subgroup.add_argument(
        "--wandb-api", type = str, default = None
    )
    subgroup.add_argument(
        "--wandb-entity", type = str, default = None,
    )
    subgroup.add_argument(
        "--wandb-project", type = str, default = None,
    )
    subgroup.add_argument(
        "--wandb-name", type = str, default = None,
    )
    subgroup.add_argument(
        "--wandb-group", type = str, default = None,
    )
    return group

