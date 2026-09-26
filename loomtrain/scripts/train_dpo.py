from loomtrain.tasks import (
    DPOModule,
    DPODataModule,

)

from loomtrain.core.loops import SupervisedLoop
from loomtrain import core as lt

def train():
    lt.add_extra_arguments_by(add_dpo_args)
    args = lt.args()

    loop = SupervisedLoop(
        data_module = DPODataModule(
            strategy = lt.data_strategy.BestFitPackingStrategy(),
            dataset_dicts = [
                lt.data.DatasetDict(pth, train_count = tc, val_count = vc, 
                                    max_length = args.max_data_length,
                                    prompt_key = args.prompt_key,
                                    chosen_key = args.chosen_key,
                                    rejected_key = args.rejected_key,
                                    num_rejects = args.num_rejects,
                                    sample_rejects = lambda x, num_rejects: x[ :num_rejects] if isinstance(x, list) else x) \
                    for pth, tc, vc in zip(args.dataset_paths, args.train_samples, args.val_samples)
            ]
        ),
        train_module = DPOModule(
            strategy = lt.train_strategy.DeepspeedStrategy(),
            optim_config = lt.OptimConfig(lr = args.lr, warmup_ratio = args.warmup_ratio)
        )
    )

    return loop.fit()


def add_dpo_args(parser: "lt.ArgumentParser"):
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
        "--chosen-key", type = str, default = "chosen"
    )
    group.add_argument(
        "--rejected-key", type = str, default = "rejected"
    )
    group.add_argument(
        "--num-rejects", type = int, default = 1
    )
    group.add_argument(
        "--beta", type = float, default = 2.
    )
    group.add_argument(
        "--label-smoothing", type = float, default = 0.
    )
    group.add_argument(
        "--nll-loss-weight", type = float, default = 0.,
    )
    group.add_argument(
        "--ipo", action = "store_true"
    )

if __name__ == "__main__":
    train()
