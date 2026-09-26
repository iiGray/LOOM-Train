from loomtrain.tasks import (
    PTModule,
    PTDataModule,

)
from loomtrain.core.loops import SupervisedLoop
from loomtrain import core as lt
def train():
    lt.add_extra_arguments_by(add_pretrain_args)
    args = lt.args()


    loop = SupervisedLoop(
        data_module = PTDataModule(
            strategy = lt.data_strategy.SortPackingStrategy(),
            dataset_dicts = [
                lt.data.DatasetDict(pth, train_count = tc, val_count = vc,
                                    max_length = args.max_data_length,
                                    text_key = args.text_key) \
                    for pth, tc, vc in zip(args.dataset_paths, args.train_samples, args.val_samples)
            ]
        ),
        train_module = PTModule(
            strategy = lt.train_strategy.DeepspeedStrategy(),
            optim_config = lt.OptimConfig(lr = args.lr, warmup_ratio = args.warmup_ratio)
        ),
    )

    return loop.fit()

def add_pretrain_args(parser: "lt.ArgumentParser"):
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
        "--text-key", type = str, default = "text"
    )


if __name__ == "__main__":
    train()
