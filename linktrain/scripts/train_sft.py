from linktrain.tasks import (
    SFTModule,
    SFTDataModule,
)
from linktrain.core.loops import SupervisedLoop
from linktrain import core as lt


def train():
    lt.add_extra_arguments_by(add_sft_args)
    args = lt.args()

    loop = SupervisedLoop(
        data_module = SFTDataModule(
        strategy = lt.data_strategy.SortPackingStrategy(),
        dataset_dicts = [
            lt.data.DatasetDict(pth, train_count = tc, val_count = vc, 
                                max_length = args.max_data_length,
                                prompt_key = args.prompt_key,
                                response_key = args.response_key) \
                for pth, tc, vc in zip(args.dataset_paths, args.train_samples, args.val_samples)
        ]),
        train_module = SFTModule(
            strategy = lt.train_strategy.DeepspeedStrategy(),
            optim_config = lt.OptimConfig(lr = args.lr, warmup_ratio = args.warmup_ratio)
            ),

    )
    

    return loop.fit()


def add_sft_args(parser: "lt.ArgumentParser"):
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
    train()
