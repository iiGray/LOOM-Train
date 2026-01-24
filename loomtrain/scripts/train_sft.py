from loomtrain.tasks import (
    SFTModule,
    SFTDataModule,

)
from loomtrain import core as lt
def train():
    args = lt.args()

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