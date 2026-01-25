from loomtrain.tasks import (
    SimPOModule,
    SimPODataModule,

)
from loomtrain import core as lt
def train():
    args = lt.args()

    module = SimPOModule(optim_config = lt.OptimConfig(lr = args.lr, warmup_ratio = args.warmup_ratio))
    
    datamodule = SimPODataModule(
        dataset_dicts = [
            lt.data.DatasetDict(pth, train_count = tc, val_count = vc, 
                                max_length = args.max_data_length,
                                prompt_key = args.prompt_key,
                                chosen_key = args.chosen_key,
                                rejected_key = args.rejected_key,
                                num_rejects = args.num_rejects,
                                sample_rejects = lambda x: x[:args.num_rejects] if isinstance(x, list) else x) \
                for pth, tc, vc in zip(args.dataset_paths, args.train_samples, args.val_samples)
        ])
        
    lt.fit(
        module = module,
        datamodule = datamodule,
        train_strategy = lt.train_strategy.DeepspeedStrategy(),
        data_strategy = lt.data_strategy.SortPackingStrategy(),
    )


def add_simpo_args(parser: "lt.ArgumentParser"):
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
        "--gamma", type = float, default = 0.5,
    )
    group.add_argument(
        "--label-smoothing", type = float, default = 0.
    )
    group.add_argument(
        "--simpo-loss-type", type = str, choices = ["sigmoid", "hinge"], default = "sigmoid"
    )
    group.add_argument(
        "--nll-loss-weight", type = float, default = 0.,
    )

if __name__ == "__main__":
    lt.add_extra_arguments_by(add_simpo_args)
    train()