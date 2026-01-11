import argparse, json

from loomtrain.core import (
    LoomSFTModule,
    LoomSFTData,
    LoomDataDict,
    LoomTrainer,
    DeepspeedStrategy,
    DeepspeedConfig,
    SortPackingStrategy,
    DataConfig,
    CheckpointConfig,
    VisualizationModule,
    parallel

)


def train(args):

    data_config = DataConfig(
        collate_type = 'packing',
        packing_length = args.packing_length,
        
        train_batch_size = args.global_batch_size,
        micro_batch_size = args.micro_batch_size,
        val_batch_size = args.val_batch_size,
        val_interval = args.val_interval,
        batch_size = 10,
        num_epochs = 1
    )


    parallel_config = parallel.ParallelConfig(
        nnodes = 1,
        devices_per_node = 8,
        cp = args.cp_size,
        cp_type = args.cp_type,
        cp_args = args.cp_args
    )

    checkpoint_config = CheckpointConfig(
        save_dir = args.save_dir,
        ckpt_interval = args.ckpt_interval,
        weight_interval = args.weight_interval,
    )


    train_strategy = DeepspeedStrategy(parallel_config, data_config = data_config,
                                       deepspeed_config= DeepspeedConfig(zero_stage =3, adam_offload = True))

    data_strategy = SortPackingStrategy(parallel_config, data_config = data_config)

    trainer = LoomTrainer(train_strategy = train_strategy,
                          data_strategy = data_strategy)
    
    module = LoomSFTModule(args.model_path)
    
    datamodule = LoomSFTData([
        LoomDataDict(data_path = pth, tokenizer_path = args.model_path, train_count = tc, val_count = vc, prompt_key = args.prompt_key, response_key = args.response_key) for pth, tc, vc in zip(
            args.dataset_paths, args.train_samples, args.val_samples
        )
    ], max_length = args.max_length)


    vismodule = VisualizationModule(logtype = 'tensorboard')


    trainer.fit(
        module = module,
        datamodule = datamodule,
        vismodule = vismodule,
        checkpoint_config = checkpoint_config
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank", type = int , default = -1
    )
    
    parser.add_argument(
        "--global-batch-size", type = int , default = 64
    )
    parser.add_argument(
        "--micro-batch-size", type = int , default = 1
    )
    parser.add_argument(
        "--val-batch-size", type = int, default = 1,
    )
    parser.add_argument(
        "--cp-size", type = int, default = 8
    )
    parser.add_argument(
        "--cp-type", type = str, default = "ring"
    )
    parser.add_argument(
        "--cp-args", type = json.loads, default = '{"head_stride": 1}'
    )
    parser.add_argument(
        "--max-length", type = int, default = 128000
    )
    parser.add_argument(
        "--packing-length", type = int, default = None
    )
    parser.add_argument(
        "--val-interval", type = int, default = 20,
    )
    parser.add_argument(
        "--save-dir", type = str, required = True,
    )
    parser.add_argument(
        "--ckpt-interval", type = int, default = 20,
    )
    parser.add_argument(
        "--weight-interval", type = int, default = 20,
    )
    parser.add_argument(
        "--model-path", type = str, required = True
    )
    parser.add_argument(
        "--dataset-paths", type = str, nargs = "+", required = True
    )
    parser.add_argument(
        "--train-samples", type = int, nargs = "+", required = True
    )
    parser.add_argument(
        "--val-samples", type = int, nargs = "+", required = True
    )
    parser.add_argument(
        "--prompt-key", type = str, default = "prompt"
    )
    parser.add_argument(
        "--response-key", type = str, default = "response"
    )

    args = parser.parse_args()
    train(args)