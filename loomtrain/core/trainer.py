from tqdm import tqdm
from loomtrain.core.state import CheckpointConfig
from loomtrain.core.strategy import TrainStrategy, DataStrategy
from loomtrain.core.module import Module
from loomtrain.core.datamodule import DataModule
from loomtrain.core.visualization import NoneVisualization, VisualizationModule
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.arguments import args


def fit(module: "Module",
        datamodule: "DataModule",
        train_strategy: "TrainStrategy" = None,
        data_strategy: "DataStrategy" = None,
        vismodule: "VisualizationModule" = None,
        checkpoint_config: "CheckpointConfig" = None,
        enable_distributed: bool = True):
    
    if train_strategy is None:
        train_strategy = TrainStrategy(
            global_batch_size = args().global_batch_size,
            micro_batch_size = args().micro_batch_size,
            val_batch_size = args().val_batch_size,
            cp_size = args().cp_size,
            cp_type = args().cp_type,
            cp_args = args().cp_args,
            val_interval = args().val_interval
        )
    if data_strategy is None:
        data_strategy = DataStrategy(
            packing_length = args().packing_length
        )

    if checkpoint_config is None:
        checkpoint_config = CheckpointConfig(
            save_dir = args().save_dir,
            ckpt_interval = args().ckpt_interval,
            weight_interval = args().weight_interval,
            visulization_interval = args().visulization_interval,
            max_ckpts = args().max_ckpts,
            max_ckpts_GB = args().max_ckpts_GB
        )
    if (vismodule is None) and args().logtype:
        vismodule = VisualizationModule(
            logtype = args().logtype,
            wandb_api = args().wandb_api,
            wandb_entity = args().wandb_entity,
            wandb_project = args().wandb_project,
            wandb_group = args().wandb_group,
            wandb_name = args().wandb_name
        )
    if vismodule is None: vismodule = NoneVisualization()
        
    if enable_distributed:
        train_strategy.setup_distributed()

    module._initialize()
    datamodule._initialize()

    if data_strategy is not None:
        datamodule._connect_strategy(data_strategy)
    datamodule._connect_module(module)
    if train_strategy is not None:
        module._connect_strategy(train_strategy)
        train_strategy._connect_datamodule(datamodule)
    module._connect_datamodule(datamodule)
    
    module.config_module()

    module.zero_grad()



    if checkpoint_config.do_resume:
        module._load_ckpt(checkpoint_config)
        datamodule._load_ckpt(checkpoint_config, inplace = True)
        vismodule._load_ckpt(checkpoint_config, inplace = True)


    module.train()
    datamodule.train()

    progress_bar = tqdm(range(0, datamodule.total_train_steps), 
                        desc = f"Training epoch: {datamodule.training_epoch + 1}/{args().num_epochs}", 
                        initial = datamodule.consumed_steps,
                        disable = parallel.get_rank() != 0)

    while not datamodule.exhausted:
        batches = datamodule._update_()
        logs_dict = dict()
        state_dict = module._update_(batches)

        for k, v in state_dict.items():
            logs_dict[f"train/{k}"] = v

        progress_bar.set_description(f"Training epoch: {datamodule.training_epoch + 1}/{args().num_epochs}")
        progress_bar.set_postfix(logs_dict)
        progress_bar.update(1)
        state_dict = module._validate(datamodule)

        for k, v in state_dict.items():
            logs_dict[f"val/{k}"] = v
        
        vismodule._update_(state_dict)

        datamodule._save_ckpt(checkpoint_config, inplace = False)
        module._save_ckpt(checkpoint_config, inplace = False, update_tag = True)
        vismodule._save_ckpt(checkpoint_config, inplace = True)

        module._save_module(checkpoint_config) # save module weights for inference

    progress_bar.close()            
    vismodule.release()
