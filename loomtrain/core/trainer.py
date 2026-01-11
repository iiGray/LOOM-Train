from loomtrain.core.state import CheckpointConfig
from loomtrain.core.strategy import TrainStrategy, DataStrategy
from loomtrain.core.module import LoomModule
from loomtrain.core.datamodule import LoomDataModule
from loomtrain.core.visualization import VisualizationModule


class LoomTrainer:
    ''''''
    def __init__(
            self, *,
            train_strategy: "TrainStrategy" = None,
            data_strategy: "DataStrategy" = None,
            accelerator = None,
            devices = None,

    ):
        self.train_strategy = train_strategy
        self.data_stretegy = data_strategy
        self.accelerator = accelerator # TODO
        self.devices = devices # TODO
        self.train_strategy.setup_distributed()

    def fit(self, 
            module: "LoomModule", 
            datamodule: "LoomDataModule",
            vismodule: "VisualizationModule",
            checkpoint_config: "CheckpointConfig"):
        
        datamodule.connect_module(module)
        datamodule.connect_strategy(self.data_stretegy)
        module.connect_datamodule(datamodule)
        module.connect_strategy(self.train_strategy)


        if checkpoint_config.do_resume:
            module._load_ckpt(checkpoint_config)
            datamodule._load_ckpt(checkpoint_config, inplace = True)
            vismodule._load_ckpt(checkpoint_config, inplace = True)

        module.train()
        datamodule.train()
    
        while not datamodule.exhausted:
            batches = datamodule._update()
            logs_dict = dict()
            state_dict = module._update(batches)

            for k, v in state_dict.items():
                logs_dict[f"train/{k}"] = v

            state_dict = module._validate(datamodule)

            for k, v in state_dict.items():
                logs_dict[f"val/{k}"] = v
            
            vismodule._update(state_dict)

            datamodule._save_ckpt(checkpoint_config, inplace = True)
            module._save_ckpt(checkpoint_config)
            vismodule._save_ckpt(checkpoint_config, inplace = True)

            module._save_module(checkpoint_config) # save module weights for inference

        vismodule.release()



