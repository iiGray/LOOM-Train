from accelerate.utils import reduce
from email.policy import strict
from typing import final
from linktrain.core.arguments import args
from linktrain.core.loop import *
from linktrain.core.modules import (
    DataModule, TrainModule
)
from linktrain.core.modules.vis_module import NoneVisualization, VisualizationModule, Accum

from linktrain.core.distributed import RayRole, RayResourcePoolManager

from linktrain.core.state import CheckpointConfig

class SupervisedLoop(FitLoop):
    def __init__(
        self, 
        data_module: "DataModule",
        train_module: "TrainModule",
    
    ):
        super().__init__()
        with RayResourcePoolManager(strict_mode = True) as manager, manager.create_pool(
            nprocs = 8,
            ncpus_per_proc = 8,
            ngpus_per_proc = 1,
            roles = [RayRole.DATA, RayRole.ACTOR]
        ) as pool:
            self.data_module = pool.init_worker(data_module)
            self.train_module = pool.init_worker(train_module)



    def fit(self, vis_module: "VisualizationModule" = None, checkpoint_config: "CheckpointConfig" = None):

        if checkpoint_config is None:
            checkpoint_config = CheckpointConfig(
                save_dir = args().save_dir,
                do_resume = args().do_resume,
                ckpt_interval = args().ckpt_interval,
                weight_interval = args().weight_interval,
                visualization_interval = args().visualization_interval,
                max_ckpts = args().max_ckpts,
                max_ckpts_GB = args().max_ckpts_GB
            )
        if (vis_module is None) and args().logtype:
            vis_module = VisualizationModule(
                logtype = args().logtype,
                wandb_api = args().wandb_api,
                wandb_entity = args().wandb_entity,
                wandb_project = args().wandb_project,
                wandb_group = args().wandb_group,
                wandb_name = args().wandb_name
            )
        if vis_module is None: vis_module = NoneVisualization()


        logs_dict: "dict[str, Accum]" = None

        self.data_module.prepare_strategy()
        if checkpoint_config.do_resume:
            self.data_module.dist(sync_output = True)._load_ckpt(checkpoint_config)

        self.train_module.dist(reduce_input = True).config_optim_group(
            total_train_steps = self.data_module.dist().total_train_steps
        )
        self.train_module.prepare_strategy()
        if checkpoint_config.do_resume:
            self.train_module.dist(sync_output = True)._load_ckpt(checkpoint_config)

        logs_dict = vis_module._load_ckpt(checkpoint_config, inplace = True)
        
        total_train_steps  = self.data_module.dist(get_first = True).total_train_steps
        training_epoch = f"{self.data_module.dist(get_first = True).training_epoch + 1}/{args().num_epochs}"
        consumed_steps = self.data_module.dist(get_first = True).consumed_steps
        self._init_terminal_log(total_train_steps, training_epoch, consumed_steps)

        try:
            while not self.data_module.dist(gather = any).exhausted:
                batches = self.data_module.dist()._update_()
                logs_dict = dict() if logs_dict is None \
                    else {k: v for k, v in logs_dict.items() if isinstance(v, Accum) and v.is_global}
                state_dict = self.train_module.dist(
                    reduce_input = True,
                    get_first = True)._update_(batches)

                for k, v in state_dict.items():
                    if f"train/{k}" in logs_dict: logs_dict[f"train/{k}"] += v
                    else: logs_dict[f"train/{k}"] = v

                finished = self.data_module.dist(gather = any).exhausted

                if args().do_validate and self.data_module.dist(get_first = True).is_validating_step:
                    self.train_module.dist().eval()
                    eval_batches = self.data_module.dist().get_eval_batch()
                    state_dict = self.train_module.dist(reduce_input = True, get_first = True)._validate(eval_batches, self.data_module.dist().len_val_data_iter)
                    self.train_module.dist().train()

                    for k, v in state_dict.items():
                        logs_dict[f"val/{k}"] = v

                calculated_logs_dict = {k: v.get_value() if isinstance(v, Accum) else v for k, v in logs_dict.items()}
                training_epoch = f"{self.data_module.dist(gather = lambda x: x[0]).training_epoch + 1}/{args().num_epochs}"
                consumed_samples = str(self.data_module.dist(get_first = True).train_consumed_samples)

                self._update_terminal_log(training_epoch, consumed_samples, calculated_logs_dict)

                vis_module._update_(logs_dict)

                self.data_module.dist()._save_ckpt(checkpoint_config, inplace = False, finished = finished)
                self.train_module.dist()._save_ckpt(checkpoint_config, inplace = False, update_tag = True, finished = finished)
                vis_module._save_ckpt(checkpoint_config, inplace = True, save_interval = checkpoint_config.visualization_interval, finished = finished)

                self.train_module.dist()._save_module(checkpoint_config, finished = finished) # save 

            self.train_module.ready()
            self.data_module.ready()

        finally:
            vis_module.release()
            self._close_terminal_log()
            ray.shutdown()

