import datetime
from tqdm import tqdm
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Group
from loomtrain.core.state import CheckpointConfig
from loomtrain.core.strategy import TrainStrategy, DataStrategy
from loomtrain.core.module import Module
from loomtrain.core.datamodule import DataModule
from loomtrain.core.visualization import NoneVisualization, VisualizationModule
from loomtrain.core.parallel import parallel_state as parallel
from loomtrain.core.arguments import args

def _generate_table(log_dicts):
    table = Table(show_header=False, padding=(0, 1))
    table.add_column("Key", justify="left")
    table.add_column("Value", justify="left", no_wrap=True)
    for k, v in log_dicts.items():
        table.add_row(f"{k}:", str(v))
    return table

COMPLETE_COLOR = "bright_cyan"
REMAINING_COLOR = "yellow"

class ColoredElapsedColumn(TimeElapsedColumn):
    def render(self, task) -> Text:
        elapsed_text = super().render(task)
        elapsed_text.style = COMPLETE_COLOR
        return elapsed_text

class ColoredRemainingColumn(TimeRemainingColumn):
    def render(self, task) -> Text:
        if task.completed == 0 or task.total is None:
            return Text("-:--:--", style = REMAINING_COLOR)
        rich_text = super().render(task)
        if  "-" not in str(rich_text):
            rich_text.style = REMAINING_COLOR
            return rich_text
        
        remaining_steps = task.total - task.completed
        seconds_remaining = (task.elapsed / task.completed) * remaining_steps
        eta_string = str(datetime.timedelta(seconds=int(seconds_remaining)))
        return Text(eta_string, style = REMAINING_COLOR)

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
            visualization_interval = args().visualization_interval,
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

    if args().terminal_logtype == "tqdm":
        progress_bar = tqdm(range(0, datamodule.total_train_steps), 
                            desc = f"Training epoch: {datamodule.training_epoch + 1}/{args().num_epochs}", 
                            initial = datamodule.consumed_steps,
                            position = 0, dynamic_ncols = True, 
                            disable = parallel.get_rank() != 0)
    elif parallel.get_rank() == 0:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                style = REMAINING_COLOR,
                complete_style = COMPLETE_COLOR,
                finished_style = "green"
            ),
            TextColumn("{task.completed:.0f}/{task.total:.0f}"),        
            "|",
            ColoredElapsedColumn(),
            ColoredRemainingColumn(),
            disable = parallel.get_rank() != 0
        )
        training_task = progress.add_task("Total Training Steps:", 
                                          start = False,
                                          completed = datamodule.consumed_steps,
                                          total = datamodule.total_train_steps)
        live = Live(Group(progress), refresh_per_second=10)
        live.start()

    ################################
    ############ Training Loop ############
    try:
        if parallel.get_rank() == 0 and args().terminal_logtype == "rich":
            progress.start_task(training_task)
        while not datamodule.exhausted:
            batches = datamodule._update_()
            logs_dict = dict()
            state_dict = module._update_(batches)

            for k, v in state_dict.items():
                logs_dict[f"train/{k}"] = v

            if args().terminal_logtype == "tqdm":
                progress_bar.set_description(f"Training epoch: {datamodule.training_epoch + 1}/{args().num_epochs}")
                progress_bar.set_postfix(logs_dict)
                progress_bar.update(1)
            
            state_dict = module._validate(datamodule)

            if args().terminal_logtype == "rich" and parallel.get_rank() == 0:
                progress.update(training_task, advance = 1)
                live.update(
                    Group(
                        progress,
                        _generate_table({"Training Epoch:" : f"{datamodule.training_epoch + 1}/{args().num_epochs}",
                                         "Consumed Samples:" : str(datamodule.train_data_iter.consumed_samples), 
                                         ** logs_dict})
                    )
                )

            for k, v in state_dict.items():
                logs_dict[f"val/{k}"] = v
            
            vismodule._update_(logs_dict)

            if parallel.get_rank() == 0:
                print(f"LogsDict: {logs_dict}")

            datamodule._save_ckpt(checkpoint_config, inplace = False)
            module._save_ckpt(checkpoint_config, inplace = False, update_tag = True)
            vismodule._save_ckpt(checkpoint_config, inplace = True)

            module._save_module(checkpoint_config) # save module weights for inference
    finally:
        if args().terminal_logtype == "tqdm":
            progress_bar.close()
        elif parallel.get_rank() == 0:
            live.stop()            
        vismodule.release()