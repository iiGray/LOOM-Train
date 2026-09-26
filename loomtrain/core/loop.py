import datetime
from tqdm import tqdm
import ray
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Group
from loomtrain.core.state import CheckpointConfig
from loomtrain.core.strategy import TrainStrategy, DataStrategy
from loomtrain.core.visualization import NoneVisualization, VisualizationModule, Accum
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

REMAINING_START = 0

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
        seconds_remaining = (task.elapsed / (task.completed - REMAINING_START)) * remaining_steps
        eta_string = str(datetime.timedelta(seconds=int(seconds_remaining)))
        return Text(eta_string, style = REMAINING_COLOR)

class FitLoop:
    def __init__(self, *args, **kwargs):
        if not ray.is_initialized():
            ray.init()

    def _init_terminal_log(self, total_train_steps: int, training_epoch: int, consumed_steps: int):
        if args().terminal_logtype == "tqdm":
            self._progress_bar = tqdm(range(0, total_train_steps), 
                                desc = f"Training epoch: {training_epoch}", 
                                initial = consumed_steps,
                                position = 0, dynamic_ncols = True)
        else:
            self._progress = Progress(
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
                ColoredRemainingColumn()
            )
            global REMAINING_START
            REMAINING_START = consumed_steps - 1
            self._training_task = self._progress.add_task(
                "Total Training Steps:", 
                start = False,
                completed = consumed_steps,
                total = total_train_steps
            )
            self._live = Live(Group(self._progress), refresh_per_second=10)
            self._live.start()
        

        if args().terminal_logtype == "rich":
            self._progress.start_task(self._training_task)


    def _update_terminal_log(self, training_epoch: int, consumed_samples: int, calculated_logs_dict: dict):

        if args().terminal_logtype == "tqdm":
            self._progress_bar.set_description(f"Training epoch: {training_epoch}")
            self._progress_bar.set_postfix(calculated_logs_dict)
            self._progress_bar.update(1)

        if args().terminal_logtype == "rich":
            self._progress.update(self._training_task, advance = 1)
            self._live.update(
                Group(
                    self._progress,
                    _generate_table({"Training Epoch" : training_epoch,
                                    "Consumed Samples" : consumed_samples, 
                                    ** calculated_logs_dict})
                )
            )        

    def _close_terminal_log(self):
        if args().terminal_logtype == "tqdm":
            self._progress_bar.close()
        else:
            self._live.stop()


    def fit(self, vis_module: "VisualizationModule", checkpoint_config: "CheckpointConfig" = None):
        raise NotImplementedError
