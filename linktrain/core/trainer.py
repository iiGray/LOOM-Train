from linktrain.core.state import CheckpointConfig
from linktrain.core.modules.vis_module import NoneVisualization, VisualizationModule
from linktrain.core.arguments import args
from linktrain.core.loop import FitLoop


def fit(loop: FitLoop, vis_module: VisualizationModule = None, checkpoint_config: CheckpointConfig = None):
    if vis_module is None:
        vis_module = NoneVisualization()
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

    loop.fit(vis_module = vis_module, checkpoint_config = checkpoint_config)