from linktrain.core.loop import FitLoop


class CustomizedLoop(FitLoop):
    """A customizable training loop for training a model with optional support for reinforcement learning."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError
    
    def fit(self):
        raise NotImplementedError