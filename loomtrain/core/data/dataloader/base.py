from loomtrain.core.metas import AttrDict

class DataIterStateDict(AttrDict):
    def __init__(self, current_epoch: "int" = 0, consumed_samples: "int" = 0, ** kwargs):
        super().__init__(** kwargs)

        self.current_epoch = current_epoch
        self.consumed_samples = consumed_samples

class StatefulDataIterMixin:
    @property
    def current_epoch(self):
        '''
        The epoch have been training.
        '''
        if not hasattr(self, "_current_epoch_"):
             self._current_epoch_ = 0
        return self._current_epoch_
    @current_epoch.setter
    def current_epoch(self, e: "int"):
        self._current_epoch_  = e
    
    @property
    def consumed_samples(self):
        '''
        The samples have been trained in the current epoch 
        '''
        if not hasattr(self, "_consumed_samples_"):
            self._consuemd_samples_ = 0
        return self._consuemd_samples_

    @consumed_samples.setter
    def consumed_samples(self, s: "int"):
        self._consuemd_samples_ = s

    def set_state(self, current_epoch: "int" = 0, consumed_samples: "int" = 0):
        self.current_epoch = current_epoch
        self.consumed_samples = consumed_samples

    def get_state(self) -> "DataIterStateDict":
        return DataIterStateDict(
            current_epoch = self.current_epoch,
            consumed_samples = self.consumed_samples
        )