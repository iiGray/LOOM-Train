from loomtrain.core.metas import AttrDict

class DataLoaderStateDict(AttrDict):
    def __init__(self, current_epoch: "int" = 0, consumed_samples: "int" = 0, consumed_indices: "int" = 0, ** kwargs):
        super().__init__(** kwargs)

        self.current_epoch = current_epoch
        self.consumed_samples = consumed_samples
        self.consumed_indices = consumed_indices

class StatefulDataLoaderMixin:
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
            self._consumed_samples_ = 0
        return self._consumed_samples_

    @consumed_samples.setter
    def consumed_samples(self, s: "int"): 
        self._consumed_samples_ = s


    @property
    def consumed_indices(self):
        if not hasattr(self, "_consumed_indices_"):
            self._consumed_indices_ = 0
        return self._consumed_indices_
    @consumed_indices.setter
    def consumed_indices(self, i: "int"):
        self._consumed_indices_ = i

    def set_state(self, current_epoch: "int" = 0, consumed_samples: "int" = 0, consumed_indices: "int" = 0):
        self.current_epoch = current_epoch
        self.consumed_samples = consumed_samples
        self.consumed_indices = consumed_indices

    def get_state(self) -> "DataLoaderStateDict":
        return DataLoaderStateDict(
            current_epoch = self.current_epoch,
            consumed_samples = self.consumed_samples,
            consumed_indices = self.consumed_indices
        )