import math
from typing import Iterator, Optional, TypeVar
from torch.utils.data.sampler import Sampler

_T_co = TypeVar("_T_co", covariant=True)

class StatefulSampler(Sampler[_T_co]):
    def set_state(self, current_epoch: int, consumed_samples: int = 0):
        '''
        If you don't need this function to set state (since sampler state is useless),
        then you don't have to impelment this function
        '''
        return