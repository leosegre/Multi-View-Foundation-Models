import os
from .models import *
import torch
import numpy as np

DEBUG_MODE = False
RANK = int(os.environ.get('RANK', default = 0))
GLOBAL_STEP = 0
STEP_SIZE = 1
LOCAL_RANK = -1

def torch_custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {torch_original_repr(self)}'

torch_original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = torch_custom_repr