import math
from torch.autograd import Function
import torch
from torch.nn.utils import weight_norm

from ADQ_example import ADQ_Example
from BF16_matmul import BF16Matmul
from BF15_matmul import BF15Matmul
from BF14_matmul import BF14Matmul
from BF13_matmul import BF13Matmul
from BF12_matmul import BF12Matmul
from BF11_matmul import BF11Matmul
from BF10_matmul import BF10Matmul
from BF9_matmul import BF9Matmul
from BF_9back_12for_matmul import BF_9back_12for_Matmul
from BF_12back_9for_matmul import BF_12back_9for_Matmul


from NormalMatmul import NormalMatmul
from utils import Dtype, Stream, load_kernel, Dtype_size


def custom_matmul(input, weight, compute_flavour, epoch):
    if compute_flavour == 1:
        # NORMAL matmul
        return NormalMatmul.apply(input, weight)
    elif compute_flavour == 2:
        # BF16
        return BF16Matmul.apply(input, weight)
    elif compute_flavour == 3:
        # ADQ example with zero 1 element
        return ADQ_Example.apply(input, weight)
    elif compute_flavour == 4:
        # BF15
        return BF15Matmul.apply(input, weight)
    elif compute_flavour == 5:
        # BF14
        return BF14Matmul.apply(input, weight)
    elif compute_flavour == 6:
        # BF13
        return BF13Matmul.apply(input, weight)
    elif compute_flavour == 7:
        # BF12
        return BF12Matmul.apply(input, weight)
    elif compute_flavour == 8:
        # BF11
        return BF11Matmul.apply(input, weight)
    elif compute_flavour == 9:
        # BF10
        return BF10Matmul.apply(input, weight)
    elif compute_flavour == 10:
        # BF9
        return BF9Matmul.apply(input, weight)
    elif compute_flavour == 11:
        # 9back-12for
        return BF_9back_12for_Matmul.apply(input, weight)
    elif compute_flavour == 12:
        # 12back-9for
        return BF_12back_9for_Matmul.apply(input, weight)
    elif compute_flavour == 13:
        # dynamic compute_flavour- 30 epochs at start 16BF and the rest 9BF
        if epoch < 30 :
          print(f'Custom_matmul - epoch <30, 16BF - epoch = {epoch}')
          return BF16Matmul.apply(input, weight)
        else :
          print(f'Custom_matmul - epoch >=30, 9BF - epoch = {epoch}')
          return BF9Matmul.apply(input, weight)
    else:
        raise NotImplementedError




