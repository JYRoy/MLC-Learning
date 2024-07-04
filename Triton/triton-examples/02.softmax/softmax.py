import torch

import triton
import triton.language as tl
from triton.runtime import driver

torch.manual_seed(0)


def naive_softmax(x):
    x_max = x.max(dim=1)[0]
    print(x_max.shape)
    print(x_max[:, None].shape)
    z = x - x_max[:, None]
    print(z.shape)
    numerator = torch.exp(z)
    print(numerator.shape)
    denominator = numerator.sum(dim=1)
    print(denominator.shape)
    ret = numerator / denominator[:, None]
    print(ret.shape)
    return ret


x = torch.randn(1823, 781, device="cuda")
y_native = naive_softmax(x)
print(y_native)
