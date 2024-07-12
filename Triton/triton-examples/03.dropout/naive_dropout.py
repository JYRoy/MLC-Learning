import tabulate
import torch

import triton
import triton.language as tl

"""
---------  --------  ---------  --------  ---------  ----------  -------  --------  ---------  -------  ---------
input      -1.84012  -0.351695  -2.01904  -0.340446  0.00737043  0.22306  0.486131  -0.194233  1.15151  -0.823927
keep mask   1         1          1         1         0           0        1          0         1         0
output     -3.68024  -0.70339   -4.03808  -0.680891  0           0        0.972263   0         2.30301   0
---------  --------  ---------  --------  ---------  ----------  -------  --------  ---------  -------  ---------
"""


@triton.jit
def _dropout(
    x_ptr,  # pointer to the input
    x_keep_ptr,  # pointer to a mask of 0s and 1s
    output_ptr,  # pointer to the output
    n_elements,  # number of elements in the `x` tensor
    p,  # probability that an element of `x` is changed to zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


# Input tensor
x = torch.randn(size=(10,)).cuda()
# Dropout mask
p = 0.5
x_keep = (torch.rand(size=(10,)) > p).to(torch.int32).cuda()  # 根据 p 构造出一个mask矩阵
#
output = dropout(x, x_keep=x_keep, p=p)
print(
    tabulate.tabulate(
        [
            ["input"] + x.tolist(),
            ["keep mask"] + x_keep.tolist(),
            ["output"] + output.tolist(),
        ]
    )
)
