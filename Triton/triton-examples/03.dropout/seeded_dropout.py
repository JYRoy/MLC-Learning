import tabulate
import torch

import triton
import triton.language as tl

"""
-------------------  ---------  ---------  --------  ---------  ---------  ---------  ---------  ------  ---------  -------
input                -0.739172  -0.261161  -0.43497  0.0026802  -0.905076  -0.294771  -0.287306  2.6698  -0.967081  1.27902
output (seed = 123)   0         -0.522322   0        0           0         -0.589543   0         0       -1.93416   2.55805
output (seed = 123)   0         -0.522322   0        0           0         -0.589543   0         0       -1.93416   2.55805
output (seed = 512)   0          0         -0.86994  0.0053604   0         -0.589543  -0.574613  0        0         0
-------------------  ---------  ---------  --------  ---------  ---------  ---------  ---------  ------  ---------  -------
"""


@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


x = torch.randn(size=(10,)).cuda()
# Compare this to the baseline - dropout mask is never instantiated!
output = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)

print(
    tabulate.tabulate(
        [
            ["input"] + x.tolist(),
            ["output (seed = 123)"] + output.tolist(),
            ["output (seed = 123)"] + output2.tolist(),
            ["output (seed = 512)"] + output3.tolist(),
        ]
    )
)
