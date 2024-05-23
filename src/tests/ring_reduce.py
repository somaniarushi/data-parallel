import torch
from typing import List


def ring_reduce(tensor_lst: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Ring reduce algorithm.

    Manually and explicitly:
    # First step, reduction
    b[0] = b[0] + a[0]
    c[1] = c[1] + b[1]
    a[2] = a[2] + c[2]

    # Second step, reduction
    c[0] = c[0] + b[0]
    a[1] = a[1] + c[1]
    b[2] = b[2] + a[2]

    # Last step, shove values to next
    a[0] = c[0]
    b[1] = a[1]
    c[2] = b[2]

    # Shove values to one next again
    b[0] = a[0]
    c[1] = b[1]
    a[2] = c[2]
    """
    # Assert that all tensors have length 3
    NUM_GPUS = len(tensor_lst)

    GRADIENTS_LIST = tensor_lst
    for i in range(NUM_GPUS - 1):
        for j in range(NUM_GPUS):
            GRADIENTS_LIST[(i + j + 1) % NUM_GPUS][j] += GRADIENTS_LIST[
                (i + j) % NUM_GPUS
            ][j]

    print(GRADIENTS_LIST)
    for i in range(NUM_GPUS - 1):
        for j in range(NUM_GPUS):
            GRADIENTS_LIST[(i + j) % NUM_GPUS][j] = GRADIENTS_LIST[
                (i + j - 1) % NUM_GPUS
            ][j]

    print(GRADIENTS_LIST)
    return GRADIENTS_LIST
