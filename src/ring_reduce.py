import torch
from typing import List


def ring_reduce(tensor_lst: List[torch.Tensor]) -> torch.Tensor:
    # TODO(Arushi) -> Add support for long lists of short tensors
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

    return GRADIENTS_LIST[0]


def average_gradients(models):
    """
    For every model param, get the gradients from all models, average them
    and then load the average gradient back in linear fashion.
    """
    grads_fc1 = [model.fc1.weight.grad.cpu() for model in models]
    grads_fc2 = [model.fc2.weight.grad.cpu() for model in models]
    avg_grad_fc1 = torch.stack(grads_fc1).mean(dim=0)
    avg_grad_fc2 = torch.stack(grads_fc2).mean(dim=0)
    for model in models:
        model.fc1.weight.grad = avg_grad_fc1.to(model.device)
        model.fc2.weight.grad = avg_grad_fc2.to(model.device)


def average_gradients_ring_allreduce(models):
    """
    Average the gradients using ring reduce.
    """
    # For fc1
    NUM_GPUS = len(models)
    grads_fc1 = [model.fc1.weight.grad for model in models]
    # Pad to make the list length equal to the number of GPUs
    grads_fc1 += [torch.zeros_like(grads_fc1[0]) for _ in range(NUM_GPUS - len(models))]
    avg_grads_fc1 = ring_reduce(grads_fc1)

    # For fc2
    grads_fc2 = [model.fc2.weight.grad for model in models]
    # Pad to make the list length equal to the number of GPUs
    grads_fc2 += [torch.zeros_like(grads_fc2[0]) for _ in range(NUM_GPUS - len(models))]
    avg_grads_fc2 = ring_reduce(grads_fc2)

    # Load the average gradients back
    for model in models:
        model.fc1.weight.grad = avg_grads_fc1.to(model.device)
        model.fc2.weight.grad = avg_grads_fc2.to(model.device)
