import torch
from src.ring_reduce import ring_reduce


def test_ring_reduce_3_tensors() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    c = torch.tensor([7.0, 8.0, 9.0])

    outputs = ring_reduce([a, b, c])
    # Assert that all outputs are the same
    for output in outputs:
        assert torch.allclose(output, torch.tensor([12.0, 15.0, 18.0]))


def test_ring_reduce_4_tensors() -> None:
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([1.0, 2.0, 3.0, 4.0])
    c = torch.tensor([1.0, 2.0, 3.0, 4.0])
    d = torch.tensor([1.0, 2.0, 3.0, 4.0])

    outputs = ring_reduce([a, b, c, d])
    # Assert that all outputs are the same
    for output in outputs:
        assert torch.allclose(output, torch.tensor([4.0, 8.0, 12.0, 16.0]))
