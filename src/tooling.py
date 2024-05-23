import torch
from base import Model


def assert_models_same_weights(models):
    """
    Asserts that the weights of the models are the same.
    Assumes model is of type MLP with 2 layers.
    """
    for model in models:
        assert isinstance(model, Model), f"Model is not of type Model {type(model)}"

    for model in models[1:]:
        assert torch.allclose(models[0].fc1.weight.cpu(), model.fc1.weight.cpu())
        assert torch.allclose(models[0].fc2.weight.cpu(), model.fc2.weight.cpu())


def print_impl_diff(models, parallel_net):
    model_0_fc1 = models[0].fc1.weight.cpu()
    net_0_fc1 = parallel_net.module.fc1.weight.cpu()
    print("FC1")
    print(torch.abs(model_0_fc1 - net_0_fc1))

    model_0_fc2 = models[0].fc2.weight.cpu()
    net_0_fc2 = parallel_net.module.fc2.weight.cpu()
    print("FC2")
    print(torch.abs(model_0_fc2 - net_0_fc2))
