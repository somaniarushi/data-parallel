import torch
from src.model_base import Model

torch.manual_seed(0)


def device_copy(cpu_model, device):
    """
    Make a copy of the model on the specified device
    """
    model = Model()
    model.load_state_dict(cpu_model.state_dict())
    assert all(
        torch.allclose(a, b) for a, b in zip(model.parameters(), cpu_model.parameters())
    )
    model.to(device)
    return model


# Function to create models on all available GPUs
def create_models_on_devices(net):
    """
    For every available GPU, create a copy of the model on that GPU
    """
    cpu_model = net.cpu()
    device_ids = list(range(torch.cuda.device_count()))
    models = [device_copy(cpu_model, f"cuda:{i}") for i in device_ids]
    assert len(set(model.device for model in models)) == len(
        models
    ), "Models are not on different devices"
    return models


def print_model_devices(models):
    """
    Helper function to print the devices of the models
    """
    for model in models:
        print(model.device)


def chunk_data_across_devices(x, y, models):
    """
    Helper function to chunk data across devices for data parallelism.
    Assumes that the data can be evenly divided across devices.
    """
    x_chunks = x.chunk(len(models), dim=0)
    y_chunks = y.chunk(len(models), dim=0)
    return x_chunks, y_chunks


# Function to compute loss and perform backward pass
def compute_loss_and_backward_pass(models, x, y, criterion):
    """
    For custom data parallelism, compute loss and perform backward pass.
    - First, chunk the data across devices
    - Then, compute the loss and perform backward pass
    Stack all the losses and return the mean loss
    """
    x_chunks, y_chunks = chunk_data_across_devices(x, y, models)
    outputs = [model(x_i.to(model.device)) for model, x_i in zip(models, x_chunks)]
    losses = [
        criterion(output, y_i.to(output.device))
        for output, y_i in zip(outputs, y_chunks)
    ]
    cpu_losses = [
        loss.cpu() for loss in losses
    ]  # In disconnected GPUs, would need to communicate
    loss = torch.stack(cpu_losses).mean()
    for loss in losses:
        loss.backward()
    return loss, losses


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


def step_optimizers(models):
    """
    Create optimizers for each of the model replicas and step them
    """
    # TODO -> Remove assumption that only one step is taken
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.01) for model in models]
    for optimizer in optimizers:
        optimizer.step()
