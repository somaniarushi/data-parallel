import torch

# Set random seed for reproducibility
torch.manual_seed(0)

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

# Function to create data
def create_data(N_data, device):
    x = torch.randn(N_data, 2).to(device)
    y = torch.randn(N_data, 1).to(device)
    return x, y

# Function to create model and apply DataParallel if more than one GPU is available
def create_model():
    net = Model()
    parallel_net = torch.nn.DataParallel(net)
    return parallel_net, net

# Function to copy model to multiple devices
def device_copy(cpu_model, device):
    model = Model()
    model.load_state_dict(cpu_model.state_dict())
    assert all(torch.allclose(a, b) for a, b in zip(model.parameters(), cpu_model.parameters()))
    model.to(device)
    return model

# Function to create models on all available GPUs
def create_models_on_devices(net):
    cpu_model = net.cpu()
    device_ids = list(range(torch.cuda.device_count()))
    models = [device_copy(cpu_model, f'cuda:{i}') for i in device_ids]
    # assert that all models are on different devices
    assert len(set(model.device for model in models)) == len(models)
    return models

# Function to print model devices
def print_model_devices(models):
    for model in models:
        print(model.device)

# Function to chunk data across devices
def chunk_data_across_devices(x, y, models):
    x_chunks = x.chunk(len(models), dim=0)
    y_chunks = y.chunk(len(models), dim=0)
    return x_chunks, y_chunks

# Function to compute loss and perform backward pass
def compute_loss_and_backward_pass(models, x, y, criterion):
    x_chunks, y_chunks = chunk_data_across_devices(x, y, models)
    outputs = [model(x_i.to(model.device)) for model, x_i in zip(models, x_chunks)]
    losses = [criterion(output, y_i.to(output.device)) for output, y_i in zip(outputs, y_chunks)]
    cpu_losses = [loss.cpu() for loss in losses]
    loss = torch.stack(cpu_losses).mean()
    for loss in losses:
        loss.backward()
    return loss, losses

# Function to average gradients across models
def average_gradients(models):
    grads_fc1 = [model.fc1.weight.grad.cpu() for model in models]
    grads_fc2 = [model.fc2.weight.grad.cpu() for model in models]
    avg_grad_fc1 = torch.stack(grads_fc1).mean(dim=0)
    avg_grad_fc2 = torch.stack(grads_fc2).mean(dim=0)
    for model in models:
        model.fc1.weight.grad = avg_grad_fc1.to(model.device)
        model.fc2.weight.grad = avg_grad_fc2.to(model.device)

# Function to step the optimizers
def step_optimizers(models):
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.01) for model in models]
    for optimizer in optimizers:
        optimizer.step()

# Function to assert models have the same weights
def assert_models_same_weights(models):
    for model in models[1:]:
        assert torch.allclose(models[0].fc1.weight.cpu(), model.fc1.weight.cpu())
        assert torch.allclose(models[0].fc2.weight.cpu(), model.fc2.weight.cpu())

def torch_dp_compute_loss_and_backward_pass(parallel_net, x, y, criterion):
    optimizer = torch.optim.SGD(parallel_net.parameters(), lr=0.01)
    optimizer.zero_grad()
    output = parallel_net(x)
    loss = criterion(output, y)
    loss.backward()
    return loss


def print_impl_diff(models, parallel_net):
    model_0_fc1 = models[0].fc1.weight.cpu()
    net_0_fc1 = parallel_net.module.fc1.weight.cpu()
    print(f"FC1")
    print(torch.abs(model_0_fc1 - net_0_fc1))
    
    model_0_fc2 = models[0].fc2.weight.cpu()
    net_0_fc2 = parallel_net.module.fc2.weight.cpu()
    print(f"FC2")
    print(torch.abs(model_0_fc2 - net_0_fc2))

# Main function to run the training process
def main():
    N_data = 400
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parallel_net, net = create_model()
    parallel_net.to(device)
    criterion = torch.nn.MSELoss()
    
    # Data parallelism with PyTorch
    x, y = create_data(N_data, device)
    torch_dp_loss = torch_dp_compute_loss_and_backward_pass(parallel_net, x, y, criterion)
    optimizer = torch.optim.SGD(parallel_net.parameters(), lr=0.01)
    optimizer.step()

    # Data parallelism with custom implementation
    models = create_models_on_devices(net)
    x, y = create_data(N_data, "cpu")
    loss, _ = compute_loss_and_backward_pass(models, x, y, criterion)
    average_gradients(models)
    step_optimizers(models)
    assert_models_same_weights(models) # our replicas should not diverge
    
    torch_dp_loss_cpu = torch_dp_loss.cpu()
    loss_cpu = loss.cpu()
    print(f"Torch DP Loss: {torch_dp_loss_cpu} | Custom DP Loss: {loss_cpu} | Loss Diff: {torch.abs(torch_dp_loss_cpu - loss_cpu)}")
    print_impl_diff(models, parallel_net)

if __name__ == "__main__":
    main()
