import torch
from src.model_base import create_data, create_model
from src.torch_dp import (
    torch_dp_compute_loss_and_backward_pass,
    get_parallel_net,
    torch_step_optimizer,
)
from simple_dp import (
    create_models_on_devices,
    compute_loss_and_backward_pass,
    average_gradients,
    step_optimizers,
)
from src.tooling import (
    assert_models_same_weights,
    print_impl_diff,
)

N_data = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET = create_model()
CRITERION = torch.nn.MSELoss()


# Data parallelism with PyTorch
def run_pytorch():
    parallel_net = get_parallel_net(NET, DEVICE)
    x, y = create_data(N_data, DEVICE)
    torch_dp_loss = torch_dp_compute_loss_and_backward_pass(
        parallel_net, x, y, CRITERION
    )
    torch_step_optimizer(parallel_net)
    return parallel_net, torch_dp_loss


# Custom simple data parallelism
def run_custom():
    models = create_models_on_devices(NET)
    x, y = create_data(N_data, "cpu")  # Make on CPU, easy to copy chunks to devices
    loss, _ = compute_loss_and_backward_pass(models, x, y, CRITERION)
    average_gradients(models)
    step_optimizers(models)
    assert_models_same_weights(models)  # our replicas should not diverge
    return models, loss


parallel_net, torch_dp_loss = run_pytorch()
models, loss = run_custom()

torch_dp_loss_cpu, loss_cpu = torch_dp_loss.cpu(), loss.cpu()
print(
    f"Torch DP Loss: {torch_dp_loss_cpu} | Custom DP Loss: {loss_cpu} | Loss Diff: {torch.abs(torch_dp_loss_cpu - loss_cpu)}"
)
print_impl_diff(models, parallel_net)
