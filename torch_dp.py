import torch


def get_parallel_net(net, device):
    return torch.nn.DataParallel(net).to(device)


def torch_dp_compute_loss_and_backward_pass(parallel_net, x, y, criterion):
    optimizer = torch.optim.SGD(parallel_net.parameters(), lr=0.01)
    optimizer.zero_grad()
    output = parallel_net(x)
    loss = criterion(output, y)
    loss.backward()
    return loss


def torch_step_optimizer(parallel_net):
    optimizer = torch.optim.SGD(parallel_net.parameters(), lr=0.01)
    optimizer.step()
