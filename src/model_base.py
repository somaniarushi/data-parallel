import torch
from abc import ABC, abstractmethod


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


def create_data(N_data, device):
    x = torch.randn(N_data, 2).to(device)
    y = torch.randn(N_data, 1).to(device)
    return x, y


# Function to create model and apply DataParallel if more than one GPU is available
def create_model():
    return Model()


class OneStepTrainer(ABC):
    @abstractmethod
    def compute_loss_and_backward_pass(self, net, x, y, criterion):
        pass

    @abstractmethod
    def step_optimizer(self):
        pass
