from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from network.network import BasicNet
from wrappers.wrappers import DeepRLWrapper


class DeterministicActorNetCVaR(nn.Module):
    """
    Actor with late alpha injection (cat([alpha, action]) -> Linear -> softmax).
    """

    def __init__(
        self,
        state_dim: np.array,
        action_dim: int,
        task: DeepRLWrapper,
        action_gate: Any,
        action_scale: Any,
        batch_norm: bool = False,
        non_linear: torch.nn.functional.relu = F.relu,
    ):
        super().__init__()

        stride_time = state_dim[1] - 1 - 2
        features = task.state_dim[0]
        h1, h2 = 32, 16

        self.conv1 = nn.Conv2d(features, h2, (3, 1))
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.conv3 = nn.Conv2d(h1 + 1, 1, (1, 1))
        self.out = nn.Linear(action_dim + 1, action_dim)  # cat([alpha, action]) -> action

        self.action_scale = action_scale
        self.action_gate = action_gate
        self.non_linear = non_linear
        self.batch_norm = batch_norm

    def to_var(self, x: Any, dtype: str = "float32") -> Variable:
        """
        Converts the input to a PyTorch Variable.

        Args:
            x (Any): Array for conversion.
            dtype (str = 'float32'): Array data type.

        Returns:
            Variable: Array converted to a PyTorch Variable.
        """
        if isinstance(x, Variable):
            return x
        return Variable(torch.from_numpy(np.asarray(x, dtype=dtype)))

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Does a single forward pass of the actor network.

        Args:
            x (torch.Tensor): Tensor containing previous states.
            alpha (torch.Tensor): Tensor containing alpha values.

        Returns:
            torch.Tensor: Output of the actor network (portfolio weights).
        """
        x = self.to_var(x)
        w0 = x[:, :1, :1, :]
        x = x[:, :, 1:, :]
        alpha = self.to_var(alpha)

        h = self.non_linear(self.conv1(x))
        h = self.non_linear(self.conv2(h))
        h = torch.cat([h, w0], 1)
        action = self.conv3(h)

        cash_bias = self.to_var(torch.zeros(action.size())[:, :, :, :1])
        action = torch.cat([cash_bias, action], -1)
        batch_size = action.size()[0]
        action = action.view((batch_size, -1))
        alpha = alpha.view((batch_size, -1))

        if self.action_gate:
            action = self.action_scale * self.action_gate(action)

        Action = torch.cat([alpha, action], 1)
        Action = self.out(Action)
        return F.softmax(Action, dim=1)

    def predict(
        self, x: torch.Tensor, alpha: torch.Tensor, to_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Does a single forward pass of the actor network and optionally converts
        the output to a NumPy array on the CPU.

        Args:
            x (torch.Tensor): Input states.
            alpha (torch.Tensor): Alpha values.
            to_numpy (bool = True): True, if the output should be a NumPy array.

        Returns:
            torch.Tensor | np.array: Output of the actor network.
        """
        y = self.forward(x, alpha)
        return y.cpu().data.numpy() if to_numpy else y


class DeterministicCriticNetCVaR(nn.Module):
    """
    Critic with alpha concatenated at the bottleneck (cat([h, alpha])).
    """

    def __init__(
        self,
        state_dim: np.array,
        action_dim: int,
        task: DeepRLWrapper,
        gpu: bool = False,
        batch_norm: bool = False,
        non_linear: torch.nn.functional.relu = F.relu,
    ):
        super().__init__()

        stride_time = state_dim[1] - 1 - 2
        features = task.state_dim[0]
        h0, h1, h2 = 8, 32, 16
        self.action = actions = action_dim - 1

        self.conv1 = nn.Conv2d(features, h2, (3, 1))
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.layer0 = nn.Linear((h1 + 2) * actions, h0)
        self.layer3 = nn.Linear(h0 + 1, 1)  # mu output
        self.layer4 = nn.Linear(h0 + 1, 1)  # sigma output (scale, not variance)

        self.non_linear = non_linear
        self.batch_norm = batch_norm
        BasicNet.__init__(self, None, gpu, False)

    def to_var(self, x: Any, dtype: str = "float32") -> Variable:
        """
        Converts the input to a PyTorch Variable.

        Args:
            x (Any): Array for conversion.
            dtype (str = 'float32'): Array data type.

        Returns:
            Variable: Array converted to a PyTorch Variable.
        """
        if isinstance(x, Variable):
            return x
        return Variable(torch.from_numpy(np.asarray(x, dtype=dtype)))

    def forward(self, x: torch.Tensor, alpha: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Does a single forward pass of the critic network.

        Args:
            x (torch.Tensor): Input states.
            alpha (torch.Tensor): Alpha values (concatenated at the bottleneck).
            action (torch.Tensor): Action taken by the actor network.

        Returns:
            torch.Tensor: Concatenated (mu, sigma) of the return distribution.
        """
        x = self.to_var(x)
        action = self.to_var(action)[:, None, None, 1:]
        w0 = x[:, :1, :1, :]
        x = x[:, :, 1:, :]
        Alpha = self.to_var(alpha)

        h = self.non_linear(self.conv1(x))
        h = self.non_linear(self.conv2(h))
        h = torch.cat([h, w0, action], 1)

        batch_size = x.size()[0]
        h1 = self.non_linear(self.layer0(h.view((batch_size, -1))))
        alpha1 = Alpha.view((batch_size, -1))
        hh = torch.cat([h1, alpha1], 1)

        mu = self.layer3(hh)
        sigma = F.softplus(self.layer4(hh))
        return torch.cat([mu, sigma], 1)

    def predict(self, x: torch.Tensor, alpha: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Does a single forward pass of the critic network.

        Args:
            x (torch.Tensor): Input states.
            alpha (torch.Tensor): Alpha values.
            action (torch.Tensor): Action taken by the actor network.

        Returns:
            torch.Tensor: Output of the critic network.
        """
        return self.forward(x, alpha, action)
