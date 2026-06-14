from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from wrappers.wrappers import DeepRLWrapper


class BasicNet:
    """
    Simple base class for Neural Networks.

    Attributes:
        gpu (bool): True, if the NN should live on the GPU.
        LSTM (bool): True, if the network is an LSTM.
        FloatTensor (torch.cude.FloatTensor | torch.FloatTensor): PyTorch
            float tensor depending on whether we are on GPU or not.
    """

    def __init__(self, optimizer_fn, gpu, LSTM=False):
        self.gpu = gpu and torch.cuda.is_available()
        self.LSTM = LSTM
        if self.gpu:
            self.cuda()
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

    def to_torch_variable(self, x: Any, dtype: str = "float32") -> Variable:
        """
        Converts the input to a PyTorch Variable.

        Args:
            x (Any): Object for conversion.
            dtype (str = "float32"): Data type of the object.

        Returns:
            Variable: `x` converted to a Variable.
        """
        if isinstance(x, Variable):
            return x
        if not isinstance(x, torch.FloatTensor):
            x = torch.from_numpy(np.asarray(x, dtype=dtype))
        if self.gpu:
            x = x.cuda()
        return Variable(x)

    def reset(self, terminal: bool):
        """
        Resets the network.

        Args:
            terminal (bool): True, if the data should be
                zeroed out.
        """
        if not self.LSTM:
            return
        if terminal:
            self.h.data.zero_()
            self.c.data.zero_()
        self.h = Variable(self.h.data)
        self.c = Variable(self.c.data)


# Initialized with normal(0, 0.01) weights so that at startup
# gamma is approx. 1 + tiny_noise and beta is approx. tiny_noise.
# This breaks the alpha-invariant symmetry without significantly
# disrupting the conv features at the start of training.

# Zero-init would give actor(s, alpha_1) = actor(s, alpha_2) for all alpha,
# making every gradient that tries to create alpha-sensitive policies exactly
# zero.


class FiLM(nn.Module):
    """
    Maps scalar alpha to per-channel (gamma, beta) for C channels.

    Attributes:
        linear (nn.Linear): Linear layer of dimension `(1, 2 * n_channels)`.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.linear = nn.Linear(1, n_channels * 2)
        nn.init.normal_(self.linear.weight, 0.0, 0.01)
        nn.init.ones_(self.linear.bias[:n_channels])  # gamma bias = 1
        nn.init.zeros_(self.linear.bias[n_channels:])  # beta bias = 0

    def forward(self, h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Performs one pass of the FiLM network.
        h has dimensions [B, C, H, W] and alpha is [B, 1].

        Args:
            h (torch.Tensor): Input data.
            alpha (torch.Tensor): Batch of alpha values.

        Returns:
            torch.Tensor: Output of the FiLM network.
        """
        params = self.linear(alpha)
        C = h.size(1)
        gamma = params[:, :C].view(-1, C, 1, 1)
        beta = params[:, C:].view(-1, C, 1, 1)
        return gamma * h + beta


# Original: alpha entered only at the output via Linear(6,5) applied to
# cat([alpha, conv_features]). That weight column can be zeroed, making
# the network alpha-invariant.

# New: alpha modulates each conv layer's output via FiLM so the CVaR
# gradient reaches every feature detector. The final Linear(6,5) and
# its alpha column are removed. Softmax is applied directly to the conv
# output (same topology as the original minus the out layer).


class DeterministicActorNetCVaR(nn.Module):
    """
    Actor with FiLM conditioning.
    """

    def __init__(
        self,
        state_dim: np.array,
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

        # Change 1: FiLM modules replace late alpha concatenation
        self.film1 = FiLM(h2)
        self.film2 = FiLM(h1)

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
            alpha (torch.Tensor): Tensor containing alpha values
                for the current batch.

        Returns:
            torch.Tensor: Output of the actor network.
        """
        x = self.to_var(x)
        w0 = x[:, :1, :1, :]
        x = x[:, :, 1:, :]
        alpha_t = self.to_var(alpha).view(-1, 1)

        h = self.non_linear(self.film1(self.conv1(x), alpha_t))
        h = self.non_linear(self.film2(self.conv2(h), alpha_t))

        h = torch.cat([h, w0], 1)
        action = self.conv3(h)

        cash_bias = self.to_var(torch.zeros(action.size(0), 1, 1, 1))
        action = torch.cat([cash_bias, action], -1)
        action = action.view(action.size(0), -1)

        if self.action_gate:
            action = self.action_scale * self.action_gate(action)

        return F.softmax(action, dim=1)

    def predict(
        self, x: torch.Tensor, alpha: torch.Tensor, to_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Does a single forward pass of the actor network and optionally converts
        the output to a NumPy array and puts in on the CPU, if `to_numpy` is `True`.

        Args:
            x (torch.Tensor): Input states.
            alpha (torch.Tensor): Alpha values for the current batch.
            to_numpy (bool = True): True, if the output should be converted to
                a NumPy array.

        Returns:
            torch.Tensor | np.array: Output of the actor network, optionally
                converted to a NumPy array.
        """
        y = self.forward(x, alpha)
        return y.cpu().data.numpy() if to_numpy else y


# Change 6: No alpha-conditioning in the critic.
# The return distribution (mu, sigma) of an action is an objective property
# of (state, action): the expected return-to-go and the Markowitz portfolio
# risk do not depend on the risk preference alpha (the targets mu_t and
# sigma_t are alpha-independent by construction). Earlier the critic was
# FiLM-conditioned on alpha, which let it learn a spurious alpha-dependence
# that competed with (and could flip) the explicit kappa(alpha) tilt in
# the actor loss (the cause of direction reversal at low sigma_scale).
# Removing it makes all alpha-sensitivity flow through kappa(alpha). The
# direction is then guaranteed and sigma_scale is a clean magnitude knob.


class DeterministicCriticNetCVaR(nn.Module):
    """
    Critic without FiLM conditioning.
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
        self.layer3 = nn.Linear(h0, 1)  # mu
        self.layer4 = nn.Linear(h0, 1)  # sigma (scale, not variance)

        self.non_linear = non_linear
        self.batch_norm = batch_norm
        BasicNet.__init__(self, None, gpu, False)

    def to_var(self, x: Any, dtype="float32") -> Variable:
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
            alpha (torch.Tensor): Alpha values for the batch (not used).
            action (torch.Tensor): Action taken by the actor network.

        Returns:
            torch.Tensor: Output of the critic network.
        """
        # alpha is not used
        x = self.to_var(x)
        action = self.to_var(action)[:, None, None, 1:]
        w0 = x[:, :1, :1, :]
        x = x[:, :, 1:, :]

        h = self.non_linear(self.conv1(x))
        h = self.non_linear(self.conv2(h))

        h = torch.cat([h, w0, action], 1)
        bs = x.size(0)
        h = self.non_linear(self.layer0(h.view(bs, -1)))

        mu = self.layer3(h)
        sigma = F.softplus(self.layer4(h))
        return torch.cat([mu, sigma], 1)

    def predict(self, x: torch.Tensor, alpha: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Does a single forward pass of the critic network.

        Args:
            x (torch.Tensor): Input states.
            alpha (torch.Tensor): Alpha values for the current batch.
            action (torch.Tensor): Action taken by the actor network.

        Returns:
            torch.Tensor: Output of the critic network.
        """
        return self.forward(x, alpha, action)


class DisjointActorCriticNet:
    """
    Container for the actor and critic networks.
    By default, they are independent (disjoint) but we have
    the option to share their memory.

    Attributes:
        actor (DeterministicActorNetCVaR): Actor network.
        critic (DeterministicCriticNetCVaR): Critic network.
    """

    def __init__(
        self,
        actor_network_fn: DeterministicActorNetCVaR,
        critic_network_fn: DeterministicCriticNetCVaR,
    ):
        self.actor = actor_network_fn()
        self.critic = critic_network_fn()

    def state_dict(self) -> list:
        """
        Returns the state dictionary of the actor and critic networks.

        Returns:
            list: The `state_dict` of the actor and critic networks,
                respectively.
        """
        return [self.actor.state_dict(), self.critic.state_dict()]

    def load_state_dict(self, state_dicts: list):
        """
        Loads a `state_dict` into the actor and critic networks.

        Args:
            state_dicts (list): State dictionaries of
                the actor and critic networks, respectively.
        """
        self.actor.load_state_dict(state_dicts[0])
        self.critic.load_state_dict(state_dicts[1])

    def share_memory(self):
        """
        Shares the memory of the actor and critic networks.
        """
        self.actor.share_memory()
        self.critic.share_memory()

    def predict(self):
        """
        Predicts using the trained actor and critic networks.
        """
        self.actor.predict()
        self.critic.predict()

    def parameters(self) -> list:
        """
        Returns the parameters of the actor and critic networks.

        Returns:
            list: Parameters of the actor and critic networks, respectively.
        """
        return list(self.actor.parameters()) + list(self.critic.parameters())

    def zero_grad(self):
        """
        Zeroes out the gradients of actor and critic networks.
        """
        self.actor.zero_grad()
        self.critic.zero_grad()

    def train(self):
        """
        Trains the actor and critic networks.
        """
        self.actor.train()
        self.critic.train()

    def eval(self):
        """
        Evaluates the actor and critic networks.
        """
        self.actor.eval()
        self.critic.eval()
