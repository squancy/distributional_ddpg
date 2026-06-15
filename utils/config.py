import gym
import pandas as pd
import torch
import torch.nn.functional as F

from network.network import (
    DeterministicActorNetCVaR,
    DeterministicCriticNetCVaR,
    DisjointActorCriticNet,
)
from utils.replay_buffer import ReplayAlphaStratified
from utils.utils import Logger, OrnsteinUhlenbeckProcess
from wrappers.wrappers import env_wrapper


class Config:
    """
    Container for all configs.
    """

    q_target = 0

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        window: int,
        steps: int,
        root: str,
        tag: str,
    ):
        self.task_fn_H = None
        self.optimizer_fn = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.network_fn_H = None
        self.policy_fn = None
        self.replay_Hierachical = None
        self.target_network_update_freq = 0
        self.exploration_steps = 0
        self.history_length = 1
        self.double_q = False
        self.num_workers = 1
        self.worker = None
        self.worker_H = None
        self.update_interval = 1
        self.entropy_weight = 0.01
        self.gae_tau = 1.0
        self.noise_decay_interval = 0
        self.action_shift_fn = lambda a: a
        self.reward_shift_fn = lambda r: r
        self.reward_weight = 1
        self.hybrid_reward = False
        self.target_type = self.q_target
        self.master_fn = None
        self.master_optimizer_fn = None
        self.num_heads = 10
        self.min_epsilon = 0
        self.success_threshold = float("inf")
        self.render_episode_freq = 0
        self.actor_high = None
        self.critic_high = None

        self.task_fn = lambda: env_wrapper(df=df_train, steps=128, window_length=window)
        self.task_fn_vali = lambda: env_wrapper(
            df=df_val, steps=len(df_val) - window - 2, window_length=window, random_reset=False
        )
        self.task_fn_test = lambda: env_wrapper(
            df=df_test, steps=500, window_length=window, random_reset=False
        )
        # Build the env once for network-dimension inference and expose it so the
        # caller can reuse it (e.g. for n_assets) instead of constructing another.
        # Each random_reset=True construction draws from the global RNG, so an
        # extra throwaway env would shift the stream and break seeded reproduction.
        self.task = task = self.task_fn()

        self.actor_network_fn = lambda: DeterministicActorNetCVaR(
            state_dim=task.state_dim,
            task=task,
            action_gate=None,
            action_scale=1.0,
            non_linear=F.relu,
            batch_norm=False,
        )
        self.critic_network_fn = lambda: DeterministicCriticNetCVaR(
            state_dim=task.state_dim,
            action_dim=task.action_dim,
            task=task,
            non_linear=F.relu,
            batch_norm=False,
            gpu=False,
        )
        self.network_fn = lambda: DisjointActorCriticNet(
            self.actor_network_fn, self.critic_network_fn
        )
        self.actor_optimizer_fn = lambda params: torch.optim.Adam(
            params,
            lr=1e-5,
            weight_decay=1e-4,
        )
        self.critic_optimizer_fn = lambda params: torch.optim.Adam(
            params, lr=1e-4, weight_decay=0.001
        )
        self.replay_fn = lambda: ReplayAlphaStratified(
            memory_size=int(1e6), batch_size=32, n_bins=5
        )

        # Change 3: keep exploration alive through approx. episode 2300 so the critic
        # sees diverse portfolio allocations before the actor converges.
        self.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
            size=task.action_dim, theta=0.3, sigma=0.3, sigma_min=0.05, n_steps_annealing=300000
        )

        self.lambda_entropy = 0.03
        # Scales the env's daily portfolio std sqrt(w^T \sigma w) into reward units. The base
        # 10000/steps matches the env's reward scaling (reward = log_return*10000/steps).
        # It's the primary alpha-sensitivity signal: it sets how large
        # the (\kappa − \kappa_min) * \sigma penalty is relative to \mu, hence how far apart the
        # \alpha=5% and \alpha=50% policies sit. A larger sigma_scale results in a wider frontier.
        self.sigma_scale = 2.5 * 10000.0 / steps
        self.discount = 0.99
        self.min_memory_size = 1000
        self.max_steps = 1000000
        self.max_episode_length = 3000
        self.target_network_mix = 0.001
        self.gradient_clip = 20
        self.test_interval = 50
        self.test_repetitions = 1
        self.val_interval = 100  # validate + checkpoint-select every 100 episodes
        self.episode_limit = 5000
        self.save_interval = 500
        self.logger = Logger(root + "/log", gym.logger)
        self.tag = tag
