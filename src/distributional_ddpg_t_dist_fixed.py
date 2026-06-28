"""
Distributional DDPG with Student-t return distribution and a fixed risk
tolerance alpha supplied on the command line. Train one model per alpha level
to produce a fair comparison with the paper's four
Normal-distribution DDPG models (each trained at a single fixed alpha).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import datetime
import logging
import pickle
import random as _random

import gym
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from tensorboard_logger import configure, log_value

from eval.eval_test import TestFixedDDPG
from utils.config import ConfigFixed
from utils.utils import estimate_t_dof, kappa_of, tensor

_parser = argparse.ArgumentParser(description="Fixed-alpha Student-t Distributional DDPG")
_parser.add_argument(
    "--alpha",
    type=float,
    required=True,
    help="Fixed CVaR risk tolerance, e.g. 0.05, 0.15, 0.30, 0.50",
)
_parser.add_argument(
    "--window", type=int, default=15, help="Observation window length (default: 15)"
)
args = _parser.parse_args()
if not (0.0 < args.alpha < 1.0):
    _parser.error("--alpha must be strictly between 0 and 1")

FIXED_ALPHA = args.alpha
window = args.window
alpha_pct = int(round(FIXED_ALPHA * 100))

_random.seed(0)
np.random.seed(0)  # noqa: NPY002 (seed the global legacy RNG the code uses)
torch.manual_seed(0)
print(f"[seed] deterministic run, SEED={0}")

matplotlib.rc("figure", figsize=[15, 10])
gym.logger.setLevel(logging.INFO)

root = os.getcwd()
ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H-%M-%S")
tag = f"ddpg-t-dist-fixed-a{alpha_pct:02d}-{ts}"
print("tensorboard --logdir runs/" + tag)
try:
    configure("runs/" + tag)
except ValueError as e:
    print(e)


# STEP 1: MLE estimate of Student-t degrees of freedom
print("\n" + "=" * 60)
print("STEP 1: Estimating Student-t degrees of freedom (nu)")
print("=" * 60)
nu_hat = estimate_t_dof(data_dir="/data", train_fraction=0.8)
print(f"\n>>> nu_hat = {nu_hat:.4f}")
print("=" * 60 + "\n")

path_data = "/data/poloniex_fc.hf"
df_train = pd.read_hdf(path_data, key="train", encoding="utf-8")
df_test = pd.read_hdf(path_data, key="test", encoding="utf-8")


class DDPGAgent(mp.Process):
    """
    Fixed-alpha distributional DDPG.
    """

    def __init__(self, config: ConfigFixed):
        self.config = config
        self.task = config.task_fn()
        self.worker_network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.worker_network.state_dict())
        self.actor_opt = config.actor_optimizer_fn(self.worker_network.actor.parameters())
        self.critic_opt = config.critic_optimizer_fn(self.worker_network.critic.parameters())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.actor = self.worker_network.actor
        self.critic = self.worker_network.critic
        self.target_actor = self.target_network.actor
        self.target_critic = self.target_network.critic
        self.error = 1e-8

        # alpha is constant: pre-build the (batch_size, 1) tensor used every
        # update step, and the scalar CVaR kappa = kappa(alpha, nu).
        bs = config.batch_size
        self._alpha_batch = tensor(np.full((bs, 1), FIXED_ALPHA, dtype=np.float32))
        self._kappa = float(kappa_of(FIXED_ALPHA, nu_hat))
        self._kappa_t = torch.tensor([[self._kappa]], dtype=torch.float32)

    def soft_update(self, target: nn.Module, src: nn.Module):
        """
        Polyak-average the target network parameters toward the source network.

        Args:
            target (nn.Module): Target network (updated in place).
            src (nn.Module): Source network.
        """
        mix = self.config.target_network_mix
        for tp, p in zip(target.parameters(), src.parameters()):
            tp.detach_()
            tp.copy_(tp * (1.0 - mix) + p * mix)

    def save(self, file_name: str):
        """
        Save the worker network (actor and critic) to a file.

        Args:
            file_name (str): File to save to.
        """
        with open(file_name, "wb") as f:
            torch.save(self.worker_network.state_dict(), f)

    def episode(self, deterministic: bool = False) -> tuple:
        """
        Roll out one episode at the fixed alpha.

        Args:
            deterministic (bool = False): True for a noise-free test rollout
                (no exploration noise, no replay writes, no updates).

        Returns:
            tuple[float, int]: Total reward and number of steps taken.
        """
        self.random_process.reset_states()
        state = self.task.reset()
        config = self.config
        steps_ep = 0
        total_reward = 0.0
        alpha_arr = np.array([FIXED_ALPHA])  # shape (1,) for actor.predict

        while True:
            self.actor.eval()
            action = self.actor.predict(np.stack([state]), alpha_arr).flatten()
            if not deterministic:
                action += self.random_process.sample()

            next_state, reward, done, info = self.task.step(action)
            done = done or (config.max_episode_length and steps_ep >= config.max_episode_length)
            total_reward += reward

            prefix = "test_" if deterministic else ""
            log_value(prefix + "reward", reward, self.total_steps)
            for key in info:
                log_value(key, info[key], self.total_steps)

            if not deterministic:
                # Entropy bonus added to the stored reward to encourage diverse
                # (non-collapsed) allocations early in training.
                w = np.clip(action, 1e-8, 1.0)
                ent_bonus = config.entropy_bonus * (-np.sum(w * np.log(w)))
                # Change 5: genuine per-step Markowitz portfolio risk
                # sqrt(w^T Sigma w) (env-computed, info['portfolio risk']),
                # scaled to reward units.  This is the sigma SOURCE term that
                # replaces the collapsing pure bootstrap (see _update).
                risk = float(np.sqrt(max(info["portfolio risk"], 0.0))) * config.sigma_scale
                self.replay.feed([state, action, reward + ent_bonus, risk, next_state, int(done)])
                self.total_steps += 1

            steps_ep += 1
            state = next_state
            if done:
                break

            if not deterministic and self.replay.size() >= config.min_memory_size:
                self._update(config)

        return total_reward, steps_ep

    def _update(self, config: ConfigFixed):
        """
        One critic + actor update from a replay batch (alpha held constant).

        Args:
            config (ConfigFixed): Container for all configurations.
        """
        states, actions, rewards, risks, next_states, terminals = self.replay.sample()
        states = tensor(states)
        actions = tensor(actions)
        rewards = tensor(rewards).unsqueeze(-1)
        risks = tensor(risks).unsqueeze(-1)
        mask = tensor(1 - terminals).unsqueeze(-1)
        next_states = tensor(next_states)
        alphas = self._alpha_batch  # constant alpha; broadcast the pre-built tensor

        # critic: W_2 loss between t(nu, mu_t, sigma_t) and t(nu, mu_p, sigma_p).
        q_next = self.target_critic.predict(
            next_states, alphas, self.target_actor.predict(next_states, alphas)
        )
        mu = q_next[:, 0].unsqueeze(-1)
        sigma = q_next[:, 1].unsqueeze(-1)
        mu_t = (config.discount * mu * mask + rewards + self.error).detach()

        # Change 5: proper distributional variance bootstrap with a genuine risk
        # source term, sigma_t = sqrt(sigma_step^2 + gamma^2 * sigma_next^2).
        # The plain bootstrap sigma_t = gamma * sigma_next collapses to its only
        # fixed point 0, so kappa * sigma -> 0 and the fixed-alpha CVaR objective
        # degenerates to pure mu-maximisation (identical across alpha, and never
        # conservative at small alpha).  risks = scaled sqrt(w^T Sigma w) is the
        # per-step Markowitz risk; it is positively risk-correlated, so a small
        # alpha (large kappa) is now genuinely penalised for holding risk.
        sigma_t = torch.sqrt(risks**2 + (config.discount * sigma * mask) ** 2 + self.error).detach()

        q = self.critic.predict(states, alphas, actions)
        mu_p = q[:, 0].unsqueeze(-1) + self.error
        sigma_p = q[:, 1].unsqueeze(-1) + self.error

        nu_factor = nu_hat / (nu_hat - 2.0)
        critic_loss = torch.pow(mu_t - mu_p, 2) + nu_factor * torch.pow(sigma_t - sigma_p, 2)
        cl = critic_loss.mean()

        self.critic.zero_grad()
        self.critic_opt.zero_grad()
        cl.backward()
        grad_critic = nn.utils.clip_grad_norm_(self.critic.parameters(), config.gradient_clip)
        self.critic_opt.step()

        # actor: t-CVaR gradient, maximize E[mu - kappa * sigma] at the fixed alpha.
        Actions = self.actor.predict(states, alphas, to_numpy=False)
        score = self.critic.predict(states, alphas, Actions)
        Amu = score[:, 0].unsqueeze(-1) + self.error
        Asigma = score[:, 1].unsqueeze(-1) + self.error

        qq = Amu - self._kappa_t * Asigma
        policy_loss = -qq.mean()

        self.actor.zero_grad()
        self.actor_opt.zero_grad()
        policy_loss.backward()
        grad_actor = nn.utils.clip_grad_norm_(self.actor.parameters(), config.gradient_clip)
        self.actor_opt.step()

        log_value("critic_loss", critic_loss.sum().item(), self.total_steps)
        log_value("policy_loss", policy_loss.item(), self.total_steps)
        if config.gradient_clip:
            log_value("grad_critic", grad_critic, self.total_steps)
            log_value("grad_actor", grad_actor, self.total_steps)

        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

    def _step(self, state: torch.Tensor, alpha: float = None) -> torch.Tensor:
        """
        Predict the portfolio weights for a single state (at the fixed alpha
        unless one is explicitly provided).

        Args:
            state (torch.Tensor): A single state.
            alpha (float = None): Optional alpha override (defaults to FIXED_ALPHA).

        Returns:
            torch.Tensor: Predicted portfolio weights.
        """
        a = np.array([FIXED_ALPHA]) if alpha is None else np.array([alpha])
        return self.actor.predict(np.stack([state]), a).flatten()


config = ConfigFixed(
    df_train=df_train,
    df_test=df_test,
    window=window,
    root=root,
    tag=tag,
)
task = config.task
agent = DDPGAgent(config)
test_eval = TestFixedDDPG(agent=agent, config=config)

agent.task._plot = agent.task._plot2 = None
print("Training from random initialization.")


def run_episodes(agent: DDPGAgent):
    """
    Train the fixed-alpha agent: roll out episodes, periodically save and run
    a deterministic test episode.

    Args:
        agent (DDPGAgent): Agent to train.
    """
    config = agent.config
    ep = 0
    rewards = []
    steps_list = []
    avg_test_rew = []
    agent_type = agent.__class__.__name__

    while True:
        ep += 1
        reward, step = agent.episode()
        rewards.append(reward)
        steps_list.append(step)
        config.logger.info(
            f"episode {ep}, reward {round(reward, 4)}, avg {round(np.mean(rewards), 4)}, total_steps {agent.total_steps}, ep_steps {step}"
        )

        if config.save_interval and ep % config.save_interval == 0:
            with open(
                root + f"/video/{agent_type}-{config.tag}-online-stats-{agent.task.name}.bin", "wb"
            ) as f:
                pickle.dump([steps_list, rewards], f)

        if config.episode_limit and ep > config.episode_limit:
            break

        if config.max_steps and agent.total_steps > config.max_steps:
            break

        if config.test_interval and ep % config.test_interval == 0:
            config.logger.info("Testing...")
            agent.save(root + f"/video/{agent_type}-{config.tag}-model-{agent.task.name}.bin")
            test_rewards = [
                agent.episode(deterministic=True)[0] for _ in range(config.test_repetitions)
            ]
            avg_rew = np.mean(test_rewards)
            avg_test_rew.append(avg_rew)
            config.logger.info(
                f"Test avg reward {round(avg_rew, 4)} (±{round(np.std(test_rewards) / np.sqrt(config.test_repetitions), 4)})"
            )
            with open(
                root + f"/video/{agent_type}-{config.tag}-all-stats-{agent.task.name}.bin", "wb"
            ) as f:
                pickle.dump(
                    {"rewards": rewards, "steps": steps_list, "test_rewards": avg_test_rew}, f
                )


ckpt_name = f"t_dist_ddpg_fixed_alpha{alpha_pct:02d}_win{window}_etf.pth"
ckpt_path = os.path.join(root, "video", ckpt_name)

print("\n" + "=" * 60)
print("TRAINING - Fixed-alpha Student-t Distributional DDPG")
print(f"  alpha (fixed): {FIXED_ALPHA}  ({alpha_pct}%)")
print(f"  Window       : {window}")
print(f"  Episodes     : {config.episode_limit}")
print(f"  nu (nu_hat)  : {nu_hat:.4f}")
print(f"  kappa(a, nu) : {agent._kappa:.6f}")
print(f"  Checkpoint   : {ckpt_path}")
print("=" * 60 + "\n")

try:
    run_episodes(agent)
except KeyboardInterrupt:
    print("\nTraining interrupted.")

torch.save(agent.worker_network.state_dict(), ckpt_path)
print(f"\nTrained model saved to: {ckpt_path}")

test_eval.evaluate()
