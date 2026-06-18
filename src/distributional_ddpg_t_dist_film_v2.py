"""
Distributional DDPG with Student-t return distribution (alpha-sensitive).

Based directly on distributional_ddpg_t_dist.py.

Change 1): FiLM conditioning in actor and critic
    In the original, alpha enters only at the output layer of the actor
    (via a Linear(6,5) applied to cat([alpha, action])) and at the
    bottleneck of the critic (via cat([h, alpha])). Both are stable
    local minima where the alpha weights stay at zero.

    Fix: replace late alpha injection with Feature-wise Linear Modulation
    (FiLM) after each conv layer in both networks. A small Linear(1, 2C)
    maps scalar alpha to per-channel (gamma, beta). The feature map is
    transformed as h <- gamma * h + beta before the nonlinearity. Alpha
    now modulates every feature at every depth so the CVaR gradient flows
    through the full network.

    FiLM weights are initialised with normal(0, 0.01), not zero.
    Zero-init forces actor(s, alpha_1) = actor(s, alpha_2) at startup,
    preventing the CVaR gradient from breaking alpha-invariant symmetry.

Change 2): Training alpha range restricted to [0.05, 0.50]
    Uniform(0,1) includes alpha approx. 0 where kappa -> ∞. With poloniex
    volatility, kappa * sigma > mu for every risky portfolio at those
    extreme alphas, so the model trains toward cash for ~50% of episodes.
    Restricting to [0.05, 0.50] keeps kappa in approx. [0.87, 2.4] and
    matches the evaluation range exactly.

Change 3): OU noise annealing extended to 300 000 steps
    The original anneals from 0.3 to 0.01 over 10 000 steps (episode 78
    of 5000). After that the replay buffer fills with near-identical
    actor actions, eliminating the portfolio diversity the critic needs to
    learn action-dependent sigma. 300 000 steps (episode 2300) keeps
    exploration alive through most of training.

Change 4): Entropy regularization in the actor loss (anti-saturation)
    Without it the softmax may collapse to a single asset. It acts as
    a safeguard against saturation but has a low coefficient so that
    it won't become the main driving force in the loss function.

Change 5): Sigma grounded in genuine portfolio risk (fixes inverted alpha)
    The critic's sigma head is the scale of the return-to-go distribution and
    is what kappa(alpha) trades against. Earlier proxies made sigma mean the
    wrong thing:
      - Pure bootstrap sigma_t = gamma * sigma_next collapses to its only fixed point, 0.
      - sigma_t = |TD error| measures prediction confidence, which shrinks to 0
        for a consistently-profitable concentrated asset. The critic then
        labels that bet "low risk", so the high-kappa (low-alpha) policy piles
        into it -> low alpha becomes aggressive and alpha sensitivity inverts.
    Fix: use the env's Markowitz portfolio return std sqrt(w^T * sigma * w)
    (already computed from the historical covariance window, exposed as
    info['portfolio risk']) as a per-step risk source, bootstrapped properly:
    sigma_t = sqrt(sigma_step^2 + gamma^2 * sigma_next^2)
    sigma_step is positively correlated with risk: concentrated/volatile -> high,
    diversified -> low, cash -> 0. A high-return low-vol asset gets LOW sigma.
    High kappa (low alpha) therefore minimizes portfolio vol -> diversifies -> conservative.
    Correct polarity, and no cash collapse (diversified portfolios have low sigma and positive mu).
    sigma_step is stored per transition in the replay buffer (the `risks` field).

Change 6): Critic is not alpha-conditioned
    The critic's (mu, sigma) targets are alpha-independent (objective return and
    Markowitz risk of an action), so FiLM-conditioning the critic on alpha only
    let it learn a spurious alpha-dependence that competed with the explicit
    kappa(alpha) tilt and won when kappa * sigma was weak, flipping the sensitivity direction
    (observed when sigma_scale was halved). So now, only the actor is alpha-conditioned.
    Now all alpha-sensitivity flows through kappa(alpha): direction is guaranteed
    and sigma_scale is a clean magnitude knob.

Change 7): (kappa - kappa_min) relative-risk-aversion
    Actor loss: L = -E[mu - (kappa(alpha) - kappa_min) * sigma] - lambda_e * H(pi),
    kappa_min = kappa(alpha=0.50).
    Subtracting kappa_min zeroes the risk penalty at the most risk-tolerant alpha
    (pure mu-max, fully invested) while keeping the full penalty at small alpha,
    spreading the ends apart (alpha=5% -> cash, alpha=50% -> invested) rather than sliding
    the whole frontier toward cash.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
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

from eval.eval_test import TestDistributionalDDPG
from utils.config import Config
from utils.utils import estimate_t_dof, kappa_of, tensor
from wrappers.wrappers import env_wrapper

_random.seed(0)
np.random.seed(0)  # noqa: NPY002 — seed the global legacy RNG the code uses
torch.manual_seed(0)
print(f"[seed] deterministic run, SEED={0}")

gym.logger.setLevel(logging.INFO)

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--eval-only",
    action="store_true",
    help="Skip training; load saved checkpoint and run evaluation.",
)
args = _parser.parse_args()
matplotlib.rc("figure", figsize=[15, 10])

window = 15
root = os.getcwd()
steps = 128

ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H-%M-%S")
tag = "ddpg-t-dist-film-v2-" + ts
print("tensorboard --logdir runs/" + tag)

try:
    configure("runs/" + tag)
except ValueError as e:
    print(e)


# 1): MLE estimate of Student-t degrees of freedom
print("\n" + "=" * 60)
print("STEP 1: Estimating Student-t degrees of freedom (nu)")
print("=" * 60)
nu_hat = estimate_t_dof(data_dir="/data", train_fraction=0.8)
print(f"\n>>> nu_hat = {nu_hat:.4f}\n")


# Change 7): (kappa − kappa_min) reparametrization.
# Training/eval is when alpha is in [0.05, 0.50].
# The most risk-tolerant level is alpha=0.50, where
# kappa is at its minimum. Subtracting kappa_min makes the actor loss at alpha=0.50
# a pure mu-maximizer (zero risk penalty, so it's fully invested and aggressive)
# while alpha=0.05 keeps the full (kappa(0.05) − kappa(0.50)) penalty.
# This speards the ends apart (5% is mostly cash, while 50% is mostly invested)
# instead of merely shifting the whole frontier toward cash, which
# is what raising sigma_scale alone would do.
# Direction is unaffected: kappa − kappa_min
# is still decreasing in alpha and non-negative over the range.
ALPHA_TRAIN_MAX = 0.50
KAPPA_MIN = float(kappa_of(ALPHA_TRAIN_MAX, nu_hat))
print(
    f">>> kappa(alpha=0.05)={kappa_of(0.05, nu_hat):.3f}  kappa(alpha=0.50)=kappa_min={KAPPA_MIN:.3f}  "
    f"-> relative penalty range [0, {kappa_of(0.05, nu_hat) - KAPPA_MIN:.3f}]\n"
)

path_data = "/data/poloniex_fc.hf"
df_train_full = pd.read_hdf(path_data, key="train", encoding="utf-8")
df_test = pd.read_hdf(path_data, key="test", encoding="utf-8")

# Held-out validation set for checkpoint selection.
# Selecting on in-sample data would pick the most overfit model. Selecting on
# df_test would be leakage that invalidates the reported test number, so we
# carve a chronological tail off the training data and the model trains on
# df_train (inner 80%) and is selected on df_val (last 20%), which it never
# trains on. df_test remains untouched until the final evaluation.
_val_split = int(len(df_train_full) * 0.8)
df_train = df_train_full.iloc[:_val_split]
df_val = df_train_full.iloc[_val_split:]
print(
    f"Train/val split: {len(df_train)} train rows, {len(df_val)} val rows "
    f"(held out); {len(df_test)} test rows."
)


class DDPGAgent(mp.Process):
    """
    Implements distributional DDPG.

    Attributes:
        config (Config): Container for all configurations.
        task (DeepRLWrapper): Wrapped portfolio environment.
        worker_network (DisjointActorCriticNet): Worker network containing
            an actor and a critic.
        target_network (DisjointActorCriticNet): Target network containing
            an actor and a critic.
        actor_opt (torch.optim.Adam): Optimizer for the worker actor network.
        critic_opt (torch.optim.Adam): Optimizer for the worker critic network.
        replay (ReplayAlphaStratified): Replay buffer.
        random_process (OrnsteinUhlenbeckProcess): Ornstein-Uhlenbeck process.
        total_steps (int): Number of steps taken so far.
        actor (DeterministicActorNetCVaR): Worker actor network.
        critic (DeterministicCriticNetCVaR): Worker critic network.
        target_actor (DeterministicActorNetCVaR): Target worker actor network.
        target_critic (DeterministicCriticNetCVaR): Target worker critic network.
        error (float): Epsilon for safe numerical calculations.
    """

    def __init__(self, config: Config):
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

    def soft_update(self, target: nn.Module, src: nn.Module):
        """
        Implemenets a soft update using polyak averaging.
        The target network's parameters are updated using the
        parameters of the critic network.

        Args:
            target (nn.Module): Target network.
            critic (nn.Module): Critic network.
        """
        mix = self.config.target_network_mix
        for tp, p in zip(target.parameters(), src.parameters()):
            tp.detach_()
            tp.copy_(tp * (1.0 - mix) + p * mix)

    def save(self, file_name: str):
        """
        Saves the worker network (both actor and critic) to a file.

        Args:
            file_name (str): File to save.
        """
        with open(file_name, "wb") as f:
            torch.save(self.worker_network.state_dict(), f)

    def episode(self, deterministic: bool = False) -> tuple:
        """
        Rolls out one episode.

        Args:
            deterministic (bool = False): True, if the episode should
                be deterministic (used for testing).

        Returns:
            tuple[int, int]: Total reward achieved in the episode
                and the number of steps taken.
        """
        self.random_process.reset_states()
        state = self.task.reset()
        config = self.config
        steps_ep = 0
        total_reward = 0.0

        # Change 2: training alpha restricted to evaluation range [0.05, 0.50]
        alpha = np.random.uniform(0.05, 0.50)  # noqa: NPY002

        while True:
            self.actor.eval()
            action = self.actor.predict(np.stack([state]), np.stack([alpha])).flatten()
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
                # Also add the Markowitz portfolio std V = sqrt(w^T \sigma w) to the
                # replay buffer. V^2 is the variance of the portfolio return, so it
                # is small for diversified portfolios and large for concentrated (risky) ones.
                risk = float(np.sqrt(max(info["portfolio risk"], 0.0))) * config.sigma_scale
                self.replay.feed([state, action, reward, risk, next_state, alpha, int(done)])
                self.total_steps += 1

            steps_ep += 1
            state = next_state
            if done:
                break

            if not deterministic and self.replay.size() >= config.min_memory_size:
                self._update(config)

        return total_reward, steps_ep

    def _update(self, config: Config):
        """
        Updates the networks using the experiences collected during previous episodes.

        Args:
            config (Config): Container for all configurations.
        """
        experiences = self.replay.sample()
        states, actions, rewards, risks, next_states, alphas, terminals = experiences
        states = tensor(states)
        actions = tensor(actions)
        rewards = tensor(rewards).unsqueeze(-1)
        risks = tensor(risks).unsqueeze(-1)
        alphas = tensor(alphas).unsqueeze(-1)
        mask = tensor(1 - terminals).unsqueeze(-1)
        next_states = tensor(next_states)

        # critic: W_2 loss between t(\nu, \mu_t, \sigma_t) and t(\nu, \mu_p, \sigma_p)
        q_next = self.target_critic.predict(
            next_states, alphas, self.target_actor.predict(next_states, alphas)
        )
        mu_next = q_next[:, 0].unsqueeze(-1)
        sigma_next = q_next[:, 1].unsqueeze(-1)
        mu_t = (config.discount * mu_next * mask + rewards + self.error).detach()  # target mu

        q = self.critic.predict(states, alphas, actions)
        mu_p = q[:, 0].unsqueeze(-1) + self.error
        sigma_p = q[:, 1].unsqueeze(-1) + self.error

        # The target sigma is now \sigma_t = sqrt(\sigma_{step}^2 + \gamma^2 \sigma_{next}^2)
        # instead of \sigma_t = \gamma \sigma_{next}.
        # σ_step is the env's Markowitz portfolio return std sqrt(wᵀΣw) for
        # this transition (stored in "risks").
        # This new target is needed because:
        #   - The \sigma_{step}^2 source term prevents \sigma_t to fall below the portfolio's
        #     risk, which stabilizes learning.
        #   - The original target had its only fixed point at \sigma = 0, a degenerate solution.
        # The new target \sigma_t can be shown to be a correct bootstrap for the 2nd moment,
        # instead of the first (note that \mu_t is a correct boostrap for the 1st moment).
        sigma_t = torch.sqrt(
            risks**2 + (config.discount * sigma_next * mask) ** 2 + self.error
        ).detach()

        # W_2 Wassterstein distance between two t-distributed random variables
        # with the same degrees of freedom \nu: W_2^2 = (\mu_t − \mu_p)^2 + \nu/(\nu−2)*(\sigma_t − \sigma_p)^2
        nu_factor = nu_hat / (nu_hat - 2.0)
        critic_loss = torch.pow(mu_t - mu_p, 2) + nu_factor * torch.pow(sigma_t - sigma_p, 2)
        cl = critic_loss.mean()

        self.critic.zero_grad()
        self.critic_opt.zero_grad()
        cl.backward()
        grad_critic = nn.utils.clip_grad_norm_(self.critic.parameters(), config.gradient_clip)
        self.critic_opt.step()

        # actor: t-CVaR gradient
        # Maximize E[\hat{\mu} − (\kappa(\alpha, \nu) − \kappa_min)\hat{\sigma}]
        # \kappa(\alpha, \nu) = f_\nu(q_\alpha)*(\nu + q_\alpha^2) / (\alpha*(\nu−1))
        # \kappa_min = \kappa(\alpha=0.50, \nu).
        self.actor.train()
        Actions = self.actor.predict(states, alphas, to_numpy=False)
        score = self.critic.predict(states, alphas, Actions)
        Amu = score[:, 0].unsqueeze(-1) + self.error
        Asigma = score[:, 1].unsqueeze(-1) + self.error

        alphas_np = alphas.detach().numpy()
        # \kappa − \kappa_min: zero penalty at the most risk-tolerant \alpha,
        # full penalty at the most risk-averse \alpha.
        kappa_rel = np.clip(kappa_of(alphas_np, nu_hat) - KAPPA_MIN, 0.0, None)
        kappa_t = tensor(kappa_rel)
        cvar_loss = -(Amu - kappa_t * Asigma).mean()

        # Entropy regularization. Its coefficient is fairly low, so it only
        # has a slight effect. Makes sure that the portfolio does not collapse
        # into a single asset too early.
        entropy = -torch.sum(Actions * torch.log(Actions + 1e-8), dim=1).mean()
        policy_loss = cvar_loss - config.lambda_entropy * entropy

        self.actor.zero_grad()
        self.actor_opt.zero_grad()
        policy_loss.backward()
        grad_actor = nn.utils.clip_grad_norm_(self.actor.parameters(), config.gradient_clip)
        self.actor_opt.step()

        # Log useful values to tensorboard
        max_weight = Actions.max(dim=1)[0].mean().item()
        log_value("critic_loss", cl.item(), self.total_steps)
        log_value("policy_loss", policy_loss.item(), self.total_steps)
        log_value("cvar_loss", cvar_loss.item(), self.total_steps)
        log_value("entropy", entropy.item(), self.total_steps)
        log_value("critic_mu", Amu.mean().item(), self.total_steps)
        log_value("critic_sigma", Asigma.mean().item(), self.total_steps)
        log_value("sigma_t_mean", sigma_t.mean().item(), self.total_steps)
        log_value("risk_step", risks.mean().item(), self.total_steps)
        log_value("max_weight", max_weight, self.total_steps)
        if config.gradient_clip:
            log_value("grad_critic", grad_critic, self.total_steps)
            log_value("grad_actor", grad_actor, self.total_steps)

        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

    def _step(self, state: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Given a set of states and alpha, predicts a set of actions
        using the actor network.

        Args:
            state (torch.Tensor): Batch of states.
            alpha (torch.Tensor): Set of alpha values.

        Returns:
            torch.Tensor: Predicted actions.
        """
        return self.actor.predict(np.stack([state]), np.stack([alpha])).flatten()


config = Config(
    df_train=df_train,
    df_test=df_test,
    df_val=df_val,
    window=window,
    steps=steps,
    root=root,
    tag=tag,
)
task = config.task  # reuse the env Config already built (avoids an extra RNG draw)
agent = DDPGAgent(config)
test_eval = TestDistributionalDDPG(
    window=window, df_val=df_val, n_assets=task.action_dim - 1, agent=agent, config=config
)

print("Training from random initialization.")


def run_episodes(agent: DDPGAgent) -> float:
    """
    Trains the model by running several episodes.

    Args:
        agent (DDPGAgent): Agent to train.

    Returns:
        float: Best validation accuracy during training.
    """
    config = agent.config
    ep = 0
    rewards = []
    steps_list = []
    avg_test_rew = []
    agent_type = agent.__class__.__name__

    best_val_metric = -np.inf  # best mean val accumulated return so far
    best_state = None  # state_dict of the best-on-validation model

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
                root + f"/video/{agent_type}-{config.tag}-online-stats-{agent.task.name}.bin",
                "wb",
            ) as f:
                pickle.dump([steps_list, rewards], f)

        if config.episode_limit and ep > config.episode_limit:
            break

        if config.max_steps and agent.total_steps > config.max_steps:
            break

        if config.test_interval and ep % config.test_interval == 0:
            agent.save(root + f"/video/{agent_type}-{config.tag}-model-{agent.task.name}.bin")
            test_rewards = [
                agent.episode(deterministic=True)[0] for _ in range(config.test_repetitions)
            ]
            avg_rew = np.mean(test_rewards)
            avg_test_rew.append(avg_rew)
            config.logger.info(
                f"Test avg reward {round(avg_rew, 4)} (±{round(np.std(test_rewards) / np.sqrt(config.test_repetitions), 4)})"
            )

            _diag_env = env_wrapper(
                df=df_val, steps=len(df_val) - window - 2, window_length=window, random_reset=False
            )
            _diag_state = _diag_env.reset()
            _diag_weights = {a: agent._step(_diag_state, a) for a in [0.05, 0.50]}
            print(f"\n--- ep {ep} diagnostics ---")
            print(f"  avg reward (train)    : {np.mean(rewards[-config.test_interval :]):.4f}")
            print(f"  avg reward (train-det): {avg_rew:.4f}")
            print(f"  alpha=5%  weights: {np.round(_diag_weights[0.05], 3)}")
            print(f"  alpha=50% weights: {np.round(_diag_weights[0.50], 3)}")
            print(
                f"  max_weight alpha=5% : {_diag_weights[0.05].max():.3f}  "
                f"max_weight alpha=50%: {_diag_weights[0.50].max():.3f}"
            )
            print("---")
            with open(
                root + f"/video/{agent_type}-{config.tag}-all-stats-{agent.task.name}.bin",
                "wb",
            ) as f:
                pickle.dump(
                    {"rewards": rewards, "steps": steps_list, "test_rewards": avg_test_rew}, f
                )

        # Validation-based checkpoint selection on held-out df_val.
        # The final model is the one that generalized best here.
        if config.val_interval and ep % config.val_interval == 0:
            val_metric, val_accs = test_eval.validate(agent)
            log_value("val_return", val_metric, agent.total_steps)
            improved = val_metric > best_val_metric
            print(
                f"  [val] ep {ep} mean acc.return {val_metric:.4%}  "
                f"per-alpha {['%.2f%%' % (r * 100) for r in val_accs]}"
                f"{'  <- new best' if improved else ''}"
            )
            if improved:
                best_val_metric = val_metric
                best_state = copy.deepcopy(agent.worker_network.state_dict())
                agent.save(best_model_path)

    if best_state is not None:
        agent.worker_network.load_state_dict(best_state)
        print(f"\nLoaded best-on-validation model (mean val acc.return {best_val_metric:.4%}).")
    return best_val_metric


agent.task._plot = agent.task._plot2 = None

trained_model_path = (
    root + f"/video/t_dist_ddpg_film_v2_win15_{int(config.sigma_scale * steps)}_etf.pth"
)
best_model_path = (
    root + f"/video/t_dist_ddpg_film_v2_win15_{int(config.sigma_scale * steps)}_etf_best.pth"
)

if args.eval_only:
    if not os.path.exists(trained_model_path):
        print(f"[ERROR] Checkpoint not found: {trained_model_path}")
        print("Train first (omit --eval-only).")
        sys.exit(1)
    agent.worker_network.load_state_dict(torch.load(trained_model_path, map_location="cpu"))
    print(f"Loaded checkpoint: {trained_model_path}")
else:
    print("\n" + "=" * 60)
    print("TRAINING - FiLM Student-t Distributional DDPG")
    print(f"  Window   : {window}")
    print(f"  Episodes : {config.episode_limit}")
    print(f"  nu_hat   : {nu_hat:.4f}")
    print("  alpha    : Uniform(0.05, 0.50)")
    print("  selection: best mean val acc.return on held-out df_val")
    print("=" * 60 + "\n")

    try:
        run_episodes(agent)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    torch.save(agent.worker_network.state_dict(), trained_model_path)
    print(f"\nBest-on-validation model saved to: {trained_model_path}")

test_eval.diagnostic()
