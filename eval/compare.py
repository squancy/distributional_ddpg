"""
Combined model comparison on the held-out test split.

It compares, on poloniex_fc.hf (key='test'), at alpha in {0.05, 0.15, 0.30, 0.50}:
  1. T-dist fixed-alpha models: one checkpoint per alpha, each trained at its
     own window (5, 10, 15, 10 for alpha 5/15/30/50%, matching the paper).
     Evaluated at its own alpha only.

  2. T-dist alpha-sensitive (FiLM) models: one checkpoint each, differing in
     sigma_scale (25k, 20k, 15k). A single net is evaluated at ALL alphas,
     tracing a risk-return frontier.

  3. Paper's Normal-DDPG fixed-alpha models: one checkpoint per alpha.  The
     architecture variant is dictated by each saved checkpoint: Type A (no alpha
     input, no `out` layer) for alpha 5%, Type B (late-alpha injection, has an
     `out` layer) for alpha 15/30/50%.  Evaluated at its own alpha.

  4. Fixed-strategy baselines: 1/N, all-cash, and the best single
     buy-and-hold asset (test-set oracle reference).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from Environment.AlphaEnv import PortfolioEnv
from eval import frontier  # leaf module: numpy/scipy only (Markowitz + t-CVaR optimizers)
from network.network import DeterministicActorNetCVaR as FilmActor
from network.network_fixed import DeterministicActorNetCVaR as LateAlphaActor
from utils.utils import MDD, sharpe
from wrappers.wrappers import DeepRLWrapper, env_wrapper

logging.basicConfig(level=logging.WARNING)

DATA_PATH = "/data/poloniex_fc.hf"
ALPHAS = [0.05, 0.15, 0.30, 0.50]
# Dense grid for the alpha-sensitive sweep: a fine set of risk levels in
# [0.05, 0.50] (step 0.025) that includes the four canonical alphas above and
# adds 15 intermediate ones.  Only the FiLM models -- one network conditioned on
# alpha -- can be queried at these off-grid levels; it traces the continuous
# risk-return frontier the discretely-trained fixed/Normal models only sample.
DENSE_ALPHAS = [round(float(a), 3) for a in np.linspace(0.05, 0.50, 19)]
CVAR_LEVEL = 0.05  # fixed tail fraction for realized-CVaR reporting
TEST_STEPS = 500
BASELINE_WINDOW = 15  # window for the alpha-independent fixed-weight baselines

# --- classical-benchmark / efficient-frontier settings (added) --------------
# Causal covariance windows for the rolling Markowitz / t-CVaR benchmarks:
# L=15 is *matched* to the RL state window (the fairest apples-to-apples), and
# L=60 is a best-practice, better-conditioned estimate (disclosed as such).
MV_LOOKBACK_MATCHED = 15
MV_LOOKBACK_BEST = 60
SHRINKAGE = "lw"  # Ledoit-Wolf shrinkage for frontier/benchmark Sigma
_NU_CACHE: dict = {}

CHECKPOINT_DIR = os.path.join(os.getcwd(), "video")

OUTPUT_DIR = os.path.join(os.getcwd(), "results")

# arch "A" = no alpha input, "B" = late-alpha injection, "F" = FiLM.
# Fixed-alpha families: {alpha: (checkpoint_candidates, window, arch)}.
FIXED_FAMILIES = {
    "t_dist_fixed": {
        "label": "T-dist fixed-alpha",
        "color": "tomato",
        "linestyle": "--",
        "alphas": {
            0.05: (["t_dist_ddpg_fixed_alpha05_win5_etf.pth"], 5, "B"),
            0.15: (["t_dist_ddpg_fixed_alpha15_win10_etf.pth"], 10, "B"),
            0.30: (["t_dist_ddpg_fixed_alpha30_win15_etf.pth"], 15, "B"),
            0.50: (["t_dist_ddpg_fixed_alpha50_win10_etf.pth"], 10, "B"),
        },
    },
    "normal": {
        "label": "Normal-DDPG (paper, fixed-alpha)",
        "color": "royalblue",
        "linestyle": "-.",
        "alphas": {
            0.05: (["distributional_ddpg_cvar_win10_05_etf.pth"], 5, "A"),
            0.15: (["distributional_ddpg_cvar_win10_15_etf.pth"], 10, "B"),
            0.30: (["distributional_ddpg_cvar_win10_301_etf.pth"], 15, "B"),
            # NOTE: this checkpoint actually contains an `out` layer (Type B)
            # loading it as Type A would silently drop `out.*` and mis-evaluate.
            0.50: (["distributional_ddpg_cvar_win1050_etf.pth"], 10, "B"),
        },
    },
}

# Alpha-sensitive FiLM models: one checkpoint each, evaluated at all alphas.
FILM_MODELS = [
    {
        "key": "film_25k",
        "label": "T-dist FiLM (sigma_scale=25k)",
        "checkpoints": ["t_dist_ddpg_film_v2_win15_25000_etf_best.pth"],
        "window": 15,
        "color": "seagreen",
    },
    {
        "key": "film_20k",
        "label": "T-dist FiLM (sigma_scale=20k)",
        "checkpoints": ["t_dist_ddpg_film_v2_win15_20000_etf.pth"],
        "window": 15,
        "color": "mediumseagreen",
    },
    {
        "key": "film_15k",
        "label": "T-dist FiLM (sigma_scale=15k)",
        "checkpoints": ["t_dist_ddpg_film_v2_win15_15000_etf.pth"],
        "window": 15,
        "color": "darkgreen",
    },
]

BASELINE_STYLE = {
    "equal": ("equal-weight 1/N", "grey", "D"),
    "cash": ("all-cash", "black", "P"),
    "best": ("best asset (test-oracle)", "darkorange", "*"),
}


class NoAlphaActor(nn.Module):
    """
    Paper Normal-DDPG actor with no alpha input and no output layer
    (the alpha=5% and alpha=50% checkpoints). Inference only.

    Attributes:
        conv1 (nn.Conv2d): 1st convolutional layer.
        conv2 (nn.Conv2d): 2nd convolutional layer.
        conv3 (nn.Conv2d): 3rd convolutional layer.
    """

    def __init__(self, state_dim: np.array, action_dim: int):
        super().__init__()
        stride_time = state_dim[1] - 1 - 2
        features = state_dim[0]
        h1, h2 = 32, 16
        self.conv1 = nn.Conv2d(features, h2, (3, 1))
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.conv3 = nn.Conv2d(h1 + 1, 1, (1, 1))

    @staticmethod
    def _to_var(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.from_numpy(np.asarray(x, dtype="float32"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_var(x)
        w0 = x[:, :1, :1, :]
        x = x[:, :, 1:, :]
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = torch.cat([h, w0], 1)
        action = self.conv3(h)
        cash_bias = torch.zeros(action.size()[0], 1, 1, 1)
        action = torch.cat([cash_bias, action], -1)
        return F.softmax(action.view(action.size()[0], -1), dim=1)

    def predict(self, x: torch.Tensor, to_numpy: bool = True):
        y = self.forward(x)
        return y.cpu().data.numpy() if to_numpy else y


class InferenceAgent:
    """
    Actor wrapper. `_step` adapts to whether the actor consumes alpha.

    Attributes:
        actor (nn.Module): The loaded actor network.
        arch (str): Architecture tag ("A", "B" or "F").
    """

    def __init__(self, actor: nn.Module, arch: str):
        self.actor = actor
        self.arch = arch
        self.actor.eval()

    def _step(self, state: np.array, alpha: float) -> np.array:
        """
        Returns the portfolio weights for one state at the given alpha.

        Args:
            state (np.array): State.
            alpha (float): Alpha value.

        Returns:
            np.array: Portfolio allocation.
        """
        if self.arch == "A":
            action = self.actor.predict(np.stack([state]))
        else:
            action = self.actor.predict(np.stack([state]), np.stack([alpha]))
        return action.flatten()


# Helpers
def find_checkpoint(names: list[str]) -> str | None:
    """
    Returns the first existing path for any of `names`.

    Args:
        names (list[str]): List of filenames.

    Returns:
        str | None: First existing path or `None`.
    """
    for n in names:
        p = os.path.join(CHECKPOINT_DIR, n)
        if os.path.exists(p):
            return p
    return None


def save_fig(fig, filename: str) -> str:
    """
    Saves `fig` to the output dir and reports the absolute path.

    Args:
        fig: Matplotlib figure.
        filename (str): Filename.

    Returns:
        str: Path where the figure has been saved.
    """
    saved = None
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if os.path.exists(path):
            saved = path
            print(f"  saved: {path}  ({os.path.getsize(path) // 1024} KB)")
    except OSError as e:
        print(f"  [skip] could not write to {OUTPUT_DIR}: {e}")
    if not saved:
        print(f"  [WARN] {filename} was not written to ANY location.")
    return saved


def make_test_env(df_test: pd.DataFrame, window: int) -> DeepRLWrapper:
    """
    Returns a deterministic test environment at the given window.

    Args:
        df_test (pd.DataFrame): Test dataframe.
        window (int): Trading window length.
    """
    return env_wrapper(df=df_test, steps=TEST_STEPS, window_length=window, random_reset=False)


def build_agent(df_test: pd.DataFrame, window: int, ckpt_path: str, arch: str):
    """
    Build the test env and the actor for `arch`, load the actor weights, and
    wrap it in an InferenceAgent.

    Args:
        df_test (pd.DataFrame): Test dataframe.
        window (int): Trading window length.
        ckpt_path (str): Path to the saved checkpoint.
        arch (str): Architecture (one of "A", "F", "B").

    Returns:
        tuple[InferenceAgent, DeepRLWrapper]: The agent and a fresh env.
    """
    env = make_test_env(df_test, window)
    if arch == "A":
        actor = NoAlphaActor(env.state_dim, env.action_dim)
    elif arch == "F":
        actor = FilmActor(state_dim=env.state_dim, task=env, action_gate=None, action_scale=1.0)
    elif arch == "B":
        actor = LateAlphaActor(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            task=env,
            action_gate=None,
            action_scale=1.0,
        )
    else:
        raise ValueError(f"unknown arch {arch!r}")

    sd = torch.load(ckpt_path, map_location="cpu")
    actor_sd = sd[0] if isinstance(sd, (list, tuple)) else sd
    missing, unexpected = actor.load_state_dict(actor_sd, strict=False)
    if missing:
        raise RuntimeError(f"actor missing keys in {os.path.basename(ckpt_path)}: {missing}")
    if unexpected:
        print(f"  [info] ignoring unused checkpoint keys: {unexpected}")
    actor.eval()
    return InferenceAgent(actor, arch), env


def _info_df(env: PortfolioEnv) -> pd.DataFrame:
    """
    Converts the infos in the environment into a dataframe.

    Args:
        env (PortfolioEnv): Environment.

    Returns:
        pd.DataFrame: Dataframe.
    """
    df = pd.DataFrame(env.unwrapped.infos)
    df.index = pd.to_datetime(df["date"] * 1e9)
    return df


def evaluate_policy(agent: InferenceAgent, env: PortfolioEnv, alpha: float) -> pd.DataFrame:
    """
    Roll the deterministic policy out on `env` at the given alpha.

    Args:
        agent (InferenceAgent): DDPG agent.
        env (PortfolioEnv): Portfolio environment.
        alpha (float): Alpha level.

    Returns:
        pd.DataFrame: Dataframe of infos of the environment after it has been exhausted.
    """
    state = env.reset()
    done = False
    while not done:
        state, _, done, _ = env.step(agent._step(state, alpha))
    return _info_df(env)


def evaluate_fixed_weights(env: PortfolioEnv, weights: np.array) -> pd.DataFrame:
    """
    Roll out a constant portfolio on `env`.

    Args:
        env (PortfolioEnv): Portfolio environment.
        weights (np.array): Portfolio vector weights.

    Returns:
        pd.DataFrame: Dataframe of infos of the environment after it has been exhausted.
    """
    weights = np.asarray(weights, dtype=np.float32)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(weights)
    return _info_df(env)


def cvar(returns: np.array, level: float = CVAR_LEVEL) -> float:
    """
    Calculates expected shortfall: mean of the worst `level`-fraction of step returns.

    Args:
        returns (np.array): Array of returns.
        level (float = CVAR_LEVEL): CVaR level.

    Returns:
        float: Expected shortfall.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return float("nan")
    q = np.quantile(returns, level)
    tail = returns[returns <= q]
    return float(tail.mean()) if tail.size else float(q)


def metrics(df: pd.DataFrame) -> dict:
    """
    Calculates AR / MDD / CVaR@5% / Sharpe / Calmar of one rollout.

    Args:
        df (pd.DataFrame): Dataframe containing portfolio value.

    Returns:
        dict: Dictionary containing the metrics.
    """
    pv = df["portfolio_value"]
    rets = df["rate_of_return"].values
    acc = float(pv.iloc[-1] - 1.0)
    mdd = MDD(pv.values)
    return dict(
        AR=acc,
        MDD=mdd,
        CVaR=cvar(rets),
        SR=sharpe(rets),
        Calmar=(acc / mdd if mdd > 1e-12 else float("nan")),
    )


# Eval
def evaluate_fixed_alpha_families(df_test: pd.DataFrame, results: dict):
    """
    Evaluate each fixed-alpha family at its own alpha.

    Args:
        df_test (pd.DataFrame): Test dataframe.
        results (dict): Dictionary in which the results are saved.
    """
    for family, spec in FIXED_FAMILIES.items():
        print("\n" + "=" * 64)
        print(f"{spec['label']}  (one checkpoint per alpha)")
        print("=" * 64)
        for alpha in ALPHAS:
            names, window, arch = spec["alphas"][alpha]
            ckpt = find_checkpoint(names)
            if ckpt is None:
                print(f"  alpha={alpha:.0%}  [SKIP] missing {names[0]}")
                results[(family, alpha)] = None
                continue
            agent, env = build_agent(df_test, window, ckpt, arch)
            df = evaluate_policy(agent, env, alpha)
            m = metrics(df)
            results[(family, alpha)] = {"df": df, "window": window, **m}
            print(
                f"  alpha={alpha:.0%}  win={window}  arch={arch}  AR={m['AR']:+.2%}  "
                f"MDD={m['MDD']:.2%}  CVaR={m['CVaR']:+.3%}  SR={m['SR']:.3f}"
            )


def evaluate_film_models(df_test: pd.DataFrame, results: dict):
    """
    Evaluate each alpha-sensitive FiLM model at ALL alphas.

    Args:
        df_test (pd.DataFrame): Test dataframe.
        results (dict): Dictionary in which the results are saved.
    """
    print("\n" + "=" * 64)
    print("T-dist FiLM (alpha-sensitive)  -  one checkpoint, all alphas")
    print("=" * 64)
    for model in FILM_MODELS:
        ckpt = find_checkpoint(model["checkpoints"])
        if ckpt is None:
            print(f"\n  {model['label']}  [SKIP] missing {model['checkpoints'][0]}")
            for alpha in ALPHAS:
                results[(model["key"], alpha)] = None
            continue
        print(f"\n  {model['label']}  ckpt={os.path.basename(ckpt)}  win={model['window']}")
        agent, _ = build_agent(df_test, model["window"], ckpt, "F")
        for alpha in ALPHAS:
            env = make_test_env(df_test, model["window"])
            df = evaluate_policy(agent, env, alpha)
            m = metrics(df)
            results[(model["key"], alpha)] = {"df": df, "window": model["window"], **m}
            print(
                f"    α={alpha:.0%}  AR={m['AR']:+.2%}  MDD={m['MDD']:.2%}  "
                f"CVaR={m['CVaR']:+.3%}  SR={m['SR']:.3f}"
            )


def run_baselines(df_test: pd.DataFrame, results: dict) -> int:
    """
    Evaluates fixed strategy baselines: equal-weight, all-cash, and best single asset.

    Args:
        df_test (pd.DataFrame): Test dataframe.
        results (dict): Dictionary in which the results are saved.

    Returns:
        int: Index of the best performing asset.
    """
    print("\n" + "=" * 64)
    print("Fixed-strategy baselines")
    print("=" * 64)
    env = make_test_env(df_test, BASELINE_WINDOW)
    n_assets = env.action_dim - 1

    equal_w = np.array([0.0] + [1.0 / n_assets] * n_assets)
    cash_w = np.array([1.0] + [0.0] * n_assets)

    results["equal"] = {
        "df": evaluate_fixed_weights(make_test_env(df_test, BASELINE_WINDOW), equal_w)
    }
    results["equal"].update(metrics(results["equal"]["df"]))
    results["cash"] = {
        "df": evaluate_fixed_weights(make_test_env(df_test, BASELINE_WINDOW), cash_w)
    }
    results["cash"].update(metrics(results["cash"]["df"]))

    best_idx, best_ar, best_df = 1, -np.inf, None
    for i in range(1, n_assets + 1):
        w = np.zeros(n_assets + 1)
        w[i] = 1.0
        dfb = evaluate_fixed_weights(make_test_env(df_test, BASELINE_WINDOW), w)
        ar = float(dfb["portfolio_value"].iloc[-1] - 1.0)
        if ar > best_ar:
            best_ar, best_idx, best_df = ar, i, dfb
    results["best"] = {"df": best_df, "idx": best_idx}
    results["best"].update(metrics(best_df))

    for key in ("equal", "cash", "best"):
        label = BASELINE_STYLE[key][0]
        m = results[key]
        print(
            f"  {label:<28} AR={m['AR']:+.2%}  MDD={m['MDD']:.2%}  "
            f"CVaR={m['CVaR']:+.3%}  SR={m['SR']:.3f}"
        )
    return best_idx


# Reporting
def _model_rows() -> list[tuple]:
    """
    (key, label) pairs for every learned model, in display order.

    Returns:
        list[tuple]: List of key-label pairs.
    """
    rows = [(f, spec["label"]) for f, spec in FIXED_FAMILIES.items()]
    rows += [(m["key"], m["label"]) for m in FILM_MODELS]
    return rows


def print_summary(results: dict, best_idx: int):
    """
    Prints per-alpha metrics table for all learned models, then the baselines.

    Args:
        results (dict): Dictionary of results.
        best_idx (int): Index of the best performing asset.
    """
    print("\n" + "=" * 88)
    print("RESULTS SUMMARY (test split)")
    print("=" * 88)
    print(
        f"{'Model':<32}{'alpha':>6}{'win':>5}{'AR':>10}{'MDD':>9}"
        f"{'CVaR@5%':>10}{'SR':>8}{'Calmar':>8}"
    )
    print("-" * 88)
    for alpha in ALPHAS:
        for key, label in _model_rows():
            r = results.get((key, alpha))
            if not r:
                continue
            print(
                f"{label:<32}{alpha:>6.0%}{r['window']:>5}{r['AR']:>+10.2%}"
                f"{r['MDD']:>9.2%}{r['CVaR']:>+10.3%}{r['SR']:>8.3f}{r['Calmar']:>8.2f}"
            )
        print("-" * 88)
    for key in ("equal", "cash", "best"):
        label = BASELINE_STYLE[key][0]
        r = results[key]
        print(
            f"{label:<32}{'-':>6}{'-':>5}{r['AR']:>+10.2%}"
            f"{r['MDD']:>9.2%}{r['CVaR']:>+10.3%}{r['SR']:>8.3f}{r['Calmar']:>8.2f}"
        )
    print("=" * 88)
    print(
        "AR = accumulated return   MDD = max drawdown   "
        f"CVaR@5% = realized expected shortfall   SR = Sharpe   best asset #{best_idx}"
    )


def plot_value_curves(results: dict):
    """
    One subplot per alpha: every model evaluated at that alpha + baselines.

    Args:
        results (dict): Dictionary of results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, alpha in enumerate(ALPHAS):
        ax = axes[idx]

        # fixed-alpha families (the model trained at this alpha)
        for family, spec in FIXED_FAMILIES.items():
            r = results.get((family, alpha))
            if r:
                ax.plot(
                    r["df"].index,
                    r["df"]["portfolio_value"],
                    color=spec["color"],
                    linestyle=spec["linestyle"],
                    lw=1.6,
                    label=f"{spec['label']} (win={r['window']})  AR={r['AR']:+.1%}, SR={r['SR']:.2f}",
                )

        # alpha-sensitive FiLM models evaluated at this alpha
        for model in FILM_MODELS:
            r = results.get((model["key"], alpha))
            if r:
                ax.plot(
                    r["df"].index,
                    r["df"]["portfolio_value"],
                    color=model["color"],
                    lw=1.8,
                    label=f"{model['label']}  AR={r['AR']:+.1%}, SR={r['SR']:.2f}",
                )

        # baselines (alpha-independent)
        ax.plot(
            results["equal"]["df"].index,
            results["equal"]["df"]["portfolio_value"],
            color="grey",
            lw=1.0,
            alpha=0.8,
            label=f"equal-weight 1/N  AR={results['equal']['AR']:+.1%}",
        )
        ax.plot(
            results["best"]["df"].index,
            results["best"]["df"]["portfolio_value"],
            color="darkorange",
            lw=1.0,
            linestyle=":",
            alpha=0.9,
            label=f"best asset #{results['best']['idx']} (oracle)  AR={results['best']['AR']:+.1%}",
        )
        ax.axhline(
            1.0,
            color="black",
            lw=0.6,
            alpha=0.4,
            label=f"all-cash  AR={results['cash']['AR']:+.1%}",
        )

        ax.set_ylabel("Portfolio value (start = 1.0)")
        ax.legend(fontsize=7.0, loc="upper left")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    print("\nFigure 1 (portfolio-value curves):")
    save_fig(fig, "comparison_value_curves.png")
    plt.close(fig)


def plot_frontier(results: dict):
    """
    Risk-return frontier (AR vs MDD).

    Args:
        results (dict): Dictionary of results.
    """
    fig, ax = plt.subplots(figsize=(11, 8))

    # alpha-sensitive FiLM models: a connected frontier across alphas
    for model in FILM_MODELS:
        present = [a for a in ALPHAS if results.get((model["key"], a))]
        if not present:
            continue
        pts = [
            (results[(model["key"], a)]["MDD"], results[(model["key"], a)]["AR"]) for a in present
        ]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=model["color"], lw=2, ms=8, label=model["label"])
        for a, (x, y) in zip(present, pts):
            ax.annotate(f" {a:.0%}", (x, y), color=model["color"], fontsize=8, va="center")

    # fixed-alpha families: discrete points (one per alpha)
    for family, spec in FIXED_FAMILIES.items():
        present = [a for a in ALPHAS if results.get((family, a))]
        if not present:
            continue
        pts = [(results[(family, a)]["MDD"], results[(family, a)]["AR"]) for a in present]
        xs, ys = zip(*pts)
        ax.plot(
            xs, ys, spec["linestyle"] + "s", color=spec["color"], lw=1.6, ms=8, label=spec["label"]
        )
        for a, (x, y) in zip(present, pts):
            ax.annotate(f" {a:.0%}", (x, y), color=spec["color"], fontsize=8, va="center")

    # baselines as standalone markers
    for key in ("equal", "cash", "best"):
        label, color, marker = BASELINE_STYLE[key]
        r = results[key]
        ax.scatter([r["MDD"]], [r["AR"]], color=color, s=140, marker=marker, zorder=5, label=label)

    ax.set_xlabel("Max drawdown", fontsize=12)
    ax.set_ylabel("Accumulated return", fontsize=12)
    ax.axhline(0.0, color="black", lw=0.5, alpha=0.4)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    print("\nFigure 2 (risk-return frontier):")
    save_fig(fig, "comparison_frontier.png")
    plt.close(fig)


# Dense alpha sweep (alpha-sensitive models only)
def _monotone_fraction(values: list[float], rising: bool = True) -> float:
    """
    Fraction of adjacent pairs that respect the expected ordering -- a quick
    smoothness/monotonicity score for a metric swept over alpha.

    Args:
        values (list[float]): Metric values in alpha order.
        rising (bool = True): Whether the metric is expected to rise with alpha.

    Returns:
        float: Fraction of adjacent pairs in the expected order (1.0 = monotone).
    """
    if len(values) < 2:
        return float("nan")
    ok = sum((b >= a - 1e-12) if rising else (b <= a + 1e-12) for a, b in zip(values, values[1:]))
    return ok / (len(values) - 1)


def evaluate_film_dense(df_test: pd.DataFrame, dense_results: dict):
    """
    Evaluate each alpha-sensitive FiLM model on the DENSE_ALPHAS grid (not just
    the four canonical levels).

    Stores, per (model_key, alpha), the usual metrics plus the time-averaged
    portfolio allocation (`mean_w`) used by the allocation plot.

    Args:
        df_test (pd.DataFrame): Test dataframe.
        dense_results (dict): Dictionary in which the results are saved, keyed by
            (model_key, alpha).
    """
    print("\n" + "=" * 64)
    print(
        f"T-dist FiLM dense alpha sweep  -  {len(DENSE_ALPHAS)} levels in "
        f"[{DENSE_ALPHAS[0]:.3f}, {DENSE_ALPHAS[-1]:.3f}]"
    )
    print("=" * 64)
    for model in FILM_MODELS:
        ckpt = find_checkpoint(model["checkpoints"])
        if ckpt is None:
            print(f"\n  {model['label']}  [SKIP] missing {model['checkpoints'][0]}")
            for alpha in DENSE_ALPHAS:
                dense_results[(model["key"], alpha)] = None
            continue
        print(f"\n  {model['label']}  ckpt={os.path.basename(ckpt)}")
        agent, _ = build_agent(df_test, model["window"], ckpt, "F")
        for alpha in DENSE_ALPHAS:
            env = make_test_env(df_test, model["window"])
            df = evaluate_policy(agent, env, float(alpha))
            m = metrics(df)
            wcols = [c for c in df.columns if c.startswith("weight_")]
            dense_results[(model["key"], alpha)] = {
                **m,
                "mean_w": df[wcols].mean(),
                "wcols": wcols,
            }
        rows = [dense_results[(model["key"], a)] for a in DENSE_ALPHAS]
        print(
            f"    monotone-rising fraction over alpha:  "
            f"AR={_monotone_fraction([r['AR'] for r in rows], True):.0%}  "
            f"MDD={_monotone_fraction([r['MDD'] for r in rows], True):.0%}  "
            f"CVaR={_monotone_fraction([r['CVaR'] for r in rows], False):.0%}"
        )


def plot_dense_frontier(results: dict, dense_results: dict):
    """
    Continuous risk-return frontier traced by the alpha-sensitive net: each FiLM
    model's dense (MDD, AR) points, connected and colour-graded by alpha, with
    the four-point fixed-alpha / Normal models and the baselines overlaid as
    reference markers.

    Args:
        results (dict): Canonical-alpha results (for the reference points).
        dense_results (dict): Dense-alpha FiLM results.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    norm = plt.Normalize(DENSE_ALPHAS[0], DENSE_ALPHAS[-1])
    sc = None
    for model in FILM_MODELS:
        rows = [(a, dense_results.get((model["key"], a))) for a in DENSE_ALPHAS]
        rows = [(a, r) for a, r in rows if r]
        if not rows:
            continue
        alphas = [a for a, _ in rows]
        xs = [r["MDD"] for _, r in rows]
        ys = [r["AR"] for _, r in rows]
        ax.plot(xs, ys, "-", color=model["color"], lw=1.2, alpha=0.5, zorder=2)
        sc = ax.scatter(
            xs,
            ys,
            c=alphas,
            cmap="viridis",
            norm=norm,
            s=45,
            zorder=3,
            edgecolor=model["color"],
            linewidth=1.0,
        )
        ax.annotate(
            model["label"],
            (xs[-1], ys[-1]),
            fontsize=8,
            color=model["color"],
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
        )
    if sc is not None:
        fig.colorbar(sc, ax=ax).set_label("alpha (risk tolerance)", fontsize=11)

    for family, spec in FIXED_FAMILIES.items():
        present = [a for a in ALPHAS if results.get((family, a))]
        if not present:
            continue
        xs = [results[(family, a)]["MDD"] for a in present]
        ys = [results[(family, a)]["AR"] for a in present]
        ax.scatter(
            xs,
            ys,
            marker="s",
            facecolor="none",
            edgecolor=spec["color"],
            s=90,
            lw=1.6,
            zorder=4,
            label=spec["label"],
        )

    for key in ("equal", "cash", "best"):
        label, color, marker = BASELINE_STYLE[key]
        r = results[key]
        ax.scatter([r["MDD"]], [r["AR"]], color=color, s=150, marker=marker, zorder=5, label=label)

    ax.set_xlabel("Max drawdown", fontsize=12)
    ax.set_ylabel("Accumulated return", fontsize=12)
    ax.axhline(0.0, color="black", lw=0.5, alpha=0.4)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    print("\nFigure 3 (continuous frontier, dense alpha):")
    save_fig(fig, "comparison_dense_frontier.png")
    plt.close(fig)


def plot_metrics_vs_alpha(dense_results: dict):
    """
    Each metric (AR, MDD, CVaR@5%, Sharpe) as a function of alpha for every FiLM
    model on the dense grid.

    Args:
        dense_results (dict): Dense-alpha FiLM results.
    """
    panels = [
        ("AR", "Accumulated return", True),
        ("MDD", "Max drawdown", True),
        ("CVaR", "CVaR@5% (downside)", False),
        ("SR", "Sharpe ratio", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (mkey, mlabel, _) in zip(axes.flatten(), panels):
        for model in FILM_MODELS:
            rows = [(a, dense_results.get((model["key"], a))) for a in DENSE_ALPHAS]
            rows = [(a, r) for a, r in rows if r]
            if not rows:
                continue
            ax.plot(
                [a for a, _ in rows],
                [r[mkey] for _, r in rows],
                "-o",
                color=model["color"],
                ms=3,
                lw=1.5,
                label=model["label"],
            )
        for a in ALPHAS:
            ax.axvline(a, color="grey", lw=0.6, ls=":", alpha=0.6)
        ax.set_xlabel("alpha (risk tolerance)")
        ax.set_ylabel(mlabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    plt.tight_layout()
    print("\nFigure 4 (metrics vs alpha):")
    save_fig(fig, "comparison_metrics_vs_alpha.png")
    plt.close(fig)


def plot_allocation_vs_alpha(dense_results: dict):
    """
    Stacked-area of the time-averaged portfolio allocation vs alpha, one panel per
    FiLM model.

    Args:
        dense_results (dict): Dense-alpha FiLM results.
    """
    fig, axes = plt.subplots(1, len(FILM_MODELS), figsize=(6 * len(FILM_MODELS), 6), squeeze=False)
    for ax, model in zip(axes[0], FILM_MODELS):
        rows = [(a, dense_results.get((model["key"], a))) for a in DENSE_ALPHAS]
        rows = [(a, r) for a, r in rows if r]
        if not rows:
            ax.set_visible(False)
            continue
        alphas = [a for a, _ in rows]
        wcols = rows[0][1]["wcols"]
        weights = np.array([[r["mean_w"][c] for c in wcols] for _, r in rows])  # [n_alpha, n_asset]
        labels = [c.replace("weight_", "") for c in wcols]
        ax.stackplot(alphas, weights.T, labels=labels, alpha=0.85)
        ax.set_xlim(alphas[0], alphas[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel("alpha (risk tolerance)")
        ax.set_ylabel("time-averaged weight")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    plt.tight_layout()
    print("\nFigure 5 (allocation vs alpha):")
    save_fig(fig, "comparison_allocation_vs_alpha.png")
    plt.close(fig)


def _nu_hat() -> float:
    """
    Estimate nu for the Student-t distribution using historical data.

    Returns:
        float: Estimated nu.
    """
    if "nu" not in _NU_CACHE:
        from utils.utils import estimate_t_dof

        repo_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        for d in ("/data", repo_data):
            try:
                _NU_CACHE["nu"] = float(estimate_t_dof(data_dir=d, train_fraction=0.8))
                break
            except (FileNotFoundError, ValueError):
                continue
        if "nu" not in _NU_CACHE:
            raise FileNotFoundError(
                "could not locate ETF CSVs to estimate nu (need /data or ./data)"
            )
    return _NU_CACHE["nu"]


def realized_sigma_mu(df: pd.DataFrame) -> tuple[float, float]:
    """
    Per-step realized volatility and mean of a rollout's net returns.

    Args:
        df (pd.DataFrame): Rollout dataframe with a `rate_of_return` column.

    Returns:
        tuple[float, float]: (realized_vol, realized_mean).
    """
    r = df["rate_of_return"].values
    return float(np.std(r)), float(np.mean(r))


def evaluate_dynamic_weights(env: PortfolioEnv, weight_fn) -> pd.DataFrame:
    """
    Roll out a time-varying weight policy on `env`. This is causal.

    Args:
        env (PortfolioEnv): Portfolio environment.
        weight_fn (callable): `(src, state) -> weights`.

    Returns:
        pd.DataFrame: Rollout info dataframe.
    """
    state = env.reset()
    done = False
    while not done:
        src = env.unwrapped.src
        weights = np.asarray(weight_fn(src, state), dtype=np.float32)
        state, _, done, _ = env.step(weights)
    return _info_df(env)


def _rolling_weight_fn(
    kind: str,
    lookback: int,
    nu: float | None = None,
    alpha: float | None = None,
    relative: bool = False,
):
    """
    Build a causal `weight_fn` for `evaluate_dynamic_weights` that re-estimates
    mu, Sigma on the trailing `lookback` closes each step and solves the chosen
    classical optimizer.

    Args:
        kind (str): One of "tangency", "gmv", "tcvar".
        lookback (int): Trailing covariance window length.
        nu (float | None): Student-t dof (for "tcvar").
        alpha (float | None): CVaR level (for "tcvar").
        relative (bool): Use the actor-matched relative tilt (for "tcvar").

    Returns:
        callable: `(src, state) -> full weight vector`.
    """

    def fn(src, state) -> np.ndarray:
        end = src.step + src.window_length  # realising close is at index `end`
        start = max(0, end - lookback)
        closes = src.data[:, start:end, 3]  # (n_risky, <=lookback), all < realising close
        n = closes.shape[0]
        cash_only = np.array([1.0] + [0.0] * n)
        if closes.shape[1] < 3:  # not enough history yet -> stay in cash
            return cash_only
        mu, sigma = frontier.estimate_mu_sigma(closes, shrinkage=SHRINKAGE)
        if kind == "tcvar":
            return frontier.t_cvar_weights(mu, sigma, alpha, nu, relative=relative)
        if kind == "gmv":
            return np.concatenate([[0.0], frontier.min_variance_portfolio(sigma)])
        if kind == "tangency":
            w = frontier.tangency_portfolio(mu, sigma, rf=0.0)
            return cash_only if w is None else np.concatenate([[0.0], w])
        raise ValueError(f"unknown kind {kind!r}")

    return fn


def _close_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Close-price matrix (n_assets, T) from a MultiIndex (asset, feature) frame,
    in the env's asset order.

    Args:
        df (pd.DataFrame): OHLCV frame.

    Returns:
        np.ndarray: Close prices of shape (n_assets, T).
    """
    return df.xs("Close", axis=1, level=1).values.T


def _score(df: pd.DataFrame, info_set: str, label: str, **extra) -> dict:
    """
    Package a rollout into a benchmark record: realized (sigma, mu), metrics, and
    disclosure fields.

    Args:
        df (pd.DataFrame): Rollout dataframe.
        info_set (str): "causal", "trained", or "oracle".
        label (str): Human-readable label.
        **extra: Extra fields (e.g. protocol disclosure).

    Returns:
        dict: Benchmark record.
    """
    s, m_ = realized_sigma_mu(df)
    rec = {"df": df, "info_set": info_set, "label": label, "sigma": s, "mret": m_}
    rec.update(metrics(df))
    rec.update(extra)
    return rec


def run_classical_benchmarks(df_test: pd.DataFrame, df_train: pd.DataFrame, bench: dict) -> dict:
    """
    Evaluate the Markowitz and parametric-t mean-CVaR benchmarks in each
    information set (causal / trained-once / oracle), all scored through the same
    env and cost model as the RL policies.

    Args:
        df_test (pd.DataFrame): Test frame.
        df_train (pd.DataFrame): Training frame (for the trained-once benchmark).
        bench (dict): Dictionary in which the benchmark records are saved.

    Returns:
        dict: `bench`, populated. Also stores the oracle/trained frontier curves
            under bench["_frontier_oracle"], bench["_frontier_trained"].
    """
    nu = _nu_hat()
    win = BASELINE_WINDOW
    print("\n" + "=" * 64)
    print(f"Classical benchmarks (nu={nu:.2f}, shrinkage={SHRINKAGE!r})")
    print("=" * 64)

    # static estimates: oracle (look-ahead) & trained-once
    mu_or, sig_or = frontier.estimate_mu_sigma(_close_matrix(df_test), shrinkage=SHRINKAGE)
    mu_tr, sig_tr = frontier.estimate_mu_sigma(_close_matrix(df_train), shrinkage=SHRINKAGE)
    bench["_frontier_oracle"] = {
        "curve": frontier.markowitz_frontier(mu_or, sig_or),
        "cml": frontier.capital_market_line(mu_or, sig_or, rf=0.0),
    }
    bench["_frontier_trained"] = {
        "curve": frontier.markowitz_frontier(mu_tr, sig_tr),
        "cml": frontier.capital_market_line(mu_tr, sig_tr, rf=0.0),
    }

    def _fixed(mu, sig, kind, n):
        cash = np.array([1.0] + [0.0] * n)
        if kind == "gmv":
            return np.concatenate([[0.0], frontier.min_variance_portfolio(sig)])
        w = frontier.tangency_portfolio(mu, sig, rf=0.0)
        return cash if w is None else np.concatenate([[0.0], w])

    n_assets = len(df_test.columns.levels[0])

    # Oracle & trained static Markowitz points (tangency, GMV).
    for tag, mu, sig, iset in (
        ("oracle", mu_or, sig_or, "oracle"),
        ("trained", mu_tr, sig_tr, "trained"),
    ):
        for kind, name in (("tangency", "Markowitz tangency"), ("gmv", "Markowitz GMV")):
            w = _fixed(mu, sig, kind, n_assets)
            df = evaluate_fixed_weights(make_test_env(df_test, win), w)
            suffix = " (oracle, look-ahead)" if iset == "oracle" else " (trained-once)"
            bench[f"mv_{kind}_{tag}"] = _score(
                df, iset, name + suffix, estimator=f"whole-{tag} mu,Sigma"
            )
            print(
                f"  {name + suffix:<40} AR={bench[f'mv_{kind}_{tag}']['AR']:+.2%}  "
                f"MDD={bench[f'mv_{kind}_{tag}']['MDD']:.2%}"
            )

    # Oracle & trained static t-CVaR@alpha families (relative + absolute).
    for tag, mu, sig, iset in (
        ("oracle", mu_or, sig_or, "oracle"),
        ("trained", mu_tr, sig_tr, "trained"),
    ):
        for rel in (True, False):
            fam = frontier.t_cvar_family(mu, sig, nu, ALPHAS, relative=rel)
            vlab = "rel" if rel else "abs"
            for alpha, w in fam.items():
                df = evaluate_fixed_weights(make_test_env(df_test, win), w)
                lab = f"t-CVaR@a {'(rel)' if rel else '(abs)'} {tag}"
                bench[(f"tcvar_{tag}_{vlab}", alpha)] = _score(df, iset, lab, alpha=alpha)

    # causal rolling Markowitz
    for L, ltag in ((MV_LOOKBACK_MATCHED, "L15"), (MV_LOOKBACK_BEST, "L60")):
        for kind, name in (("tangency", "Markowitz tangency"), ("gmv", "Markowitz GMV")):
            df = evaluate_dynamic_weights(make_test_env(df_test, win), _rolling_weight_fn(kind, L))
            key = f"mv_{kind}_causal{ltag}"
            bench[key] = _score(df, "causal", f"{name} (causal, {ltag})", estimator=f"trailing-{L}")
            print(
                f"  {name + f' (causal {ltag})':<40} AR={bench[key]['AR']:+.2%}  "
                f"MDD={bench[key]['MDD']:.2%}"
            )

    # causal rolling t-CVaR@alpha
    for alpha in ALPHAS:
        df = evaluate_dynamic_weights(
            make_test_env(df_test, win),
            _rolling_weight_fn("tcvar", MV_LOOKBACK_MATCHED, nu=nu, alpha=alpha, relative=True),
        )
        bench[("tcvar_causalL15_rel", alpha)] = _score(
            df, "causal", "t-CVaR@a (rel, causal L15)", alpha=alpha, estimator="trailing-15"
        )
        m = bench[("tcvar_causalL15_rel", alpha)]
        print(
            f"  t-CVaR causal a={alpha:.0%}  AR={m['AR']:+.2%}  MDD={m['MDD']:.2%}  "
            f"CVaR={m['CVaR']:+.3%}"
        )

    return bench


def plot_mv_overlay(results: dict, bench: dict):
    """
    Two-panel frontier overlay:
      left: per-step (sigma, mu) space with the Markowitz efficient frontier
            curve, Capital Market Line and oracle vs trained.
            RL FiLM points, classical benchmark points, and the
            1/N / cash / best baselines.
      right: (MDD, AR) space with benchmarks overlaid.

    Args:
        results (dict): Canonical-alpha RL + baseline results (each has a "df").
        bench (dict): Classical benchmark records from run_classical_benchmarks.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 8))
    norm = plt.Normalize(ALPHAS[0], ALPHAS[-1])

    # left: efficient frontier + CML in (sigma, mu)
    fo = bench.get("_frontier_oracle", {})
    if fo.get("curve") is not None and len(fo["curve"]["vol"]):
        c = fo["curve"]
        axL.plot(
            c["vol"],
            c["ret"],
            "-",
            color="black",
            lw=1.5,
            alpha=0.6,
            label="Markowitz frontier (oracle, look-ahead)",
        )
        cml = fo.get("cml")
        if cml is not None:
            xmax = max(c["vol"].max(), 1e-6)
            axL.plot(
                [0, xmax],
                [0, cml["slope"] * xmax],
                "--",
                color="black",
                lw=1.2,
                alpha=0.6,
                label="Capital Market Line (oracle)",
            )
            axL.scatter(
                [cml["sigma_tan"]], [cml["mu_tan"]], marker="*", s=180, color="black", zorder=6
            )
    ft = bench.get("_frontier_trained", {})
    if ft.get("curve") is not None and len(ft["curve"]["vol"]):
        c = ft["curve"]
        axL.plot(
            c["vol"],
            c["ret"],
            ":",
            color="dimgrey",
            lw=1.3,
            alpha=0.7,
            label="Markowitz frontier (trained-once)",
        )

    sc = None
    for model in FILM_MODELS:
        pts = [(a, results.get((model["key"], a))) for a in ALPHAS]
        pts = [(a, r) for a, r in pts if r]
        if not pts:
            continue
        xs = [realized_sigma_mu(r["df"])[0] for _, r in pts]
        ys = [realized_sigma_mu(r["df"])[1] for _, r in pts]
        axL.plot(xs, ys, "-", color=model["color"], lw=1.0, alpha=0.5, zorder=3)
        sc = axL.scatter(
            xs,
            ys,
            c=[a for a, _ in pts],
            cmap="viridis",
            norm=norm,
            s=55,
            zorder=4,
            edgecolor=model["color"],
            linewidth=1.0,
            label=model["label"],
        )
    # causal t-CVaR benchmark points
    tcv = [(a, bench.get(("tcvar_causalL15_rel", a))) for a in ALPHAS]
    tcv = [(a, r) for a, r in tcv if r]
    if tcv:
        axL.scatter(
            [r["sigma"] for _, r in tcv],
            [r["mret"] for _, r in tcv],
            marker="D",
            facecolor="none",
            edgecolor="crimson",
            s=70,
            lw=1.6,
            zorder=5,
            label="t-CVaR@a (causal L15, fair)",
        )
    for key, style in (
        ("mv_tangency_causalL15", ("^", "purple", "MV tangency (causal L15)")),
        ("mv_tangency_oracle", ("^", "black", "MV tangency (oracle)")),
    ):
        r = bench.get(key)
        if r:
            axL.scatter(
                [r["sigma"]],
                [r["mret"]],
                marker=style[0],
                color=style[1],
                s=90,
                zorder=6,
                label=style[2],
            )
    for key in ("equal", "cash", "best"):
        if key in results:
            label, color, marker = BASELINE_STYLE[key]
            s, m_ = realized_sigma_mu(results[key]["df"])
            axL.scatter([s], [m_], color=color, s=130, marker=marker, zorder=5, label=label)
    if sc is not None:
        fig.colorbar(sc, ax=axL).set_label("alpha (risk tolerance)")
    axL.set_xlabel("Per-step realized volatility  std(return)")
    axL.set_ylabel("Per-step realized mean return")
    axL.axhline(0.0, color="grey", lw=0.5, alpha=0.4)
    axL.set_title(
        "Efficient frontier & CML vs learned / classical policies\n"
        "(cash-holding strategies lie on/inside the CML)",
        fontsize=10,
    )
    axL.grid(True, alpha=0.25)
    axL.legend(fontsize=7, loc="best")

    # right: (MDD, AR) space with benchmarks
    for model in FILM_MODELS:
        pts = [(a, results.get((model["key"], a))) for a in ALPHAS]
        pts = [(a, r) for a, r in pts if r]
        if not pts:
            continue
        axR.plot(
            [r["MDD"] for _, r in pts],
            [r["AR"] for _, r in pts],
            "-o",
            color=model["color"],
            ms=4,
            lw=1.2,
            label=model["label"],
        )
    marker_by_iset = {"causal": "D", "trained": "s", "oracle": "*"}
    seen = set()
    for key, r in bench.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        if not isinstance(r, dict) or "MDD" not in r:
            continue
        iset = r["info_set"]
        lbl = {
            "causal": "classical (causal, fair)",
            "trained": "classical (trained-once)",
            "oracle": "classical (oracle ceiling)",
        }[iset]
        axR.scatter(
            [r["MDD"]],
            [r["AR"]],
            marker=marker_by_iset[iset],
            facecolor="none" if iset != "oracle" else "gold",
            edgecolor="grey" if iset != "oracle" else "darkorange",
            s=70,
            zorder=4,
            label=lbl if lbl not in seen else None,
        )
        seen.add(lbl)
    for key in ("equal", "cash", "best"):
        if key in results:
            label, color, marker = BASELINE_STYLE[key]
            r = results[key]
            axR.scatter(
                [r["MDD"]], [r["AR"]], color=color, s=130, marker=marker, zorder=5, label=label
            )
    axR.set_xlabel("Max drawdown (lower = safer)")
    axR.set_ylabel("Accumulated return")
    axR.axhline(0.0, color="grey", lw=0.5, alpha=0.4)
    axR.set_title("Return vs drawdown: learned vs classical (by information set)", fontsize=10)
    axR.grid(True, alpha=0.25)
    axR.legend(fontsize=7, loc="best")

    plt.tight_layout()
    print("\nFigure (efficient-frontier / classical overlay):")
    save_fig(fig, "comparison_mv_overlay.png")
    plt.close(fig)


def build_results_frame(results: dict, bench: dict) -> pd.DataFrame:
    """
    Assemble the numeric results table: every RL model x alpha, the classical
    benchmarks grouped by information set, and the fixed baselines. Each row
    carries an `Info set` column so the comparison footing is never ambiguous.

    Args:
        results (dict): RL + baseline results.
        bench (dict): Classical benchmark records.

    Returns:
        pd.DataFrame: Tidy results table.
    """
    rows = []

    def _row(model, alpha, iset, r):
        rows.append(
            {
                "Model": model,
                "alpha": alpha,
                "Info set": iset,
                "AR": r["AR"],
                "MDD": r["MDD"],
                "CVaR@5%": r["CVaR"],
                "SR": r["SR"],
                "Calmar": r.get("Calmar", float("nan")),
            }
        )

    for alpha in ALPHAS:
        for key, label in _model_rows():
            r = results.get((key, alpha))
            if r:
                _row(label, f"{alpha:.0%}", "causal (RL)", r)
    # classical benchmarks
    for key, r in bench.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        if not isinstance(r, dict) or "AR" not in r:
            continue
        alpha = f"{key[1]:.0%}" if isinstance(key, tuple) else "-"
        _row(r["label"], alpha, r["info_set"], r)
    # fixed baselines
    for key in ("equal", "cash", "best"):
        if key in results:
            lbl = BASELINE_STYLE[key][0]
            _row(lbl, "-", "fixed", results[key])

    return pd.DataFrame(rows)


def write_results_tables(frame: pd.DataFrame, stem: str = "results_table"):
    """
    Write the results table to CSV (raw metrics() values).

    Args:
        frame (pd.DataFrame): Table from build_results_frame.
        stem (str): Output filename stem (in results/).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"{stem}.csv")
    frame.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}")


def write_protocol_table(bench: dict, stem: str = "eval_protocol"):
    """
    Write the disclosure legend: one row per competitor family stating its
    information set, mu/Sigma estimator + window, objective + nu, and that costs
    are paid. Every figure/table caption should reference this.

    Args:
        bench (dict): Benchmark records (for nu / presence checks).
        stem (str): Output filename stem (in results/).
    """
    nu = _NU_CACHE.get("nu", float("nan"))
    protocol = [
        (
            "T-dist FiLM (RL)",
            "causal (trailing 15)",
            "learned actor; state=15d window",
            f"E[mu - (K(a,nu)-Kmin) sigma] + entropy; nu={nu:.2f}",
            "yes",
        ),
        (
            "Markowitz tangency/GMV (causal)",
            "causal (trailing 15 & 60)",
            "sample mu,Sigma + Ledoit-Wolf",
            "max Sharpe / min variance (long-only)",
            "yes",
        ),
        (
            "t-CVaR@a (causal, fair)",
            "causal (trailing 15)",
            "sample mu,Sigma + Ledoit-Wolf",
            f"max w'mu - (K(a,nu)-Kmin) sqrt(w'Sigma w); nu={nu:.2f}",
            "yes",
        ),
        (
            "Markowitz / t-CVaR (trained-once)",
            "train period, fixed on test",
            "mu,Sigma over 2010-2018",
            "as above (static weights)",
            "yes",
        ),
        (
            "Markowitz / t-CVaR (ORACLE)",
            "whole test (LOOK-AHEAD ceiling)",
            "mu,Sigma over the test window",
            "as above -- upper bound only",
            "yes",
        ),
        (
            "1/N, all-cash, best-asset",
            "fixed / test-oracle (best)",
            "none",
            "constant weights",
            "yes",
        ),
    ]
    df = pd.DataFrame(
        protocol,
        columns=["Competitor", "Information set", "mu,Sigma estimator", "Objective", "Costs paid"],
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"{stem}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}")


def main():
    print("Loading test data ...")
    df_test = pd.read_hdf(DATA_PATH, key="test", encoding="utf-8")
    df_train = pd.read_hdf(DATA_PATH, key="train", encoding="utf-8")

    results: dict = {}
    evaluate_fixed_alpha_families(df_test, results)
    evaluate_film_models(df_test, results)
    best_idx = run_baselines(df_test, results)

    dense_results: dict = {}
    evaluate_film_dense(df_test, dense_results)

    # Classical benchmarks (Markowitz / parametric-t mean-CVaR) in every
    # information set, scored through the same env.
    bench: dict = {}
    run_classical_benchmarks(df_test, df_train, bench)

    print_summary(results, best_idx)
    plot_value_curves(results)
    plot_frontier(results)
    plot_dense_frontier(results, dense_results)
    plot_metrics_vs_alpha(dense_results)
    plot_allocation_vs_alpha(dense_results)

    # Efficient-frontier overlay + numeric result & disclosure tables.
    plot_mv_overlay(results, bench)
    write_results_tables(build_results_frame(results, bench))
    write_protocol_table(bench)

    from eval import interpret

    interpret.run_all(
        df_test=df_test,
        film_models=FILM_MODELS,
        dense_alphas=DENSE_ALPHAS,
        find_checkpoint=find_checkpoint,
        build_agent=build_agent,
        make_test_env=make_test_env,
        evaluate_policy=evaluate_policy,
        save_fig=save_fig,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
