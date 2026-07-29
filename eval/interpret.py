from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _rollout_capture(agent, env, alpha: float):
    """
    Roll the policy out, capturing the state fed to the actor at each decision
    step alongside the env's info dataframe.

    Args:
        agent: InferenceAgent with `_step(state, alpha)`.
        env: Portfolio environment.
        alpha (float): Risk level.

    Returns:
        tuple: (states list, info dataframe).
    """
    states = []
    state = env.reset()
    done = False
    while not done:
        a = agent._step(state, alpha)
        states.append(np.asarray(state, dtype=np.float32))
        state, _, done, _ = env.step(a)
    df = pd.DataFrame(env.unwrapped.infos)
    df.index = pd.to_datetime(df["date"] * 1e9)
    return states, df


def alpha_response_at_states(
    agent, env, states: dict[str, np.ndarray], alphas, asset_names: list[str]
) -> dict[str, np.ndarray]:
    """
    For each named fixed state, sweep alpha and record the actor's weight vector.

    Args:
        agent: InferenceAgent.
        env: Environment (for asset context).
        states (dict[str, np.ndarray]): Named representative states.
        alphas (Iterable[float]): Risk levels to sweep.
        asset_names (list[str]): ["Cash", *risky].

    Returns:
        dict[str, np.ndarray]: {name: weights (n_alpha, n_assets+1)}.
    """
    out = {}
    for name, st in states.items():
        w = np.array([agent._step(st, float(a)) for a in alphas])  # (n_alpha, n_assets+1)
        out[name] = w
    return out


def plot_alpha_response(resp: dict[str, np.ndarray], alphas, asset_names, save_fig):
    """
    Plot weight-vs-alpha curves, one panel per representative state.

    Args:
        resp (dict): Output of alpha_response_at_states.
        alphas (Iterable[float]): Risk levels.
        asset_names (list[str]): ["Cash", *risky].
        save_fig (callable): compare.save_fig.
    """
    alphas = list(alphas)
    fig, axes = plt.subplots(1, len(resp), figsize=(6 * len(resp), 5), squeeze=False)
    for ax, (name, w) in zip(axes[0], resp.items()):
        for j, asset in enumerate(asset_names):
            ax.plot(alphas, w[:, j], "-o", ms=3, lw=1.5, label=asset)
        ax.set_title(f"State: {name}", fontsize=10)
        ax.set_xlabel("alpha (risk tolerance)")
        ax.set_ylabel("weight")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Instantaneous policy response to alpha at fixed states (isolates the FiLM conditioning)",
        fontsize=11,
    )
    plt.tight_layout()
    print("\nFigure (state-conditioned alpha response):")
    save_fig(fig, "interpret_alpha_response.png")
    plt.close(fig)


def _select_states(states, df):
    """
    Pick representative states from a rollout: first step, a calm date, and the
    COVID-crash date (nearest available).

    Args:
        states (list[np.ndarray]): Captured states (aligned to df.iloc[1:]).
        df (pd.DataFrame): Rollout info dataframe.

    Returns:
        dict[str, np.ndarray]: Named states.
    """
    dates = df.index

    def at(target):
        pos = int(np.argmin(np.abs(dates - pd.Timestamp(target))))
        return states[max(pos - 1, 0)]

    return {
        "first test step": states[0],
        "calm (2019-06)": at("2019-06-03"),
        "COVID crash (2020-03)": at("2020-03-16"),
    }


def allocation_over_alpha_time(agent, make_test_env, evaluate_policy, df_test, window: int, alphas):
    """
    Roll the FiLM policy out at each alpha and stack the per-asset weights into an
    (alpha, time, asset) cube.

    Args:
        agent: InferenceAgent.
        make_test_env (callable): compare.make_test_env.
        evaluate_policy (callable): compare.evaluate_policy.
        df_test (pd.DataFrame): Test frame.
        window (int): Trading window.
        alphas (Iterable[float]): Risk levels.

    Returns:
        tuple: (cube (n_alpha, T, n_assets+1), dates, wcols labels).
    """
    alphas = list(alphas)
    cube, dates, wcols = [], None, None
    for a in alphas:
        env = make_test_env(df_test, window)
        df = evaluate_policy(agent, env, float(a))
        if wcols is None:
            wcols = [c for c in df.columns if c.startswith("weight_")]
            dates = df.index
        cube.append(df[wcols].values)
    return np.array(cube), dates, [c.replace("weight_", "") for c in wcols]


def plot_allocation_heatmap(cube, dates, labels, alphas, save_fig):
    """
    One heatmap per asset: weight over (alpha [rows] x time [cols]).

    Args:
        cube (np.ndarray): (n_alpha, T, n_assets+1).
        dates (pd.DatetimeIndex): Time axis.
        labels (list[str]): Asset labels.
        alphas (Iterable[float]): Risk levels.
        save_fig (callable): compare.save_fig.
    """
    alphas = list(alphas)
    n_assets = cube.shape[2]
    fig, axes = plt.subplots(n_assets, 1, figsize=(13, 2.2 * n_assets), sharex=True, squeeze=False)
    tick_pos = np.linspace(0, len(dates) - 1, 6).astype(int)
    tick_lab = [dates[i].strftime("%Y-%m") for i in tick_pos]
    for a_idx, ax in enumerate(axes[:, 0]):
        im = ax.imshow(
            cube[:, :, a_idx],
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=1,
            extent=[0, len(dates) - 1, alphas[0], alphas[-1]],
        )
        ax.set_ylabel(f"{labels[a_idx]}\nalpha", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    axes[-1, 0].set_xticks(tick_pos)
    axes[-1, 0].set_xticklabels(tick_lab)
    axes[-1, 0].set_xlabel("date")
    fig.suptitle(
        "Allocation weight over (alpha x time) -- crisis regimes visible "
        "(e.g. flight-to-cash in 2020)",
        fontsize=11,
    )
    plt.tight_layout()
    print("\nFigure (allocation heatmap alpha x time):")
    save_fig(fig, "interpret_allocation_heatmap.png")
    plt.close(fig)


def run_all(
    df_test,
    film_models,
    dense_alphas,
    find_checkpoint,
    build_agent,
    make_test_env,
    evaluate_policy,
    save_fig,
):
    """
    Run the interpretation analyses on the first available FiLM model.

    Args:
        df_test (pd.DataFrame): Test frame.
        film_models (list[dict]): FILM_MODELS spec from compare.py.
        dense_alphas (Iterable[float]): Dense alpha grid (response + heatmap).
        find_checkpoint, build_agent, make_test_env, evaluate_policy, save_fig:
            Injected helpers from compare.py.
    """
    model = next((m for m in film_models if find_checkpoint(m["checkpoints"])), None)
    if model is None:
        print("\n[interpret] no FiLM checkpoint found -- skipping interpretation analyses.")
        return
    ckpt = find_checkpoint(model["checkpoints"])
    window = model["window"]
    print("\n" + "=" * 64)
    print(f"Policy interpretation on {model['label']}  ({ckpt.split('/')[-1]})")
    print("=" * 64)

    agent, env = build_agent(df_test, window, ckpt, "F")
    asset_names = ["Cash"] + list(env.unwrapped.src.asset_names)

    # 1. state-conditioned alpha response
    states, df_probe = _rollout_capture(agent, make_test_env(df_test, window), 0.30)
    resp = alpha_response_at_states(
        agent, env, _select_states(states, df_probe), dense_alphas, asset_names
    )
    plot_alpha_response(resp, dense_alphas, asset_names, save_fig)

    # 2. allocation heatmap (alpha x time)
    cube, dates, labels = allocation_over_alpha_time(
        agent, make_test_env, evaluate_policy, df_test, window, dense_alphas
    )
    plot_allocation_heatmap(cube, dates, labels, dense_alphas, save_fig)
