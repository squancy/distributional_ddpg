from typing import Any

import numpy as np
import pandas as pd
import torch

from Environment.AlphaEnv import PortfolioEnv
from src.distributional_ddpg_t_dist_film_v2 import DDPGAgent
from utils.config import Config
from wrappers.wrappers import DeepRLWrapper, env_wrapper


class TestDistributionalDDPG:
    """
    Container for testing/evaluation-related subroutines.

    Attributes:
        window (int): Trading window size.
        df_val (pd.DataFrame): Validation dataframe.
        n_assets (int): Number of risky assets.
        agent (DDPGAgent): DDPG agent.
        config (Config): Configuration instance.
    """

    def __init__(
        self, window: int, df_val: pd.DataFrame, n_assets: int, agent: DDPGAgent, config: Config
    ):
        self.window = window
        self.df_val = df_val
        self.eval_alphas = [0.05, 0.15, 0.30, 0.50]
        self.cvar_level = 0.05
        self.n_assets = n_assets
        self.agent = agent
        self.config = config

    def test_algo(self, env: DeepRLWrapper, algo: DDPGAgent, alpha_eval: float) -> tuple:
        """
        Test the trained model at a given alpha level.

        Args:
            env (DeepRLWrapper): Portfolio environment.
            algo (DDPGAgent): Trained DDPG algorithm.
            alpha_eval (float): Alpha level used for evaluation.

        Returns:
            tuple: Logged portfolio values during the evaluation,
                dataframe containing several logged values.
        """
        state = env.reset()
        done = False
        while not done:
            action = algo._step(state, alpha_eval)
            state, _, done, _ = env.step(action)
        df = pd.DataFrame(env.unwrapped.infos)
        df.index = pd.to_datetime(df["date"] * 1e9)
        return df["portfolio_value"], df

    def validate(self, agent: DDPGAgent, alphas: list) -> tuple:
        """
        Deterministic rollout on the held-out validation slice (df_val) at each
        alpha.

        Args:
            agent (DDPGAgent): Model to validate.
            alphas (list): List of alpha values used for validation.

        Returns:
            tuple: Mean return across all alpha levels and list of per-alpha
                returns.
        """
        if not alphas:
            alphas = self.eval_alphas
        accs = []
        for a in alphas:
            pv, _ = self.test_algo(
                env_wrapper(
                    df=self.df_val,
                    steps=len(self.df_val) - self.window - 2,
                    window_length=self.window,
                    random_reset=False,
                ),
                agent,
                a,
            )
            accs.append(float(pv.iloc[-1] - 1.0))
        return float(np.mean(accs)), accs

    def cvar(self, returns: torch.Tensor, level: float) -> float:
        """
        Implements expected shortfall.

        Args:
            returns (torch.Tensor): List of returns.
            level (float): Alpha level.

        Returns:
            float: Expected shortfall (CVaR) calculated from the
                returns.
        """
        if not level:
            level = self.cvar_level
        returns = np.asarray(returns, dtype=float)
        if returns.size == 0:
            return float("nan")
        q = np.quantile(returns, level)
        tail = returns[returns <= q]
        return float(tail.mean()) if tail.size else float(q)

    def metrics(self, df: pd.DataFrame) -> dict:
        """
        Calculates several metrics: accumulated return,
            maximum drawdown, CVaR, Sharpe ratio and Calmar ratio.

        Args:
            df (pd.DataFrame): Dataframe containing returns.

        Returns:
            dict: Dictionary containing the metrics.
        """
        pv = df["portfolio_value"]
        rets = df["rate_of_return"].values
        acc = float(pv.iloc[-1] - 1.0)
        mdd = self.MDD(pv.values)
        return dict(
            acc=acc,
            mdd=mdd,
            cvar=self.cvar(rets),
            sharpe=self.sharpe(rets),
            calmar=(acc / mdd if mdd > 1e-12 else float("nan")),
        )

    def run_fixed_weights(env: PortfolioEnv, weights: np.array) -> pd.DataFrame:
        """
        Simulates a constant portfolio.

        Args:
            env (PortfolioEnv): Environment for the portfolio.
            weights (np.array): Portfolio allocation of size
                (number of assets + 1), to account for cash.

        Returns:
            pd.DataFrame: Dataframe containing logged metrics
                during the simulation.
        """
        weights = np.asarray(weights, dtype=np.float32)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(weights)
        df = pd.DataFrame(env.unwrapped.infos)
        df.index = pd.to_datetime(df["date"] * 1e9)
        return df

    def _row(label: str, m: dict) -> str:
        """
        Constructs a string of metrics.

        Args:
            label (str): Label to print.
            m (dict): Dictionary containing metrics
                such as MDD, CVaR, Sharpe, etc.

        Returns:
            str: String containing the metrics.
        """
        return (
            f"  {label:<18} ret {m['acc']:+8.2%}   CVaR@5% {m['cvar']:+7.3%}   "
            f"MDD {m['mdd']:6.2%}   Sharpe {m['sharpe']:6.3f}   "
            f"Calmar {m['calmar']:6.2f}"
        )

    def frontier(self, name: str, env_fn: Any) -> list:
        """
        Prints the risk-return frontier traced as alpha varies.

        Args:
            name (str): Evaluation type info (validation, test, etc.).
            env_fn (Any): Function that creates a portfolio environment
                and returns a DeepRLWrapper.

        Returns:
            list: Metrics for each alpha level.
        """
        print("\n" + "=" * 78)
        print(f"FRONTIER - {name}")
        print("=" * 78)
        ms = []
        for a in self.eval_alphas:
            _, df = self.test_algo(env_fn(), self.agent, alpha_eval=a)
            m = self.metrics(df)
            ms.append(m)
            print(self._row(f"alpha={a:.0%}", m))
        return ms

    def pick_best_asset_on_val(self) -> int:
        """
        Choose the single best buy-and-hold asset by validation return (no test
        look-ahead), to use as a reference point on both val and test.

        Returns:
            int: Index of the best risky asset.
        """
        best_i, best_acc = 1, -np.inf
        for i in range(1, self.n_assets + 1):
            w = np.zeros(self.n_assets + 1)
            w[i] = 1.0
            acc = float(
                self.run_fixed_weights(
                    env_wrapper(
                        df=self.df_val,
                        steps=len(self.df_val) - self.window - 2,
                        window_length=self.window,
                        random_reset=False,
                    ),
                    w,
                )["portfolio_value"].iloc[-1]
                - 1.0
            )
            if acc > best_acc:
                best_acc, best_i = acc, i
        return best_i

    def baselines(self, name: str, env_fn: Any, best_asset_idx: int):
        """
        Prints metrics for fixed-strategy reference points: equally weighted portfolio,
            all cash and best asset on the validation set.

        Args:
            name (str): Name of the evaluation (validation, test, etc.).
            env_fn (Any): Portfolio environment constructor.
            best_asset_idx (int): Index of the best risky asset.
        """
        print("\n" + "-" * 78)
        print(f"BASELINES - {name}")
        print("-" * 78)
        equal = np.array([0.0] + [1.0 / self.n_assets] * self.n_assets)  # 1/N over risky
        cash = np.array([1.0] + [0.0] * self.n_assets)
        best = np.zeros(self.n_assets + 1)
        best[best_asset_idx] = 1.0
        print(self._row("equal-weight 1/N", self.metrics(self.run_fixed_weights(env_fn(), equal))))
        print(self._row("all-cash", self.metrics(self.run_fixed_weights(env_fn(), cash))))
        print(
            self._row(
                f"best-asset #{best_asset_idx} (val)",
                self.metrics(self.run_fixed_weights(env_fn(), best)),
            )
        )

    def diagnostic(self):
        print("\nDiagnostic - first portfolio allocation per alpha (val env):")
        _denv = env_wrapper(
            df=self.df_val,
            steps=len(self.df_val) - self.window - 2,
            window_length=self.window,
            random_reset=False,
        )
        _dstate = _denv.reset()
        for _a in self.eval_alphas:
            print(f"  alpha={_a:.0%}  weights={np.round(self.agent._step(_dstate, _a), 3)}")

        _best_asset = self.pick_best_asset_on_val()
        print(f"\nBest single asset by validation return: #{_best_asset}")

        self.frontier("VALIDATION  (selection set, optimistic)", self.config.task_fn_vali)
        self.baselines("VALIDATION", self.config.task_fn_vali, _best_asset)

        self.frontier("TEST  (held-out, honest estimate)", self.config.task_fn_test)
        self.baselines("TEST", self.config.task_fn_test, _best_asset)
