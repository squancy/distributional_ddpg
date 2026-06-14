from __future__ import annotations

import logging
from typing import Any

import gym
import gym.spaces
import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)
eps = 1e-8


class DataSrc:
    """
    Acts as data provider for each new episode.

    Attributes:
        steps (int): Number of steps in each episode.
        random_reset (bool = True): True, if the starting positing should be set randomly in the window.
            Otherwise, the starting position will be the first element in the window.
        scale (bool = True): True, if data should be scaled for each episode.
        scale_extra_cols (bool = False): True, if non-price columns should be scaled as well.
        window_length (int = 50): Memory window length.
        idx (int): Index of the current position in the dataframe.
        asset_names (list[str]): List of assets in the dataframe.
        features (list[str]): List of features in the dataframe.
        data (Any): Data extracted from the dataframe of dimension (time, asset_names, features).
        _data (Any): Data transposed so it has dimension (asset_names, time, features).
        _times (pandas.DataFrame.index): Indicies (timestamps) of the current window.
        price_columns (list[str]): Close, High, Low and Open.
        non_price_columns (set): Set of all columns not in `price_columns`.
        step (int): Current step in the episode.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        steps: int,
        scale: bool = True,
        scale_extra_cols: bool = False,
        window_length: int = 50,
        random_reset: bool = True,
    ):
        self.steps = steps + 1
        self.random_reset = random_reset
        self.scale = scale
        self.scale_extra_cols = scale_extra_cols
        self.window_length = window_length
        self.idx = self.window_length
        self.asset_names = df.columns.levels[0].tolist()
        self.features = df.columns.levels[1].tolist()

        # data = (time, asset_names, features)
        data = df.values.reshape((len(df), len(self.asset_names), len(self.features)))

        # _data = (asset_names, time, features)
        self._data = np.transpose(data, (1, 0, 2))

        self._times = df.index
        self.price_columns = ["Close", "High", "Low", "Open"]
        self.non_price_columns = set(df.columns.levels[1]) - set(self.price_columns)

        if scale_extra_cols:
            x = self._data.reshape((-1, len(self.features)))
            self.stats = dict(mean=x.mean(0), std=x.std(0))
        self.reset()

    def _step(self) -> tuple[np.array, np.array, bool, np.array]:
        """
        Steps the environment by one.
        Extracts the current data window, calculates returns and optionally scales the data window
        and the extra columns.

        Returns:
            tuple[np.array, np.array, bool, np.array]: Tuple of history (extracted data window),
                returns, boolean indicating whether the episode has ended and the set
                of closing prices for the window.
        """
        self.step += 1
        data_window = self.data[:, self.step : self.step + self.window_length, :].copy()

        # features = (Open, High, Low, Close, Adj Close, Volume)
        cprice = self.data[:, self.step : self.step + self.window_length, 3]

        # y1 should be the ('cash', 'stock1', 'stock2', 'stock3', ...)
        y1 = data_window[:, -1, 3] / data_window[:, -2, 3]
        y1 = np.concatenate([[1.0], y1])  # add cash price

        nb_pc = len(self.price_columns)
        if self.scale:
            last_close_price = data_window[:, -1, 3]
            data_window[:, :, :nb_pc] /= last_close_price[:, np.newaxis, np.newaxis]

        if self.scale_extra_cols:
            data_window[:, :, nb_pc:] -= self.stats["mean"][None, None, nb_pc:]
            data_window[:, :, nb_pc:] /= self.stats["std"][None, None, nb_pc:]
            data_window[:, :, nb_pc:] = np.clip(
                data_window[:, :, nb_pc:],
                self.stats["mean"][nb_pc:] - self.stats["std"][nb_pc:] * 10,
                self.stats["mean"][nb_pc:] + self.stats["std"][nb_pc:] * 10,
            )
        history = data_window
        done = bool(self.step >= self.steps)
        return history, y1, done, cprice

    def reset(self):
        """
        Resets the environment. If `self.random_reset` is `True`, the current position is set randomly.
        Otherwise, the current position will be the first greater than the window length.
        Also sets `self.data` and `self.times` accordingly.
        """
        self.step = 0
        if self.random_reset:
            # noqa: NPY002 — global legacy RNG kept on purpose to stay
            # stream-equivalent with the reference implementation (seedable via
            # np.random.seed); do NOT "modernize" to np.random.default_rng().
            self.idx = np.random.randint(  # noqa: NPY002
                low=self.window_length + 1, high=self._data.shape[1] - self.steps - 2
            )
        else:
            self.idx = self.window_length + 1
        self.data = self._data[
            :, self.idx - self.window_length : self.idx + self.steps + 1, :
        ].copy()
        self.times = self._times[self.idx - self.window_length : self.idx + self.steps + 1]


class PortfolioSim:
    """
    Portfolio management simulation.

    Attributes:
        cost (float): Trading cost.
        time_cost (float): Cost of holding.
        steps (int): Number of steps in each episode.
        asset_names (list[str]): List of asset names.
        utility (str): Utility function to use as reward. One of "Log", "Power", "Exp", "MV".
        gamma (float): Risk aversion parameter used in utility functions.
        alpha (float): Alpha used in CVaR.
    """

    def __init__(
        self,
        steps: int,
        asset_names: list[str] | None = None,
        trading_cost: float = 0.0025,
        time_cost: float = 0.0000,
        utility: str = "Log",
        gamma: float = 1,
        alpha: float = 0.05,
    ):
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.asset_names = asset_names if not None else []
        self.utility = utility
        self.gamma = gamma
        self.alpha = alpha
        self.reset()

    def assets_historical_returns_and_covariances(
        self, prices: np.array
    ) -> tuple[np.array, np.array]:
        """
        Calculate discrete returns and covariances of prices.

        Args:
            prices (np.array): Two dimensional price array of dimensions (assets, time).

        Returns:
            tuple[np.array, np.array]: Tuple of returns and covariances.
        """
        returns = prices[:, 1:] / prices[:, :-1] - 1
        expreturns = np.array([np.mean(returns[r]) for r in range(prices.shape[0])])
        covars = np.cov(returns)
        return expreturns, covars

    def _step(
        self, w1: np.array, y1: np.array, cprice: np.array
    ) -> tuple[np.array, np.array, bool]:
        """
        Step the environment by implementing the equations in Section 3 (portfolio optimization).
        Based on [Wang and Ku, 2022](https://doi.org/10.1016/j.eswa.2022.116807).

        Args:
            w1 (np.array): Array of new portfolio weights.
            y1 (np.array): Price relative vector (returns).
            cprice (np.array): Two dimensional price array of dimensions (assets, time).

        Returns:
            tuple[np.array, np.array, bool]: Tuple of reward, info and boolean indicating whether the episode has ended.
        """
        w0 = self.w0
        p0 = self.p0
        d50 = cprice
        ret50, cor50 = self.assets_historical_returns_and_covariances(d50)
        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # Eq. (3.11)
        c1 = (
            self.cost * (np.abs(dw1[1:] - w1[1:])).sum()
        )  # Eq. (3.12), excluding cash to avoid double counting transaction costs
        p1 = p0 * (1 - c1) * np.dot(y1, w0)  # Eq. (3.15)
        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding
        protfolio_risk = np.dot(w1[1:].T, np.dot(cor50, w1[1:]))  # Eq. (3.9)
        mw = np.array([1 / len(self.asset_names)] * len(self.asset_names))
        market_risk = np.dot(
            mw.T, np.dot(cor50, mw)
        )  # same as portfolio risk but with an equally weighted portfolio
        p1 = np.clip(
            p1, 0, np.inf
        )  # final portfolio value must be non-negative (no short sells are allowed)
        rho1 = p1 / p0 - 1  # Eq. (3.14), rate of returns

        mu_cvar = np.dot(w1[1:], ret50)
        sigma_cvar = np.sqrt(np.dot(w1[1:].T, np.dot(cor50, w1[1:])))

        CVaR_n = self.alpha**-1 * norm.pdf(norm.ppf(self.alpha)) * sigma_cvar - mu_cvar
        if self.utility == "Log":  # log rate of return
            r1 = np.log(p1 / p0)
            reward = r1 * 10000 / self.steps
        elif self.utility == "Power":  # CRRA power utility function
            r1 = ((p1 / p0) ** self.gamma - 1) / self.gamma
            reward = r1 * 10000 / self.steps
        elif self.utility == "Exp":  # exponential utility function
            r1 = (1 - np.exp(-self.gamma * p1 / p0)) / self.gamma
            reward = r1 * 10000 / self.steps
        elif self.utility == "MV":
            r_50 = np.dot(w1[1:], ret50) - self.gamma * np.dot(w1[1:].T, np.dot(cor50, w1[1:]))
            r1 = np.log(p1 / p0)
            reward = r_50 * 100 / self.steps
        else:
            raise Exception(f"Invalid value for utility: {self.utility}")

        self.w0 = w1
        self.p0 = p1

        done = bool(p1 == 0)

        info = {
            "reward": reward,
            "log-return": r1,
            "portfolio_value": p1,
            "market_return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": c1,
            "portfolio risk": protfolio_risk,
            "market risk": market_risk,
            "CVaR": CVaR_n,
        }

        for i, name in enumerate(["Cash"] + self.asset_names):
            info["weight_" + name] = w1[i]
            info["price_" + name] = y1[i]
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        """
        Sets initial portfolio weight vector to all cash,
        initial portfolio value to one and clears logs.
        """
        self.infos = []
        self.w0 = np.array([1] + [0] * len(self.asset_names))
        self.p0 = 1


class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059).

    Attributes:
        src (DataSrc): Data provider instance.
        output_mode (str): Decides the shape of history. One of:
            - "EIIE": (assets, window, 3)
            - "atari": (window, window, 3) and assets is padded
            - "mlp": (assets * window * 3)
        sim (PortfolioSim): Portfolio simulation environment instance.
        action_space (gym.Spaces.Box): Action space as an OpenAI gym attribute.
            Actions are portfolio vectors, i.e., the coefficients of a convex combination.
        observation_space (gym.Spaces.Dict): Observation space as an OpenAI gym attribute.
            Keys are 'history' and 'weights'.
    """

    metadata = {"render.modes": ["notebook", "humman"]}

    def __init__(
        self,
        df,
        steps,
        trading_cost=0.0025,
        time_cost=0.00,
        window_length=50,
        utility="Log",
        gamma=2,
        output_mode="EIIE",
        scale=True,
        scale_extra_cols=False,
        random_reset=True,
    ):
        self.src = DataSrc(
            df=df,
            steps=steps,
            scale=scale,
            scale_extra_cols=scale_extra_cols,
            window_length=window_length,
            random_reset=random_reset,
        )
        nb_assets = len(self.src.asset_names)
        self.output_mode = output_mode
        self.sim = PortfolioSim(
            asset_names=self.src.asset_names,
            utility=utility,
            gamma=gamma,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps,
        )
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(nb_assets + 1,))

        if output_mode == "EIIE":
            obs_shape = (nb_assets, window_length, len(self.src.features))
        elif output_mode == "mlp":
            obs_shape = (nb_assets) * window_length * (len(self.src.features))
        else:
            raise Exception(f"Invalid value for output_mode: {self.output_mode}")

        self.observation_space = gym.spaces.Dict(
            {"history": gym.spaces.Box(-np.inf, np.inf, obs_shape), "weights": self.action_space}
        )
        self.reset()

    def step(self, action: np.array) -> tuple[dict[str, Any], np.array, bool, dict[str, Any]]:
        """
        Step the environment.

        Args:
            action (np.array): Portfolio weights.

        Returns:
            tuple[dict[str: Any]]: Dictionary of history and weights, rewards, boolean indicating
                the end of episode and logs.
        """
        np.testing.assert_almost_equal(action.shape, (len(self.sim.asset_names) + 1,))
        action_take = np.clip(action, 0, 1)
        logger.debug("action: %s", action_take)
        weights = action_take
        weights = weights / (weights.sum() + eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)

        # Sanity checks
        assert self.action_space.contains(weights), (
            f"action should be within {self.action_space} but is {weights}"
        )

        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, 3, err_msg=f'weights should sum to 1. action="{weights}"'
        )

        history, y1, done1, cprice = self.src._step()
        reward, info, done2 = self.sim._step(weights, y1, cprice)

        info["market_value"] = np.cumprod([inf["market_return"] for inf in self.infos + [info]])[-1]
        info["date"] = self.src.times[self.src.step].timestamp()
        info["steps"] = self.src.step
        self.infos.append(info)

        if self.output_mode == "EIIE":
            pass
        elif self.output_mode == "atari":
            padding = history.shape[1] - history.shape[0]
            history = np.pad(history, [[0, padding], [0, 0], [0, 0]], mode="constant")
        elif self.output_mode == "mlp":
            history = history.flatten()
        return {"history": history, "weights": weights}, reward, done1 or done2, info

    def reset(self) -> np.array:
        """
        Resets the portfolio simulator, clears logs, sets action to initial
        portfolio weights (all cash) and then steps the environment once.

        Returns:
            np.array: Observation of the first step after resetting.
        """
        self.sim.reset()
        self.src.reset()
        self.infos = []
        action = self.sim.w0
        observation, _, _, _ = self.step(action)
        return observation


if __name__ == "__main__":
    # Example usage
    df_train = pd.read_hdf("data/poloniex_fc.hf", key="train")
    df_test = pd.read_hdf("data/poloniex_fc.hf", key="test")

    df = PortfolioEnv(
        df=df_train,
        steps=256,
        scale=True,
        scale_extra_cols=True,
        window_length=50,
        random_reset=True,
    )
