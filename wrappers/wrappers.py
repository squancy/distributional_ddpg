from __future__ import annotations

from typing import Any

import gym.spaces
import numpy as np
import pandas as pd

from Environment.AlphaEnv import PortfolioEnv


def _concat_states(state: dict[str, Any]) -> np.array:
    """
    Concatenates history and weights into a single state.

    Args:
        state (dict[str, Any]): State containing history and weights.

    Returns:
        np.array: Single state containing both weights and history.
    """
    history = state["history"]
    weights = state["weights"]
    weight_insert_shape = (history.shape[0], 1, history.shape[2])
    if len(weights) - 1 == history.shape[0]:
        weight_insert = np.ones(weight_insert_shape) * weights[1:, np.newaxis, np.newaxis]
    elif len(weights) - 1 == history.shape[2]:
        weight_insert = np.ones(weight_insert_shape) * weights[np.newaxis, np.newaxis, 1:]
    else:
        weight_insert = np.ones(weight_insert_shape) * weights[np.newaxis, 1:, np.newaxis]
    state = np.concatenate([weight_insert, history], axis=1)
    return state


class TransposeHistory(gym.Wrapper):
    """
    Given a portfolio environment, it transposes the history matrix.

    Attributes:
        axes (tuple[int, int, int]): Axes determining the transposition.
        observation_space (gym.spaces.Dict): Dictionary representing the observation space.
            Keys are 'history' (containing the transposed history) and 'weights'.
    """

    def __init__(self, env: PortfolioEnv, axes: tuple = (2, 1, 0)):
        super().__init__(env)
        self.axes = axes

        hist_space = self.observation_space.spaces["history"]
        hist_shape = hist_space.shape
        self.observation_space = gym.spaces.Dict(
            {
                "history": gym.spaces.Box(
                    hist_space.low.min(),
                    hist_space.high.max(),
                    (hist_shape[axes[0]], hist_shape[axes[1]], hist_shape[axes[2]]),
                ),
                "weights": self.observation_space.spaces["weights"],
            }
        )

    def step(self, action: np.array) -> tuple[np.array, np.array, bool, dict[str, Any]]:
        """
        Steps the transposed environment by one.

        Args:
            action (np.array): Action to take.

        Returns:
            tuple[np.array, np.array, bool, dict[str, Any]]: Tuple of next state,
                reward, boolean indicating whether the episode has ended and info.
        """
        state, reward, done, info = self.env.step(action)
        state["history"] = np.transpose(state["history"], self.axes)
        return state, reward, done, info

    def reset(self) -> np.array:
        """
        Resets the transposed environment.

        Returns:
            np.array: State after reset.
        """
        state = self.env.reset()
        state["history"] = np.transpose(state["history"], self.axes)
        return state


class ConcatStates(gym.Wrapper):
    """
    Concatenates both state arrays for models that take a single inputs.
    Ref: https://github.com/openai/gym/blob/master/gym/wrappers/README.md

    Attributes:
        observation_space (gym.spaces.Box): Observation space.
    """

    def __init__(self, env: PortfolioEnv):
        super().__init__(env)
        hist_space = self.observation_space.spaces["history"]
        hist_shape = hist_space.shape
        self.observation_space = gym.spaces.Box(
            -np.inf, +np.inf, shape=(hist_shape[0], hist_shape[1] + 1, hist_shape[2])
        )

    def step(self, action: np.array) -> tuple[np.array, np.array, bool, dict[str, Any]]:
        """
        Steps the concatenated environment by one.

        Args:
            action (np.array): Action to take.

        Returns:
            tuple[np.array, np.array, bool, dict[str, Any]]: Tuple of next state,
                reward, boolean indicating whether the episode has ended and info.
        """
        state, reward, done, info = self.env.step(action)
        state = _concat_states(state)
        return state, reward, done, info

    def reset(self) -> np.array:
        """
        Resets the concatenated environment.

        Returns:
            np.array: State after reset.
        """
        state = self.env.reset()
        return _concat_states(state)


class DeepRLWrapper(gym.Wrapper):
    """
    Adds a few extra attributes to the environment.

    Attributes:
        render_on_reset (bool): True, if a notebook should be rendered each
            time the environment is reset.
        state_dim (tuple): Dimension of states.
        action_dim (int): Dimension of actions.
        name (str): Name of the environment.
    """

    def __init__(self, env: PortfolioEnv):
        super().__init__(env)
        self.render_on_reset = False
        self.state_dim = self.observation_space.shape
        self.action_dim = self.action_space.shape[0]
        self.name = "DDPGEnv"

    def step(self, action: np.array) -> tuple[np.array, np.array, bool, dict[str, Any]]:
        """
        Steps the environment by one.

        Args:
            action (np.array): Action to take.

        Returns:
            tuple[np.array, np.array, bool, dict[str, Any]]: Tuple of next state,
                reward, boolean indicating whether the episode has ended and info.
        """
        return self.env.step(action)

    def reset(self) -> np.array:
        """
        Resets the concatenated environment.

        Returns:
            np.array: State after reset.
        """
        return self.env.reset()


def env_wrapper(
    df: pd.DataFrame, steps: int, window_length: int, random_reset: bool = True
) -> DeepRLWrapper:
    """
    Given a portfolio environment, it transposes the history, concatenates states
    and wraps the environment in a Deep RL wrapper.

    Args:
        df (pd.DataFrame): Dataframe containing the data for the environment.
        steps (int): Number of total steps in the environment.
        window_length (int): Lookback window length.
        random_reset (bool = True): True for training (start at a random point in
            the series). Must be False for full-pass validation/test rollouts,
            where ``steps`` spans the whole slice and there is no room left for a
            random start offset.

    Returns:
        DeepRLWrapper: A wrapped portfolio environment.
    """
    env = PortfolioEnv(
        df=df,
        steps=steps,
        window_length=window_length,
        utility="Log",
        output_mode="EIIE",
        gamma=1,
        scale=True,
        scale_extra_cols=True,
        random_reset=random_reset,
    )
    env = TransposeHistory(env)
    env = ConcatStates(env)
    return DeepRLWrapper(env)
