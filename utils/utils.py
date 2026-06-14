import glob
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import t as t_dist
from tensorboardX import SummaryWriter

EPS = 1e-7


def estimate_t_dof(data_dir: str = "/data", train_fraction: float = 0.8) -> float:
    """
    Estimates the degress of freedom (nu) using MLE based on the
    combined ETF prices.

    Args:
        data_dir (str): Directory where the CSV files live.
        train_fraction (str): Fraction of all data which is used for training.
            The parameter is only estimated using the first `train_fraction * 100`%
            of data to avoid information leakage.

    Returns:
        float: Estimated degrees of freedom.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir!r}")
    all_returns = []
    for fpath in csv_files:
        df = pd.read_csv(fpath, index_col="Date", parse_dates=True).sort_index()
        if "Close" not in df.columns:
            continue
        close = df["Close"].dropna()
        log_ret = np.log(close / close.shift(1)).dropna().values
        n_train = max(int(len(log_ret) * train_fraction), 2)
        log_ret = log_ret[:n_train]
        mu_r, std_r = log_ret.mean(), log_ret.std(ddof=1)
        if std_r < 1e-12:
            continue
        all_returns.append((log_ret - mu_r) / std_r)
        print(f"  {os.path.basename(fpath)}: {n_train} training returns")
    if not all_returns:
        raise ValueError("No valid return series found.")
    pooled = np.concatenate(all_returns)
    n, x2 = len(pooled), pooled**2
    print(f"  Pooled {n} standardized returns from {len(all_returns)} assets.")

    def neg_ll(nu) -> float:
        """
        Negative log-likelihood of t-distribution with a given value of nu.

        Args:
            nu (float): Degrees of freedom.

        Returns:
            float: Negative log-likelihood evaluated at nu.
        """
        if nu <= 2.0:
            return 1e10
        return -(
            n * gammaln((nu + 1) / 2)
            - n * gammaln(nu / 2)
            - 0.5 * n * np.log(nu * np.pi)
            - 0.5 * (nu + 1) * np.sum(np.log(1 + x2 / nu))
        )

    res = minimize_scalar(neg_ll, bounds=(2.01, 100.0), method="bounded", options={"xatol": 1e-6})
    return float(res.x)


def kappa_of(alpha: float, nu: float) -> float:
    """
    Calculates kappa(alpha, nu).
    Background: t-CVaR scaling kappa(alpha, nu) = f_nu(q_alpha) * (nu + q_alpha ** 2) / (alpha * (nu - 1)),
    where f_nu is the density of t-distribution and q_alpha is the alpha-quantile function.

    Args:
        alpha (float): Risk level alpha.
        nu (float): Degrees of freedom.

    Returns:
        float: kappa(alpha, nu).
    """
    a = np.asarray(alpha, dtype=float)
    q = t_dist.ppf(a, df=nu)
    return t_dist.pdf(q, df=nu) * (nu + q**2) / (a * (nu - 1.0))


def sharpe(returns: np.array, freq: int = 30, rfr: float = 0.0) -> float:
    """
    Given a set of returns, calculates the Sharpe ratio.

    Args:
        returns (np.array): Array of returns.
        freq (int = 30): Frequency.
        rfr (float = 0.0): Risk-free return.

    Returns:
        float: Sharpe ratio.
    """
    return (np.sqrt(freq) * np.mean(returns - rfr)) / (np.std(returns - rfr) + EPS)


def MDD(X: np.array) -> float:
    """
    Given a set of returns, calculates the maximum drawdown.

    Args:
        X (np.array): List of returns.

    Returns:
        float: Maximum drawdown.
    """
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd


class RandomProcess:
    """
    Base class for random processes.
    """

    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    """
    Implements an annealed Gaussian process.

    Attributes:
        mu (float): Mean of the normal distribution.
        sigma (float): Variance of the normal distribution.
        n_steps (int): Current step in the process.
        sigma_min (float | None = None): Minimum value of sigma (defaults to sigma).
    """

    def __init__(self, mu: float, sigma: float, n_steps_annealing: int, sigma_min: float = None):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.0
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self) -> float:
        """
        Returns the current value of sigma.

        Returns:
            float: Current sigma.
        """
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    """
    Implements an Ornstein-Uhlenbeck process.

    Attributes:
        theta (float): Parameter.
        mu (float = 0.0): Mean.
        sigma (float = 1.0): Parameter.
        dt (float = 1e-2): Change in t.
        x0 (float | None = None): Initial value of the process (defaults to zero).
        size (int = 1): Dimension of the output.
        sigma_min (float | None = None): Minimum value of sigma (defaults to sigma).
        n_steps_annealing (float = 1000): Total number of steps in the process.
    """

    def __init__(
        self,
        theta: float,
        mu: float = 0.0,
        sigma: float = 1.0,
        dt: float = 1e-2,
        x0: float = None,
        size: int = 1,
        sigma_min: float = None,
        n_steps_annealing: float = 1000,
    ):
        super().__init__(
            mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing
        )
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self) -> float:
        """
        Samples from the OU process.

        Returns:
            float: Sample from the OU process.
        """
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            # noqa: NPY002 — global legacy RNG kept on purpose for stream-
            # equivalence with the reference implementation; do not modernize.
            + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)  # noqa: NPY002
        )
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        """
        Resets the process. If `x0` was given, it resets to that value and
        to zero otherwise.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class Logger:
    """
    Tensorboard logger base class.

    Attributes:
        writer (SummaryWriter): Writer to the log directory.
        info (Any): Logger for info messages.
        debug (Any): Logger for debug messages.
        skip (bool): True, if logging should be skipped.
    """

    def __init__(self, log_dir: str, vanilla_logger: Any, skip: bool = False):
        self.writer = SummaryWriter(log_dir)
        self.info = vanilla_logger.info
        self.debug = vanilla_logger.debug
        self.skip = skip

    def scalar_summary(self, tag, value, step):
        """
        Adds a scalar summary to the writer.

        Args:
            tag (Any): Tag.
            value (Any): Value to log.
            step (Any): Step.
        """
        if self.skip:
            return
        self.writer.add_scalar(tag, value, step)

    def histo_summary(self, tag, values, step):
        """
        Adds a histogram to the writer.

        Args:
            tag (Any): Tag.
            values (Any): Values to log.
            step (Any): Step.
        """
        if self.skip:
            return
        self.writer.add_histogram(tag, values, step, bins=1000)


def tensor(x: Any) -> torch.Tensor:
    """
    Converts an object to a PyTorch tensor.

    Returns:
        torch.Tensor: Converted object.
    """
    return torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
