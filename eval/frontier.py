from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as t_dist

EPS = 1e-12


def simple_returns(close_prices: np.ndarray) -> np.ndarray:
    """
    Per-step simple returns of a (n_assets, T) close-price matrix.

    Args:
        close_prices (np.ndarray): Close prices of shape (n_assets, T).

    Returns:
        np.ndarray: Returns of shape (n_assets, T - 1).
    """
    close_prices = np.asarray(close_prices, dtype=float)
    return close_prices[:, 1:] / close_prices[:, :-1] - 1.0


def ledoit_wolf_shrink(returns: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf estimator of the covariance matrix of returns.
    It provides a more stable alternative than the naive estimation
    which is needed because if the window size is small, the covariance
    matrix can be noisy.
    See “A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices”, Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2, February 2004, pages 365-411.

    Args:
        returns (np.ndarray): Returns of shape (n_assets, T).

    Returns:
        np.ndarray: Shrunk covariance of shape (n_assets, n_assets).
    """
    returns = np.atleast_2d(np.asarray(returns, dtype=float))
    n, t = returns.shape
    x = returns - returns.mean(axis=1, keepdims=True)
    sample = (x @ x.T) / t  # MLE covariance
    mu = np.trace(sample) / n
    target = mu * np.eye(n)

    d2 = np.sum((sample - target) ** 2)
    if d2 <= 0 or t < 2:
        return sample
    # pi-hat: mean squared deviation of the per-observation covariances.
    b2 = 0.0
    for k in range(t):
        xk = x[:, k : k + 1]
        b2 += np.sum((xk @ xk.T - sample) ** 2)
    b2 = b2 / (t**2)
    delta = float(np.clip(b2 / d2, 0.0, 1.0))
    return delta * target + (1.0 - delta) * sample


def estimate_mu_sigma(
    close_prices: np.ndarray, shrinkage: str | float | None = "lw"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the per-step mean vector and covariance of risky-asset returns.

    Args:
        close_prices (np.ndarray): Close prices of shape (n_assets, T).
        shrinkage (str | float | None): "lw" for Ledoit-Wolf, a float in [0, 1]
            for a fixed shrink toward the scaled identity, or None for the raw
            sample covariance.

    Returns:
        tuple[np.ndarray, np.ndarray]: (mu, Sigma).
    """
    rets = simple_returns(close_prices)
    mu = rets.mean(axis=1)
    if shrinkage == "lw":
        sigma = ledoit_wolf_shrink(rets)
    elif shrinkage is None:
        sigma = np.atleast_2d(np.cov(rets))
    else:
        sample = np.atleast_2d(np.cov(rets))
        n = sample.shape[0]
        target = (np.trace(sample) / n) * np.eye(n)
        delta = float(np.clip(shrinkage, 0.0, 1.0))
        sigma = delta * target + (1.0 - delta) * sample
    sigma = sigma + EPS * np.eye(sigma.shape[0])
    return mu, sigma


def _solve(fun, n: int, constraints, x0: np.ndarray | None = None) -> np.ndarray:
    """
    Minimize `fun` over long-only weights with the given constraints (SLSQP).

    Args:
        fun (callable): Objective mapping a length-n weight vector to a scalar.
        n (int): Number of (risky) decision weights.
        constraints (list): SLSQP constraint dicts.
        x0 (np.ndarray | None): Optional warm start; defaults to 1/n.

    Returns:
        np.ndarray: Optimal weight vector of length n.
    """
    if x0 is None:
        x0 = np.full(n, 1.0 / n)
    res = minimize(
        fun,
        x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return np.clip(res.x, 0.0, 1.0)


def min_variance_portfolio(sigma: np.ndarray) -> np.ndarray:
    """
    Global minimum-variance, long-only, fully-invested (risky-only) portfolio.

    Args:
        sigma (np.ndarray): Covariance matrix (n_risky, n_risky).

    Returns:
        np.ndarray: Risky weights of length n_risky (sum to 1).
    """
    n = sigma.shape[0]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return _solve(lambda w: float(w @ sigma @ w), n, cons)


def markowitz_frontier(mu: np.ndarray, sigma: np.ndarray, n_points: int = 40) -> dict:
    """
    Long-only mean-variance efficient frontier over the risky assets: for a
    grid of target returns, minimize variance subject to the target.

    Args:
        mu (np.ndarray): Expected returns (n_risky,).
        sigma (np.ndarray): Covariance (n_risky, n_risky).
        n_points (int): Number of target-return grid points.

    Returns:
        dict: {"vol": (k,), "ret": (k,), "weights": (k, n_risky), "gmv": (n_risky,)}
            where k <= n_points (infeasible targets are dropped).
    """
    n = sigma.shape[0]
    targets = np.linspace(float(mu.min()), float(mu.max()), n_points)
    vols, rets, weights = [], [], []
    for target in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tgt=target: float(w @ mu) - tgt},
        ]
        w = _solve(lambda w: float(w @ sigma @ w), n, cons)
        # keep only points that actually met the return target (feasible)
        if abs(float(w @ mu) - target) > 1e-4:
            continue
        vols.append(float(np.sqrt(max(w @ sigma @ w, 0.0))))
        rets.append(float(w @ mu))
        weights.append(w)
    return {
        "vol": np.array(vols),
        "ret": np.array(rets),
        "weights": np.array(weights),
        "gmv": min_variance_portfolio(sigma),
    }


def tangency_portfolio(mu: np.ndarray, sigma: np.ndarray, rf: float = 0.0) -> np.ndarray | None:
    """
    Maximum-Sharpe (tangency) long-only, fully-invested risky portfolio.

    Returns None in the degenerate case where no risky asset beats the
    risk-free rate (`max(mu) <= rf`), signalling an all-cash optimum.

    Args:
        mu (np.ndarray): Expected returns (n_risky,).
        sigma (np.ndarray): Covariance (n_risky, n_risky).
        rf (float): Risk-free (cash) per-step return (0 in this environment).

    Returns:
        np.ndarray | None: Risky weights (sum to 1), or None if degenerate.
    """
    if float(np.max(mu)) <= rf + EPS:
        return None
    n = sigma.shape[0]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_sharpe(w: np.ndarray) -> float:
        vol = np.sqrt(max(w @ sigma @ w, EPS))
        return -float((w @ mu - rf) / vol)

    return _solve(neg_sharpe, n, cons)


def capital_market_line(mu: np.ndarray, sigma: np.ndarray, rf: float = 0.0) -> dict | None:
    """
    Capital Market Line data: the tangency point and the max attainable Sharpe
    slope. The CML is the honest efficient boundary for cash-holding strategies.

    Args:
        mu (np.ndarray): Expected returns (n_risky,).
        sigma (np.ndarray): Covariance (n_risky, n_risky).
        rf (float): Risk-free per-step return.

    Returns:
        dict | None: {"sigma_tan", "mu_tan", "slope", "weights"} or None if the
            tangency portfolio is degenerate (all-cash optimum).
    """
    w = tangency_portfolio(mu, sigma, rf=rf)
    if w is None:
        return None
    sigma_tan = float(np.sqrt(max(w @ sigma @ w, 0.0)))
    mu_tan = float(w @ mu)
    slope = (mu_tan - rf) / sigma_tan if sigma_tan > EPS else 0.0
    return {"sigma_tan": sigma_tan, "mu_tan": mu_tan, "slope": slope, "weights": w}


def kappa_of(alpha: float, nu: float) -> float:
    """
    Identical formula to `utils.utils.kappa_of`.

    Args:
        alpha (float): Risk level.
        nu (float): Degrees of freedom.

    Returns:
        float: K(alpha, nu).
    """
    a = np.asarray(alpha, dtype=float)
    q = t_dist.ppf(a, df=nu)
    return t_dist.pdf(q, df=nu) * (nu + q**2) / (a * (nu - 1.0))


def t_cvar_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
    nu: float,
    relative: bool = False,
) -> np.ndarray:
    """
    Static parametric-t mean-CVaR@alpha portfolio: maximize the alpha-percentile
    expectation `Gamma = w^T mu - K * sqrt(w^T Sigma w)` over long-only weights
    with cash as the residual.

    Args:
        mu (np.ndarray): Expected risky returns (n_risky,).
        sigma (np.ndarray): Covariance (n_risky, n_risky).
        alpha (float): CVaR risk level.
        nu (float): Student-t degrees of freedom.
        relative (bool): If True use the actor's relative tilt `K(alpha) - K(0.5)`
            (clipped at 0) to reproduce the trained objective exactly. If False
            use the full `K(alpha)`.

    Returns:
        np.ndarray: Full weight vector [cash, *risky] of length n_risky + 1.
    """
    n = sigma.shape[0]
    k = float(kappa_of(alpha, nu))
    if relative:
        k = max(k - float(kappa_of(0.5, nu)), 0.0)

    def neg_gamma(w: np.ndarray) -> float:
        risk = np.sqrt(max(w @ sigma @ w, 0.0) + EPS)
        return -float(w @ mu - k * risk)

    # cash is the residual: sum(risky) <= 1.
    cons = [{"type": "ineq", "fun": lambda w: 1.0 - np.sum(w)}]
    w = _solve(neg_gamma, n, cons, x0=np.zeros(n))
    cash = max(1.0 - float(np.sum(w)), 0.0)
    return np.concatenate([[cash], w])


def t_cvar_family(
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: float,
    alphas,
    relative: bool = False,
) -> dict[float, np.ndarray]:
    """
    The alpha-indexed family of static parametric-t mean-CVaR portfolios.

    Args:
        mu (np.ndarray): Expected risky returns.
        sigma (np.ndarray): Covariance.
        nu (float): Student-t degrees of freedom.
        alphas (Iterable[float]): Risk levels.
        relative (bool): See `t_cvar_weights`.

    Returns:
        dict[float, np.ndarray]: {alpha: full weight vector [cash, *risky]}.
    """
    return {float(a): t_cvar_weights(mu, sigma, float(a), nu, relative=relative) for a in alphas}
