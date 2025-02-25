"""Quantitative Finance Library – Developed as I learn things"""
from typing import List, Union, Sequence
import math
import numpy as np
import pandas


def needs_stability(r_t: Sequence[float], verbose: bool = False) -> bool:
    """
    Analyze a return series to determine if numerical stability measures are needed.

    Args:
        r_t (Sequence[float]): Time series of returns
        verbose (bool): If True, prints detailed analysis

    Returns:
        bool: True if stable computation is recommended

    Checks:
    1. Magnitude range (very large or very small numbers)
    2. Precision requirements (many decimal places)
    3. Series length (long series more prone to accumulation errors)
    4. Value separation (numbers very close together)
    """
    if len(r_t) == 0:
        return False

    # Convert to absolute values for magnitude checks
    abs_values = [abs(x) for x in r_t if x != 0]  # Exclude zeros
    if not abs_values:
        return False

    # Calculate key metrics
    min_val = min(abs_values)
    max_val = max(abs_values)
    length = len(r_t)

    # Define thresholds
    MAGNITUDE_THRESHOLD = 1e6  # Consider stable if range > 1e6
    PRECISION_THRESHOLD = 1e-5  # Consider stable if values < 1e-5
    LENGTH_THRESHOLD = 10000  # Consider stable if length > 10000

    # Check conditions
    needs_stable = False
    reasons = []

    # 1. Check magnitude range
    magnitude_range = max_val / min_val if min_val > 0 else 0
    if magnitude_range > MAGNITUDE_THRESHOLD:
        needs_stable = True
        reasons.append(f"Large magnitude range: {magnitude_range:.2e}")

    # 2. Check for very small values
    if min_val < PRECISION_THRESHOLD:
        needs_stable = True
        reasons.append(f"Very small values present: {min_val:.2e}")

    # 3. Check for very large values needing precision
    if max_val > 1 / PRECISION_THRESHOLD:
        needs_stable = True
        reasons.append(f"Very large values present: {max_val:.2e}")

    # 4. Check series length
    if length > LENGTH_THRESHOLD:
        needs_stable = True
        reasons.append(f"Long series length: {length} points")

    # 5. Check for close number separation
    differences = [abs(r_t[i] - r_t[i - 1]) for i in range(1, len(r_t))]
    min_diff = min(differences) if differences else float('inf')
    if PRECISION_THRESHOLD > min_diff > 0:
        needs_stable = True
        reasons.append(f"Very small separations present: {min_diff:.2e}")

    if verbose and needs_stable:
        print("Stability measures recommended due to:")
        for reason in reasons:
            print(f"- {reason}")
        print(f"\nSummary statistics:")
        print(f"- Series length: {length}")
        print(f"- Value range: [{min_val:.2e}, {max_val:.2e}]")
        print(f"- Magnitude range: {magnitude_range:.2e}")
        print(f"- Minimum separation: {min_diff:.2e}")

    return needs_stable


class OpenClose:
    def __init__(self, _open: float, _close: float) -> None:
        """
        Initialize price points for return calculations.

        Args:
            _open (float): Opening price (Pt-1)
            _close (float): Closing price (Pt)
        """
        self.open: float = _open
        self.close: float = _close

    @property
    def returns(self) -> float:
        """R_t = (Pt/Pt-1) -1 ~ R is a percentage"""
        return (self.close/self.open) - 1

    @property
    def log_returns(self) -> float:
        """r_t = ln(1+R_t)"""
        return math.log(1 + self.returns)

    @staticmethod
    def from_df(df: pandas.DataFrame, opens_key='Open', close_key='Close/Last'):
        """
        Create a list of OpenClose objects from a DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame containing price data
            opens_key (str): Column name for opening prices
            close_key (str): Column name for closing prices

        Returns:
            List[OpenClose]: List of OpenClose objects
        """
        opens = df[opens_key].tolist()
        closes = df[close_key].tolist()
        return [OpenClose(opens[i], closes[i]) for i in range(len(opens))]


def covariance(r_t: Sequence[float], lag: int) -> float:
    """
    Calculate the covariance between a time series and its lagged version.

    Args:
        r_t (Sequence[float]): Time series of returns
        lag (int): Lag parameter

    Returns:
        float: Covariance at specified lag

    .. math::
        γ_ℓ = \\frac{1}{N-ℓ}\\sum_{t=ℓ+1}^{N} (r_t - μ)(r_{t-ℓ} - μ)
    """
    n: int = len(r_t)  # Total number of elements (N)
    mean: float = sum(r_t) / n  # Mean (μ)
    scaling_factor: float = 1 / (n - lag)  # 1 / (N - ℓ)

    running_total: float = 0

    for i in range(lag, n):
        r_x: float = r_t[i]
        running_total += (r_x - mean) * (r_t[i-lag] - mean)

    gamma_l: float = scaling_factor * running_total
    return gamma_l



def autocovariance(r_t, gamma_k):
    """
    Calculate the autocovariance of r_t and it's gamma_k.
    NOTE: gamma_k is the covariance of r_t at lag-k.

    .. math::
        P_\ell=\\frac{Cov(r_t, r_{t-\ell})}{\sqrt{Var(r_t)Var(r_{t-\ell})}} = \\frac{Cov(r_t, r_{t-\ell})}{{Var(r_t)}}=\\frac{\gamma_\ell}{\gamma_o}

    :param r_t: Log return Time Series
    :param gamma_k: Covariance of r_t at lag-k
    :return:
    """
    variance = covariance(r_t, 0)
    return gamma_k / variance



def pearson_correlation(X, Y):
    """
    Calculate the Pearson correlation coefficient between two sequences.

    Args:
        X (Sequence[float]): First sequence of values
        Y (Sequence[float]): Second sequence of values

    Returns:
        float: Pearson correlation coefficient

    The Pearson correlation coefficient is calculated as:

    .. math::
        P_{xy} = \\frac{Cov(X, Y)}{\\sigma_X \\sigma_Y}
    where Cov(X, Y) is the covariance of X and Y, and \\sigma_X and \\sigma_Y are the standard deviations of X and Y, respectively.
    """
    # Convert input lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    # Calculate means
    mu_x = np.mean(X)
    mu_y = np.mean(Y)

    # Calculate covariance
    cov_xy = np.mean((X - mu_x) * (Y - mu_y))

    # Calculate standard deviations
    std_x = np.sqrt(np.mean((X - mu_x) ** 2))
    std_y = np.sqrt(np.mean((Y - mu_y) ** 2))

    # Calculate Pearson correlation coefficient
    P_xy = cov_xy / (std_x * std_y)

    return P_xy


class LogReturns:
    def __init__(self, r_t: Sequence[float]) -> None:
        """
        Initialize with a sequence of log returns.

        Args:
            r_t (Sequence[float]): Time series of logarithmic returns
        """
        self._rt: Sequence[float] = r_t

    def cov(self, lag: int) -> float:
        """
        Calculate the covariance at given lag.

        Args:
            lag (int): Lag parameter

        Returns:
            float: Covariance at specified lag
        """
        return covariance(self._rt, lag)


    def var(self) -> float:
        """
        Calculate the variance (lag-0 covariance).

        Returns:
            float: Variance of the return series
        """
        return self.cov(0)

    @property
    def returns(self) -> List[float]:
        """Return the raw return series."""
        return self._rt
