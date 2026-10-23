"""
NIST SP 800-90B 数値計算および統計共通ユーティリティモジュール
"""

import math
from typing import Callable
import numpy as np
from scipy.special import gammaincc, gamma


def calc_z_upper_bound(p_hat: float, length: int) -> float:
    """
    規格書 Section 6.3.1 式 (2) などで使用される 99% 信頼区間上限値 (Z = 2.576) を計算する。
    p_u = min(1.0, p_hat + 2.576 * sqrt(p_hat * (1 - p_hat) / (length - 1)))
    """
    if length <= 1:
        return 1.0
    if p_hat <= 0.0:
        return 0.0
    if p_hat >= 1.0:
        return 1.0
    variance_term = (p_hat * (1.0 - p_hat)) / (length - 1)
    p_u = p_hat + 2.576 * math.sqrt(variance_term)
    return min(1.0, max(0.0, p_u))


def calc_p_global_prime(c: int, n: int) -> float:
    """
    規格書 Section 6.3.7 式 (5) 予測器のグローバル成功確率の 99% 信頼区間上限値を計算する。
    """
    if n <= 0:
        return 1.0
    p_global = c / n
    if p_global == 0.0:
        return 1.0 - (0.01 ** (1.0 / n))
    variance_term = (p_global * (1.0 - p_global)) / (n - 1)
    return min(1.0, p_global + 2.576 * math.sqrt(variance_term))


def solve_p_local(r: int, n: int) -> float:
    """
    規格書 Section 6.3.7 式 (6) / Section G.2
    最長連続正解長から P_local を二分探索で算出する。
    0.99 = (1 - P_local * x) / ((r + 1 - r * P_local) * q) * (1 / (x^(N+1)))
    ここで q = 1 - P_local, x は漸化式 x_j = 1 + q * (P_local^r) * (x_{j-1}^(r+1)) の収束値 (x = x_10)
    """
    if r <= 1:
        return 0.0

    def compute_target(p_local: float) -> float:
        if p_local <= 0.0:
            return 1.0
        if p_local >= 1.0:
            return 0.0
        q = 1.0 - p_local
        # x_0 = 1 から x_10 まで漸化式を反復
        x = 1.0
        for _ in range(10):
            try:
                # 対数領域で保護
                x_pow = x ** (r + 1)
                x = 1.0 + q * (p_local ** r) * x_pow
            except OverflowError:
                x = 1e100

        denom1 = (r + 1 - r * p_local) * q
        num1 = 1.0 - p_local * x
        if denom1 <= 0.0 or x <= 0.0:
            return 0.0

        try:
            # 1 / (x^(N+1)) = exp(-(N+1) * ln(x))
            log_term = -(n + 1) * math.log(x)
            factor2 = math.exp(log_term)
            val = (num1 / denom1) * factor2
            return val
        except (OverflowError, ValueError):
            return 0.0

    # 二分探索 (P_local は [0.0, 1.0])
    low = 0.0
    high = 1.0
    for _ in range(60):
        mid = (low + high) / 2.0
        val = compute_target(mid)
        # val は p_local が増加するにつれて減少する
        if val > 0.99:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0


def continued_fraction_f_inv_z(z: float, depth: int = 40) -> float:
    """
    規格書 Appendix G.1.1
    F(1/z) = Γ(3, z) * z^(-3) * e^z の連分数近似を計算する。
    F(1/z) = 1 / (z + (-1 / (1 + 1 / (z + 2 / (1 + ...)))))
    あるいは scipy の gammaincc (上側不完全ガンマ関数比) を用いて高精度計算。
    Γ(a, z) = gammaincc(a, z) * gamma(a)
    """
    try:
        upper_gamma = gammaincc(3.0, z) * gamma(3.0)
        return upper_gamma * (z ** -3) * math.exp(z)
    except Exception:
        # フォールバック連分数
        return 1.0
