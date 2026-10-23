"""
Section 3.1.4 リスタートテスト (Restart Tests)
Section 3.1.5 コンディショニング評価 (Conditioning Component)
規格書 Section 3.1.4, 3.1.5 に基づく実装
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.stats import binom

from sp800_90b.estimators import assess_non_iid_entropy, EntropyAssessmentResult


@dataclass
class RestartTestResult:
    """リスタートテストの検証結果"""
    passed: bool
    sanity_passed: bool
    x_max: int
    p_value: float
    h_r: float
    h_c: float
    h_final: float


def run_restart_sanity_check(
    matrix: np.ndarray,
    h_i: float,
    alpha: float = 0.000005,
) -> Tuple[bool, int, float]:
    """
    3.1.4.3 Sanity Check - Most Common Value in the Rows and Columns
    matrix: 1000 x 1000 の再起動データ行列 M
    """
    p = 2.0 ** (-h_i)
    # 各行の最頻値カウントの最大値 X_R
    xr_list = []
    for row in matrix:
        _, counts = np.unique(row, return_counts=True)
        xr_list.append(np.max(counts))
    xr = max(xr_list)

    # 各列の最頻値カウントの最大値 X_C
    xc_list = []
    for col in matrix.T:
        _, counts = np.unique(col, return_counts=True)
        xc_list.append(np.max(counts))
    xc = max(xc_list)

    x_max = max(xr, xc)

    # 二項検定: Pr(X >= X_max) = 1 - binom.cdf(X_max - 1, 1000, p)
    p_val = float(1.0 - binom.cdf(x_max - 1, 1000, p))
    sanity_passed = (p_val >= alpha)

    return sanity_passed, x_max, p_val


def run_restart_tests(
    matrix: np.ndarray,
    h_i: float,
    k: int,
    alpha: float = 0.000005,
) -> RestartTestResult:
    """
    規格書 Section 3.1.4:
    1. Sanity Check を実行 (不合格なら検証失敗)
    2. 行連結データ (row dataset) と 列連結データ (column dataset) に対しエントロピー推定を実施
    3. min(H_r, H_c) < H_i / 2 の場合は検証失敗
    4. 最終エントロピー = min(H_r, H_c, H_i)
    """
    sanity_passed, x_max, p_val = run_restart_sanity_check(matrix, h_i, alpha=alpha)
    if not sanity_passed:
        return RestartTestResult(
            passed=False,
            sanity_passed=False,
            x_max=x_max,
            p_value=p_val,
            h_r=0.0,
            h_c=0.0,
            h_final=0.0,
        )

    # Row dataset & Column dataset
    row_dataset = matrix.flatten()
    col_dataset = matrix.T.flatten()

    r_res = assess_non_iid_entropy(row_dataset, k=k)
    c_res = assess_non_iid_entropy(col_dataset, k=k)

    h_r = r_res.min_entropy
    h_c = c_res.min_entropy

    if min(h_r, h_c) < (h_i / 2.0):
        return RestartTestResult(
            passed=False,
            sanity_passed=True,
            x_max=x_max,
            p_value=p_val,
            h_r=h_r,
            h_c=h_c,
            h_final=0.0,
        )

    h_final = min(h_r, h_c, h_i)
    return RestartTestResult(
        passed=True,
        sanity_passed=True,
        x_max=x_max,
        p_value=p_val,
        h_r=h_r,
        h_c=h_c,
        h_final=h_final,
    )


def compute_output_entropy(n_in: int, n_out: int, nw: int, h_in: float) -> float:
    """
    規格書 Section 3.1.5.1.2:
    Vetted なコンディショニング関数に対する出力エントロピー h_out を計算する。
    """
    p_high = 2.0 ** (-h_in)
    p_low = (1.0 - p_high) / (2.0 ** n_in - 1.0)
    n = min(n_out, nw)

    psi = (2.0 ** (n_in - n)) * p_low + p_high
    u = (2.0 ** (n_in - n)) + math.sqrt(2.0 * n * (2.0 ** (n_in - n)) * math.log(2.0))
    omega = u * p_low

    metric = max(psi, omega)
    if metric >= 1.0:
        return 0.0
    return -math.log2(metric)
