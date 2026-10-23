"""
Section 6.3.1 最頻値推定器 (The Most Common Value Estimate)
規格書 Section 6.3.1 に基づく実装
"""

import math
import numpy as np
from sp800_90b.estimators.common import calc_z_upper_bound


def estimate_mcv(data: np.ndarray) -> float:
    """
    最頻値推定器 (MCV Estimate) により最小エントロピーを算出する。
    1. 最も頻繁に出現するシンボルの出現割合 p_hat を求める。
    2. 99%信頼区間上限値 p_u を計算する。
    3. 最小エントロピー H = -log2(p_u) を返す。
    """
    length = len(data)
    if length == 0:
        return 0.0

    # 各シンボルの出現回数をカウント
    _, counts = np.unique(data, return_counts=True)
    max_count = np.max(counts)
    p_hat = max_count / length

    p_u = calc_z_upper_bound(p_hat, length)
    if p_u <= 0.0:
        return float("inf")
    if p_u >= 1.0:
        return 0.0

    return -math.log2(p_u)
