"""
Section 6.3.5 t-タプル推定器 (t-Tuple Estimate)
規格書 Section 6.3.5 に基づく実装
"""

import math
from collections import Counter
import numpy as np
from sp800_90b.estimators.common import calc_z_upper_bound


def estimate_t_tuple(data: np.ndarray, threshold: int = 35) -> float:
    """
    t-タプル推定器 (t-Tuple Estimate)
    重複ありで連続する i-tuple の出現頻度を調べ、
    最大頻度が 35 以上の最大長 t までの各 i について P_max[i] = (Q[i] / (L-i+1))^(1/i)
    の最大値を求め、99%信頼区間上限からエントロピーを算出する。
    """
    length = len(data)
    if length < 2:
        return 0.0

    # 1. 各タプル長の最大頻度を計算
    q_dict = {}
    p_max_list = []

    t = 1
    # 探索上限（通常128程度で十分かつ頻度<35となる）
    max_t_limit = min(128, length)

    # 連続スライスのタプルを高速に走査
    data_list = data.tolist()

    while t <= max_t_limit:
        # 長さ t のタプルをカウント
        counter = Counter()
        for i in range(length - t + 1):
            tpl = tuple(data_list[i : i + t])
            counter[tpl] += 1

        if not counter:
            break

        max_count = max(counter.values())
        if max_count < threshold and t > 1:
            # 閾値未満になったら終了
            break

        q_dict[t] = max_count
        p_i = max_count / (length - t + 1)
        p_max_i = p_i ** (1.0 / t)
        p_max_list.append(p_max_i)

        t += 1

    if not p_max_list:
        return 0.0

    p_hat_max = max(p_max_list)
    p_u = calc_z_upper_bound(p_hat_max, length)
    if p_u <= 0.0:
        return float("inf")
    if p_u >= 1.0:
        return 0.0

    return -math.log2(p_u)
