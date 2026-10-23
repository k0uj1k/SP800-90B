"""
Section 6.3.6 最長重複部分列推定器 (Longest Repeated Substring / LRS Estimate)
規格書 Section 6.3.6 に基づく実装
"""

import math
from collections import Counter
import numpy as np
from sp800_90b.estimators.common import calc_z_upper_bound


def estimate_lrs(data: np.ndarray, threshold: int = 35) -> float:
    """
    最長重複部分列推定器 (LRS Estimate)
    1. 最も頻繁に出現する u-tuple の回数が 35 未満となる最小の u を求める。
    2. 2回以上出現する最大のタプル長 v を求める。
    3. W = u から v までの各タプルについて衝突確率 P_W を計算し、
       P_max_W = P_W^(1/W) の最大値を求める。
    4. 99%信頼区間上限からエントロピーを算出する。
    """
    length = len(data)
    if length < 2:
        return 0.0

    data_list = data.tolist()

    # u を探索: 最頻出回数が threshold (35) 未満となる最小の u
    u = 1
    while u <= length:
        counter = Counter()
        for i in range(length - u + 1):
            counter[tuple(data_list[i : i + u])] += 1
        if not counter:
            break
        if max(counter.values()) < threshold:
            break
        u += 1

    # v を探索: 最頻出回数が >= 2 となる最大の v
    v = u
    while v <= length:
        counter = Counter()
        for i in range(length - v + 1):
            counter[tuple(data_list[i : i + v])] += 1
        if not counter or max(counter.values()) < 2:
            v -= 1
            break
        v += 1

    if v < u:
        # 推定不能な場合は MCV などの値にフォールバック
        return 1.0

    p_max_w_list = []
    for w in range(u, v + 1):
        counter = Counter()
        for i in range(length - w + 1):
            counter[tuple(data_list[i : i + w])] += 1

        # sum(C_i * (C_i - 1) / 2)
        total_collisions = sum((c * (c - 1)) // 2 for c in counter.values())
        num_subseqs = length - w + 1
        total_pairs = (num_subseqs * (num_subseqs - 1)) // 2

        if total_pairs == 0 or total_collisions == 0:
            continue

        p_w = total_collisions / total_pairs
        p_max_w = p_w ** (1.0 / w)
        p_max_w_list.append(p_max_w)

    if not p_max_w_list:
        return 1.0

    p_hat = max(p_max_w_list)
    p_u = calc_z_upper_bound(p_hat, length)
    if p_u <= 0.0:
        return float("inf")
    if p_u >= 1.0:
        return 0.0

    return -math.log2(p_u)
