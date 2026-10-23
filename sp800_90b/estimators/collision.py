"""
Section 6.3.2 衝突推定器 (The Collision Estimate)
規格書 Section 6.3.2 に基づく実装 (バイナリデータ専用)
"""

import math
import numpy as np
from sp800_90b.estimators.common import continued_fraction_f_inv_z


def estimate_collision(data: np.ndarray) -> float:
    """
    衝突推定器 (The Collision Estimate)
    バイナリ入力に対して、最初の衝突（同じ値が2回出現）が起きるまでの観測回数を計数し、
    その標本平均 X_bar からパラメータ p を二分探索で逆算してエントロピーを推定する。
    """
    length = len(data)
    if length < 2:
        return 0.0

    t_list = []
    index = 0
    while index < length:
        # index から重複が出るまでスキャン
        seen = set()
        j = index
        while j < length:
            val = int(data[j])
            if val in seen:
                break
            seen.add(val)
            j += 1

        if j < length:
            # 衝突が見つかった
            t_v = j - index + 1
            t_list.append(t_v)
            index = j + 1
        else:
            # 重複が見つからず末尾に達した
            break

    v = len(t_list)
    if v <= 1:
        return 1.0

    t_arr = np.array(t_list, dtype=np.float64)
    x_bar = np.mean(t_arr)
    s_hat = np.std(t_arr, ddof=1)

    # 99%信頼区間下限
    x_bar_prime = x_bar - 2.576 * (s_hat / math.sqrt(v))

    def expected_x(p: float) -> float:
        q = 1.0 - p
        # p >= q なので p >= 0.5
        if p >= 1.0:
            return 2.0
        if q <= 0.0:
            return 2.0

        diff = (1.0 / p) - (1.0 / q)
        # z = 1/q
        z = 1.0 / q
        f_q = continued_fraction_f_inv_z(z)

        term1 = p * (q ** -2) * (1.0 + 0.5 * diff) * f_q
        term2 = p * (q ** -1) * 0.5 * diff
        return term1 - term2

    # 二分探索: p in [0.5, 1.0]
    # expected_x(0.5) は最大で約2.718 (e), p->1.0で2.0に近づく
    low = 0.5
    high = 1.0

    x_at_low = expected_x(low)
    x_at_high = expected_x(high)

    if x_bar_prime >= x_at_low:
        return 1.0  # 最大エントロピー
    if x_bar_prime <= x_at_high:
        return 0.0

    for _ in range(60):
        mid = (low + high) / 2.0
        val = expected_x(mid)
        # val は p が増加すると減少する
        if val > x_bar_prime:
            low = mid
        else:
            high = mid

    p_sol = (low + high) / 2.0
    if p_sol <= 0.0 or p_sol > 1.0:
        return 1.0
    return -math.log2(p_sol)
