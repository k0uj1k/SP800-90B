"""
Section 6.3.8 ラグ予測器推定 (The Lag Prediction Estimate)
規格書 Section 6.3.8 に基づく実装
"""

import math
from typing import Optional
import numpy as np
from sp800_90b.estimators.common import calc_p_global_prime, solve_p_local


def estimate_lag(
    data: np.ndarray,
    k: int,
    max_d: int = 128,
) -> float:
    """
    Lag 予測器による最小エントロピー推定
    D = 128 の遅延サブ予測器を並行に実行し、最適な遅延を用いて次の値を予測する。
    """
    length = len(data)
    if length <= 2:
        return 0.0

    d_limit = min(max_d, length - 1)
    n = length - 1
    correct = np.zeros(n, dtype=np.uint8)

    scoreboard = [0] * (d_limit + 1)
    winner = 1

    data_list = data.tolist()

    # i = 1 から length - 1 まで走査 (0-indexed: i=1 is the 2nd element)
    for i in range(1, length):
        target = data_list[i]

        # 予測
        if winner <= i:
            prediction = data_list[i - winner]
        else:
            prediction = None

        if prediction is not None and prediction == target:
            correct[i - 1] = 1

        # スコアボード更新
        for d in range(1, min(d_limit + 1, i + 1)):
            if data_list[i - d] == target:
                scoreboard[d] += 1
                if scoreboard[d] >= scoreboard[winner]:
                    winner = d

    # グローバル性能
    c = int(np.sum(correct))
    p_global_prime = calc_p_global_prime(c, n)

    # ローカル性能
    max_run = 0
    curr_run = 0
    for val in correct:
        if val == 1:
            curr_run += 1
            if curr_run > max_run:
                max_run = curr_run
        else:
            curr_run = 0

    r = max_run + 1
    p_local = solve_p_local(r, n)

    metric = max(p_global_prime, p_local, 1.0 / k)
    if metric >= 1.0:
        return 0.0
    return -math.log2(metric)
