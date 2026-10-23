"""
Section 6.3.10 LZ78Y 予測器推定 (The LZ78Y Prediction Estimate)
規格書 Section 6.3.10 に基づく実装
"""

import math
from collections import defaultdict
from typing import Dict, Tuple
import numpy as np
from sp800_90b.estimators.common import calc_p_global_prime, solve_p_local


def estimate_lz78y(
    data: np.ndarray,
    k: int,
    b: int = 16,
    max_dict_size: int = 65_536,
) -> float:
    """
    LZ78Y 予測器による最小エントロピー推定
    辞書学習と最長一致コンテキストを用いた予測を行う。
    """
    length = len(data)
    if length <= b + 2:
        return 0.0

    n = length - b - 1
    correct = np.zeros(n, dtype=np.uint8)

    # 辞書: prev_tuple -> dict(next_symbol -> count)
    d_dict: Dict[Tuple[int, ...], Dict[int, int]] = {}
    dict_size = 0

    data_list = data.tolist()

    # 規格書 1-indexed: i = B + 2 to L -> 0-indexed: i = b + 1 to length - 1
    for i in range(b + 1, length):
        target = data_list[i]

        # Step 3a: 辞書の更新
        # j = B down to 1
        for j in range(b, 0, -1):
            sub_seq = tuple(data_list[i - j - 1 : i - 1])
            prev_sym = data_list[i - 1]
            if sub_seq not in d_dict:
                if dict_size < max_dict_size:
                    d_dict[sub_seq] = {prev_sym: 1}
                    dict_size += 1
            else:
                d_dict[sub_seq][prev_sym] = d_dict[sub_seq].get(prev_sym, 0) + 1

        # Step 3b: 辞書を用いて次の値を予測
        prediction = None
        max_count = 0

        for j in range(b, 0, -1):
            prev = tuple(data_list[i - j : i])
            if prev in d_dict:
                entries = d_dict[prev]
                # 最大頻度の y を選択 (タイブレーク: より大きなバイト値)
                best_y = None
                best_cnt = -1
                for y, cnt in entries.items():
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_y = y
                    elif cnt == best_cnt:
                        if best_y is None or y > best_y:
                            best_y = y
                if best_cnt > max_count:
                    prediction = best_y
                    max_count = best_cnt

        if prediction is not None and prediction == target:
            correct[i - b - 1] = 1

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
