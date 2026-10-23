"""
Section 6.3.7 Multi Most Common in Window (MultiMCW) 予測器推定
規格書 Section 6.3.7 に基づく実装
"""

import math
from collections import Counter
from typing import List, Optional
import numpy as np
from sp800_90b.estimators.common import calc_p_global_prime, solve_p_local


def estimate_multi_mcw(
    data: np.ndarray,
    k: int,
    window_sizes: Optional[List[int]] = None,
) -> float:
    """
    MultiMCW 予測器による最小エントロピー推定
    デフォルト窓サイズ: [63, 255, 1023, 4095]
    """
    length = len(data)
    if window_sizes is None:
        window_sizes = [63, 255, 1023, 4095]

    w1 = window_sizes[0]
    if length <= w1 + 1:
        return 0.0

    n = length - w1
    correct = np.zeros(n, dtype=np.uint8)

    # 4つのサブ予測器のスコアボード
    num_preds = len(window_sizes)
    scoreboard = [0] * num_preds
    winner = 0

    # 窓ごとの出現回数カウンタを効率的に管理
    # 各窓の最近の出現順序も追跡（タイブレーク用: 最も最近出現した最頻値）
    data_list = data.tolist()

    for i in range(w1, length):
        target = data_list[i]
        frequent = [None] * num_preds

        # 各サブ予測器の最頻値を算出
        for j, w in enumerate(window_sizes):
            if i >= w:
                window_slice = data_list[i - w : i]
                counts = Counter(window_slice)
                # タイブレーク: 最大頻度のうち最も最近（末尾に近い方）現れたもの
                max_cnt = -1
                best_val = None
                best_last_idx = -1
                # 直近から逆順に確認
                for idx_rev, val in enumerate(reversed(window_slice)):
                    cnt = counts[val]
                    if cnt > max_cnt:
                        max_cnt = cnt
                        best_val = val
                        best_last_idx = idx_rev
                    elif cnt == max_cnt:
                        if idx_rev < best_last_idx:  # より最近
                            best_val = val
                            best_last_idx = idx_rev
                frequent[j] = best_val
            else:
                frequent[j] = None

        # 予測と正誤判定
        prediction = frequent[winner]
        if prediction is not None and prediction == target:
            correct[i - w1] = 1

        # スコアボード更新
        for j in range(num_preds):
            if frequent[j] is not None and frequent[j] == target:
                scoreboard[j] += 1
                if scoreboard[j] >= scoreboard[winner]:
                    winner = j

    # グローバル性能
    c = int(np.sum(correct))
    p_global_prime = calc_p_global_prime(c, n)

    # ローカル性能 (最長連続正解長)
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
