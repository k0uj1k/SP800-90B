"""
Section 6.3.9 MultiMMC 予測器推定 (The MultiMMC Prediction Estimate)
規格書 Section 6.3.9 に基づく実装
"""

import math
from collections import defaultdict
from typing import Dict, Tuple
import numpy as np
from sp800_90b.estimators.common import calc_p_global_prime, solve_p_local


def estimate_multi_mmc(
    data: np.ndarray,
    k: int,
    depth: int = 16,
    max_entries: int = 100_000,
) -> float:
    """
    MultiMMC (Markov Model with Counting) 予測器による最小エントロピー推定
    深度 1 〜 16 のマルコフ遷移頻度モデルを並行して保持し、最も適中率の高いモデルで次を予測する。
    """
    length = len(data)
    if length <= 3:
        return 0.0

    n = length - 2
    correct = np.zeros(n, dtype=np.uint8)

    # 各深度 d (1..depth) のモデル: context_tuple -> (symbol -> count)
    models: Dict[int, Dict[Tuple[int, ...], Dict[int, int]]] = {
        d: defaultdict(lambda: defaultdict(int)) for d in range(1, depth + 1)
    }
    entries = [0] * (depth + 1)
    scoreboard = [0] * (depth + 1)
    winner = 1

    data_list = data.tolist()

    # 規格書 1-indexed: i=3 to L -> 0-indexed: i=2 to length-1
    for i in range(2, length):
        target = data_list[i]

        # Step 4a: モデル更新 (前ステップの遷移を学習)
        for d in range(1, depth + 1):
            if d < i:  # 直前の d 長コンテキストが存在する
                context = tuple(data_list[i - 1 - d : i - 1])
                prev_sym = data_list[i - 1]
                m_d = models[d]
                if context in m_d or entries[d] < max_entries:
                    if context not in m_d:
                        entries[d] += 1
                    m_d[context][prev_sym] += 1

        # Step 4b: 各深度での予測値 subpredict_d を計算
        subpredict = [None] * (depth + 1)
        for d in range(1, depth + 1):
            if d <= i:
                curr_context = tuple(data_list[i - d : i])
                m_d = models[d]
                if curr_context in m_d:
                    counts = m_d[curr_context]
                    if counts:
                        # 最大頻度の y を選択 (タイブレーク: より大きな値)
                        max_c = -1
                        best_y = None
                        for y, cnt in counts.items():
                            if cnt > max_c:
                                max_c = cnt
                                best_y = y
                            elif cnt == max_c:
                                if best_y is None or y > best_y:
                                    best_y = y
                        subpredict[d] = best_y

        # Step 4c: winner の予測を採用
        prediction = subpredict[winner]
        if prediction is not None and prediction == target:
            correct[i - 2] = 1

        # Step 4e: スコアボード更新
        for d in range(1, depth + 1):
            if subpredict[d] is not None and subpredict[d] == target:
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
