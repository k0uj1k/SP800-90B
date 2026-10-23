"""
Section 6.3.3 マルコフ推定器 (The Markov Estimate)
規格書 Section 6.3.3 に基づく実装 (バイナリデータ専用)
"""

import math
import numpy as np


def estimate_markov(data: np.ndarray) -> float:
    """
    マルコフ推定器 (The Markov Estimate)
    1次のマルコフモデル（2x2遷移行列）を構築し、長さ128の系列で最も生起確率の高いパスの確率から
    1サンプルあたりのエントロピーを推定する。
    """
    length = len(data)
    if length < 2:
        return 0.0

    count_0 = int(np.count_nonzero(data == 0))
    count_1 = length - count_0
    p0 = count_0 / length
    p1 = count_1 / length

    # 遷移カウント 00, 01, 10, 11
    d0 = data[:-1]
    d1 = data[1:]

    c00 = int(np.count_nonzero((d0 == 0) & (d1 == 0)))
    c01 = int(np.count_nonzero((d0 == 0) & (d1 == 1)))
    c10 = int(np.count_nonzero((d0 == 1) & (d1 == 0)))
    c11 = int(np.count_nonzero((d0 == 1) & (d1 == 1)))

    p00 = c00 / (c00 + c01) if (c00 + c01) > 0 else 0.0
    p01 = c01 / (c00 + c01) if (c00 + c01) > 0 else 0.0
    p10 = c10 / (c10 + c11) if (c10 + c11) > 0 else 0.0
    p11 = c11 / (c10 + c11) if (c10 + c11) > 0 else 0.0

    # 長さ128の最も確からしい6つの候補系列の確率を対数領域で計算
    # 1. 00...0: p0 * (p00 ^ 127)
    # 2. 0101...01: p0 * (p01 ^ 64) * (p10 ^ 63)
    # 3. 011...1: p0 * p01 * (p11 ^ 126)
    # 4. 100...0: p1 * p10 * (p00 ^ 126)
    # 5. 1010...10: p1 * (p10 ^ 64) * (p01 ^ 63)
    # 6. 11...1: p1 * (p11 ^ 127)
    def safe_log(p: float) -> float:
        return math.log2(p) if p > 0.0 else -float("inf")

    log_probs = [
        safe_log(p0) + 127 * safe_log(p00),
        safe_log(p0) + 64 * safe_log(p01) + 63 * safe_log(p10),
        safe_log(p0) + safe_log(p01) + 126 * safe_log(p11),
        safe_log(p1) + safe_log(p10) + 126 * safe_log(p00),
        safe_log(p1) + 64 * safe_log(p10) + 63 * safe_log(p01),
        safe_log(p1) + 127 * safe_log(p11),
    ]

    max_log_prob = max(log_probs)
    if max_log_prob == -float("inf"):
        return 0.0

    # min-entropy = min(-log2(p_max) / 128, 1.0)
    h = -max_log_prob / 128.0
    return min(1.0, max(0.0, h))
