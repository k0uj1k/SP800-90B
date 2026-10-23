"""
非バイナリ（シンボル）データに対する拡張統計推定器モジュール
Collision, Markov, Compression の理論を任意シンボルアルファベットに拡張適用する。
"""

import math
from collections import Counter, defaultdict
from typing import Dict, Optional
import numpy as np


def estimate_symbol_collision(data: np.ndarray, k: int) -> float:
    """
    非バイナリ（任意シンボル空間 k）に対する拡張衝突推定器
    重複シンボルが現れるまでのステップ数の標本平均から最小エントロピーを推定する。
    """
    length = len(data)
    if length < 2 or k <= 1:
        return 0.0

    t_list = []
    index = 0
    data_list = data.tolist()

    while index < length:
        seen = set()
        j = index
        while j < length:
            val = data_list[j]
            if val in seen:
                break
            seen.add(val)
            j += 1

        if j < length:
            t_v = j - index + 1
            t_list.append(t_v)
            index = j + 1
        else:
            break

    if len(t_list) <= 1:
        return math.log2(k)

    t_arr = np.array(t_list, dtype=np.float64)
    x_bar = np.mean(t_arr)
    s_hat = np.std(t_arr, ddof=1)
    v = len(t_list)

    # 99%信頼区間下限
    x_bar_prime = x_bar - 2.576 * (s_hat / math.sqrt(v))

    # 一様分布での衝突までの平均期待ステップ数は約 sqrt(pi * k / 2)
    # 最悪ケース（偏り大）ではステップ数が小さくなる
    # 衝突エントロピー近似: H_col ≈ 2 * log2(X_bar_prime) - log2(pi / 2)
    try:
        if x_bar_prime <= 2.0:
            return 0.0
        h_est = 2.0 * math.log2(x_bar_prime) - math.log2(math.pi / 2.0)
        max_h = math.log2(k)
        return min(max_h, max(0.0, h_est))
    except Exception:
        return 0.0


def estimate_symbol_markov(data: np.ndarray, k: int) -> float:
    """
    非バイナリ（任意シンボル空間 k）に対する拡張マルコフ推定器
    k x k の状態遷移行列を構築し、最も高確率な遷移系列からエントロピーを算出する。
    """
    length = len(data)
    if length < 2:
        return 0.0

    # 状態出現回数
    unique_vals, counts = np.unique(data, return_counts=True)
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    num_states = len(unique_vals)

    if num_states <= 1:
        return 0.0

    p_init = counts / length

    # 遷移行列
    trans_counts = np.zeros((num_states, num_states), dtype=np.float64)
    data_list = data.tolist()
    for i in range(length - 1):
        u = val_to_idx[data_list[i]]
        v = val_to_idx[data_list[i + 1]]
        trans_counts[u, v] += 1

    row_sums = trans_counts.sum(axis=1, keepdims=True)
    # ゼロ除算回避
    with np.errstate(divide="ignore", invalid="ignore"):
        trans_prob = np.where(row_sums > 0, trans_counts / row_sums, 0.0)

    # 長さ128の系列における最尤確率パスの探索（Viterbi類似の対数探索、または最大自己ループ/遷移）
    # 各行の最大遷移確率
    max_trans = np.max(trans_prob, axis=1)
    # 最も初期確率 * 最大遷移^127 が大きくなる状態
    best_log_p = -float("inf")
    for i in range(num_states):
        if p_init[i] > 0 and max_trans[i] > 0:
            log_p = math.log2(p_init[i]) + 127.0 * math.log2(max_trans[i])
            if log_p > best_log_p:
                best_log_p = log_p

    if best_log_p == -float("inf"):
        return 0.0

    h = -best_log_p / 128.0
    max_h = math.log2(k)
    return min(max_h, max(0.0, h))


def estimate_symbol_compression(data: np.ndarray, k: int) -> float:
    """
    非バイナリ（任意シンボル空間 k）に対する拡張 Maurer 圧縮推定器
    各シンボルそのものを辞書項目として直近の出現間隔の対数平均を計算する。
    """
    length = len(data)
    d = min(1000, length // 4)
    if length <= d * 2:
        return math.log2(k)

    last_seen: Dict[int, int] = {}
    data_list = data.tolist()

    # 辞書初期化
    for i in range(d):
        last_seen[data_list[i]] = i + 1

    v = length - d
    log2_d_list = []
    for i in range(d, length):
        val = data_list[i]
        curr_idx = i + 1
        prev_idx = last_seen.get(val, 0)
        dist = (curr_idx - prev_idx) if prev_idx != 0 else curr_idx
        last_seen[val] = curr_idx
        log2_d_list.append(math.log2(dist))

    log2_arr = np.array(log2_d_list, dtype=np.float64)
    x_bar = np.mean(log2_arr)
    s_hat = 0.5907 * math.sqrt(np.sum((log2_arr - x_bar) ** 2) / (v - 1))
    x_bar_prime = x_bar - 2.576 * (s_hat / math.sqrt(v))

    # Maurer統計量からのエントロピー推定値
    max_h = math.log2(k)
    h_est = x_bar_prime - 1.0  # 近似スケール
    return min(max_h, max(0.0, h_est))
