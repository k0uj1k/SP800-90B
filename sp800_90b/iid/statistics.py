"""
Section 5.1 並べ替え検定の統計量 (11種) の計算モジュール
規格書 Section 5.1.1 〜 5.1.11 に基づく実装
"""

import bz2
from typing import List
import numpy as np


def calc_excursion(data: np.ndarray) -> float:
    """5.1.1 Excursion Test Statistic: 平均からの累積偏差の最大値"""
    mean_val = np.mean(data)
    running_sums = np.cumsum(data)
    i_indices = np.arange(1, len(data) + 1)
    deviations = np.abs(running_sums - i_indices * mean_val)
    return float(np.max(deviations))


def calc_directional_runs_and_lengths(data: np.ndarray) -> (int, int, int):
    """
    5.1.2 Number of Directional Runs
    5.1.3 Length of Directional Runs
    5.1.4 Number of Increases and Decreases
    S'_i = -1 (if S_i > S_{i+1}) else +1
    """
    diffs = np.diff(data)
    # s'_i: if diff < 0 (i.e. s_i > s_{i+1}) -> -1, else -> +1
    s_prime = np.where(diffs < 0, -1, 1)

    # 5.1.4: max(count(-1), count(+1))
    c_minus = int(np.count_nonzero(s_prime == -1))
    c_plus = len(s_prime) - c_minus
    num_inc_dec = max(c_minus, c_plus)

    # ラン（連続同一値の区間）を特定
    if len(s_prime) == 0:
        return 0, 0, 0

    # 変化点を特定
    changes = np.diff(s_prime) != 0
    num_runs = int(np.count_nonzero(changes)) + 1

    # 各ランの長さを測定
    run_change_indices = np.where(changes)[0] + 1
    split_points = np.concatenate(([0], run_change_indices, [len(s_prime)]))
    run_lengths = np.diff(split_points)
    max_run_length = int(np.max(run_lengths)) if len(run_lengths) > 0 else 0

    return num_runs, max_run_length, num_inc_dec


def calc_median_runs(data: np.ndarray, is_binary: bool = False) -> (int, int):
    """
    5.1.5 Number of Runs Based on the Median
    5.1.6 Length of Runs Based on Median
    s'_i = -1 if s_i < X_tilde else +1
    """
    if is_binary:
        median_val = 0.5
    else:
        median_val = float(np.median(data))

    s_prime = np.where(data < median_val, -1, 1)
    if len(s_prime) == 0:
        return 0, 0

    changes = np.diff(s_prime) != 0
    num_runs = int(np.count_nonzero(changes)) + 1

    run_change_indices = np.where(changes)[0] + 1
    split_points = np.concatenate(([0], run_change_indices, [len(s_prime)]))
    run_lengths = np.diff(split_points)
    max_run_length = int(np.max(run_lengths)) if len(run_lengths) > 0 else 0

    return num_runs, max_run_length


def calc_collision_statistics(data: np.ndarray) -> (float, int):
    """
    5.1.7 Average Collision Test Statistic
    5.1.8 Maximum Collision Test Statistic
    """
    length = len(data)
    c_list = []
    i = 0
    data_list = data.tolist()

    while i < length:
        seen = set()
        j = 0
        while i + j < length:
            val = data_list[i + j]
            if val in seen:
                break
            seen.add(val)
            j += 1

        if i + j < length:
            c_list.append(j + 1)
            i += (j + 1)
        else:
            break

    if not c_list:
        return 0.0, 0

    avg_collision = float(np.mean(c_list))
    max_collision = int(np.max(c_list))
    return avg_collision, max_collision


def calc_periodicity(data: np.ndarray, lag: int) -> int:
    """5.1.9 Periodicity Test Statistic: s_i == s_{i+p} となる回数"""
    if lag >= len(data):
        return 0
    d1 = data[:-lag]
    d2 = data[lag:]
    return int(np.count_nonzero(d1 == d2))


def calc_covariance(data: np.ndarray, lag: int) -> float:
    """5.1.10 Covariance Test Statistic: sum(s_i * s_{i+p})"""
    if lag >= len(data):
        return 0.0
    d1 = data[:-lag].astype(np.float64)
    d2 = data[lag:].astype(np.float64)
    return float(np.sum(d1 * d2))


def calc_compression(data: np.ndarray) -> int:
    """
    5.1.11 Compression Test Statistic:
    空白区切り文字列にして bzip2 で圧縮した後のバイト長
    """
    str_data = " ".join(str(int(x)) for x in data)
    compressed = bz2.compress(str_data.encode("ascii"))
    return len(compressed)


def compute_all_permutation_statistics(data: np.ndarray, is_binary: bool) -> List[float]:
    """
    規格書 Section 5.1 に規定された順序で全統計量を計算して返す。
    返されるリストの順序 (計19個の統計量):
      0: Excursion
      1: Number of Directional Runs
      2: Length of Directional Runs
      3: Number of Increases and Decreases
      4: Number of Runs Based on Median
      5: Length of Runs Based on Median
      6: Average Collision
      7: Maximum Collision
      8..12: Periodicity (lag = 1, 2, 8, 16, 32)
      13..17: Covariance (lag = 1, 2, 8, 16, 32)
      18: Compression
    """
    stats = []

    # 1. Excursion
    stats.append(calc_excursion(data))

    # 2..4 Directional
    n_dir_runs, len_dir_runs, n_inc_dec = calc_directional_runs_and_lengths(data)
    stats.append(float(n_dir_runs))
    stats.append(float(len_dir_runs))
    stats.append(float(n_inc_dec))

    # 5..6 Median
    n_med_runs, len_med_runs = calc_median_runs(data, is_binary=is_binary)
    stats.append(float(n_med_runs))
    stats.append(float(len_med_runs))

    # 7..8 Collision
    avg_col, max_col = calc_collision_statistics(data)
    stats.append(avg_col)
    stats.append(float(max_col))

    # 9..13 Periodicity (1, 2, 8, 16, 32)
    lags = [1, 2, 8, 16, 32]
    for lag in lags:
        stats.append(float(calc_periodicity(data, lag)))

    # 14..18 Covariance (1, 2, 8, 16, 32)
    for lag in lags:
        stats.append(calc_covariance(data, lag))

    # 19 Compression
    stats.append(float(calc_compression(data)))

    return stats
