"""
Section 5.2 カイ二乗検定および最長重複部分列検定モジュール
規格書 Section 5.2.1 〜 5.2.5 に基づく実装
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.stats import chi2


@dataclass
class ChiSquareTestResult:
    """カイ二乗検定・IID検定の結果"""
    passed: bool
    t_statistic: float
    critical_value: float
    p_value: float
    degrees_of_freedom: int


def test_independence_non_binary(data: np.ndarray, k: int, alpha: float = 0.001) -> ChiSquareTestResult:
    """
    5.2.1 非バイナリデータの独立性カイ二乗検定
    """
    length = len(data)
    unique_vals, counts = np.unique(data, return_counts=True)
    val_to_p = {val: cnt / length for val, cnt in zip(unique_vals, counts)}

    # ペアの期待値 e_{i,j} = p_i * p_j * L / 2
    pair_expected = []
    for v1 in unique_vals:
        for v2 in unique_vals:
            e_val = val_to_p[v1] * val_to_p[v2] * length / 2.0
            pair_expected.append(((v1, v2), e_val))

    # 期待値の昇順にソートして各ビンの合計が >= 5 となるようにビン分割
    pair_expected.sort(key=lambda x: x[1])

    bins: List[List[Tuple[int, int]]] = []
    bin_exp: List[float] = []

    curr_bin: List[Tuple[int, int]] = []
    curr_sum = 0.0

    for pair, e in pair_expected:
        curr_bin.append(pair)
        curr_sum += e
        if curr_sum >= 5.0:
            bins.append(curr_bin)
            bin_exp.append(curr_sum)
            curr_bin = []
            curr_sum = 0.0

    if curr_bin:
        if not bin_exp:
            bins.append(curr_bin)
            bin_exp.append(curr_sum)
        else:
            # 末尾のビンとマージ
            bins[-1].extend(curr_bin)
            bin_exp[-1] += curr_sum

    n_bins = len(bins)
    df = n_bins - k
    if df < 1:
        return ChiSquareTestResult(passed=True, t_statistic=0.0, critical_value=0.0, p_value=1.0, degrees_of_freedom=df)

    # 観測度数 o_i の集計 (非重複ペア: j=0, 2, 4...)
    pair_to_bin = {}
    for b_idx, b_pairs in enumerate(bins):
        for p in b_pairs:
            pair_to_bin[p] = b_idx

    obs = np.zeros(n_bins, dtype=np.int32)
    data_list = data.tolist()
    for j in range(0, length - 1, 2):
        pair = (data_list[j], data_list[j + 1])
        if pair in pair_to_bin:
            obs[pair_to_bin[pair]] += 1

    t_stat = float(np.sum(((obs - bin_exp) ** 2) / bin_exp))
    crit_val = float(chi2.ppf(1.0 - alpha, df))
    p_val = float(1.0 - chi2.cdf(t_stat, df))

    return ChiSquareTestResult(
        passed=(t_stat <= crit_val),
        t_statistic=t_stat,
        critical_value=crit_val,
        p_value=p_val,
        degrees_of_freedom=df,
    )


def test_goodness_of_fit_non_binary(data: np.ndarray, k: int, alpha: float = 0.001) -> ChiSquareTestResult:
    """
    5.2.2 非バイナリデータの10分割適合度カイ二乗検定
    """
    length = len(data)
    unique_vals, counts = np.unique(data, return_counts=True)
    # e_i = c_i / 10
    e_dict = {val: cnt / 10.0 for val, cnt in zip(unique_vals, counts)}
    sorted_vals = sorted(unique_vals, key=lambda v: e_dict[v])

    bins: List[List[int]] = []
    bin_exp: List[float] = []

    curr_bin: List[int] = []
    curr_sum = 0.0

    for val in sorted_vals:
        curr_bin.append(val)
        curr_sum += e_dict[val]
        if curr_sum >= 5.0:
            bins.append(curr_bin)
            bin_exp.append(curr_sum)
            curr_bin = []
            curr_sum = 0.0

    if curr_bin:
        if not bin_exp:
            bins.append(curr_bin)
            bin_exp.append(curr_sum)
        else:
            bins[-1].extend(curr_bin)
            bin_exp[-1] += curr_sum

    n_bins = len(bins)
    df = 9 * (n_bins - 1)
    if df < 1:
        return ChiSquareTestResult(passed=True, t_statistic=0.0, critical_value=0.0, p_value=1.0, degrees_of_freedom=df)

    val_to_bin = {}
    for b_idx, b_vals in enumerate(bins):
        for v in b_vals:
            val_to_bin[v] = b_idx

    sub_len = length // 10
    total_t = 0.0

    for d in range(10):
        sub_data = data[d * sub_len : (d + 1) * sub_len]
        obs = np.zeros(n_bins, dtype=np.int32)
        for v in sub_data:
            if v in val_to_bin:
                obs[val_to_bin[v]] += 1
        total_t += float(np.sum(((obs - bin_exp) ** 2) / bin_exp))

    crit_val = float(chi2.ppf(1.0 - alpha, df))
    p_val = float(1.0 - chi2.cdf(total_t, df))

    return ChiSquareTestResult(
        passed=(total_t <= crit_val),
        t_statistic=total_t,
        critical_value=crit_val,
        p_value=p_val,
        degrees_of_freedom=df,
    )


def test_independence_binary(data: np.ndarray, alpha: float = 0.001) -> ChiSquareTestResult:
    """
    5.2.3 バイナリデータの独立性カイ二乗検定 (m-bit タプル比較)
    """
    length = len(data)
    p1 = float(np.count_nonzero(data == 1)) / length
    p0 = 1.0 - p1

    # 最大の m を見つける (min(p0, p1)^m * floor(L/m) >= 5, m <= 11)
    min_p = min(p0, p1)
    m = 1
    for cand_m in range(2, 12):
        exp_count = (min_p ** cand_m) * (length // cand_m)
        if exp_count >= 5.0:
            m = cand_m
        else:
            break

    if m < 2:
        return ChiSquareTestResult(passed=False, t_statistic=float("inf"), critical_value=0.0, p_value=0.0, degrees_of_freedom=0)

    num_blocks = length // m
    blocks = data[: num_blocks * m].reshape(-1, m)
    powers = 2 ** np.arange(m - 1, -1, -1, dtype=np.uint32)
    tuple_integers = np.dot(blocks, powers)

    # 観測度数
    counts = Counter(tuple_integers)

    # 全 2^m 通りのタプルの期待値と統計量を加算
    total_t = 0.0
    for val in range(1 << m):
        # 1のビット数をカウント
        w = bin(val).count("1")
        e = (p1 ** w) * (p0 ** (m - w)) * num_blocks
        o = counts.get(val, 0)
        total_t += ((o - e) ** 2) / e

    df = (1 << m) - 2
    crit_val = float(chi2.ppf(1.0 - alpha, df))
    p_val = float(1.0 - chi2.cdf(total_t, df))

    return ChiSquareTestResult(
        passed=(total_t <= crit_val),
        t_statistic=total_t,
        critical_value=crit_val,
        p_value=p_val,
        degrees_of_freedom=df,
    )


def test_goodness_of_fit_binary(data: np.ndarray, alpha: float = 0.001) -> ChiSquareTestResult:
    """
    5.2.4 バイナリデータの10分割適合度カイ二乗検定
    """
    length = len(data)
    p = float(np.count_nonzero(data == 1)) / length
    sub_len = length // 10

    e0 = (1.0 - p) * sub_len
    e1 = p * sub_len

    total_t = 0.0
    for d in range(10):
        sub_data = data[d * sub_len : (d + 1) * sub_len]
        o1 = float(np.count_nonzero(sub_data == 1))
        o0 = float(len(sub_data) - o1)
        total_t += ((o0 - e0) ** 2) / e0 + ((o1 - e1) ** 2) / e1

    df = 9
    crit_val = 27.887  # chi2.ppf(0.999, 9)
    p_val = float(1.0 - chi2.cdf(total_t, df))

    return ChiSquareTestResult(
        passed=(total_t <= crit_val),
        t_statistic=total_t,
        critical_value=crit_val,
        p_value=p_val,
        degrees_of_freedom=df,
    )


def test_lrs_iid(data: np.ndarray, alpha: float = 0.001) -> bool:
    """
    5.2.5 最長重複部分列 (LRS) 検定
    W: 2回以上重複する最大部分列長
    P(X >= 1) = 1 - (1 - p_col^W)^N < 0.001 の場合棄却 (Fail)
    """
    length = len(data)
    _, counts = np.unique(data, return_counts=True)
    p_col = float(np.sum((counts / length) ** 2))

    # 最長重複部分文字列長 W を二分探索で探索
    data_list = data.tolist()
    low = 1
    high = min(100, length)
    w_best = 1

    while low <= high:
        mid = (low + high) // 2
        seen = set()
        found_dup = False
        for i in range(length - mid + 1):
            sub = tuple(data_list[i : i + mid])
            if sub in seen:
                found_dup = True
                break
            seen.add(sub)

        if found_dup:
            w_best = mid
            low = mid + 1
        else:
            high = mid - 1

    w = w_best
    num_subseqs = length - w + 1
    n_pairs = (num_subseqs * (num_subseqs - 1)) // 2

    # Pr(X >= 1) = 1 - (1 - p_col^w)^N
    # 対数領域で log(1 - p_col^w) を近似: -p_col^w
    # (1 - p_col^w)^N ≈ exp(-N * p_col^w)
    log_p_col_w = w * math.log(p_col) if p_col > 0.0 else -float("inf")
    lambda_val = n_pairs * math.exp(log_p_col_w)

    # Pr(X >= 1) = 1 - exp(-lambda_val)
    if lambda_val >= 700:
        pr_x_ge_1 = 1.0
    else:
        pr_x_ge_1 = 1.0 - math.exp(-lambda_val)

    return pr_x_ge_1 >= alpha
