"""
Section 6.3.4 圧縮推定器 (The Compression Estimate)
規格書 Section 6.3.4 に基づく実装 (Maurer's Universal Statistic, バイナリデータ専用)
"""

import math
import numpy as np


def estimate_compression(data: np.ndarray, b: int = 6, d: int = 1000) -> float:
    """
    圧縮推定器 (The Compression Estimate)
    b=6ビットのブロックに分割し、辞書サイズ d=1000 で Maurer 統計量を計算、
    二分探索でパラメータ p を解いてエントロピーを推定する。
    """
    length = len(data)
    num_blocks = length // b
    if num_blocks <= d:
        return 1.0

    # bビットごとに整数ブロック化
    bits = data[: num_blocks * b].reshape(-1, b)
    powers = 2 ** np.arange(b - 1, -1, -1, dtype=np.uint32)
    s_prime = np.dot(bits, powers)

    # 辞書初期化: サイズ 2^b
    dict_size = 1 << b
    last_seen = np.zeros(dict_size, dtype=np.int32)

    # 最初の d 個を辞書に登録 (1-indexed)
    for i in range(d):
        val = s_prime[i]
        last_seen[val] = i + 1

    # テストデータ (d 個以降) に対する Maurer 距離 D_i の対数を計算
    v = num_blocks - d
    log2_d_list = []

    for i in range(d, num_blocks):
        val = s_prime[i]
        curr_idx = i + 1
        prev_idx = last_seen[val]
        if prev_idx != 0:
            dist = curr_idx - prev_idx
        else:
            dist = curr_idx
        last_seen[val] = curr_idx
        log2_d_list.append(math.log2(dist))

    log2_arr = np.array(log2_d_list, dtype=np.float64)
    x_bar = np.mean(log2_arr)

    # 規格書 Section 6.3.4 Step 5:
    # c = 0.5907 (相関補正係数)
    c_factor = 0.5907
    s_hat = c_factor * math.sqrt(np.sum((log2_arr - x_bar) ** 2) / (v - 1))

    # 99%信頼区間下限
    x_bar_prime = x_bar - 2.576 * (s_hat / math.sqrt(v))

    # 二分探索による p の算出
    # G(z) = (1/v) * sum_{t=d+1}^{num_blocks} sum_{u=1}^t log2(u) * F(z, t, u)
    # 規格書 Step 7: X'_bar = G(p) + (2^b - 1) * G(q), where q = (1 - p) / (2^b - 1)
    # 実用的かつ高精度な近似表 / 直線探索
    # G(z) は z が大きくなると (最頻値が偏ると) 小さくなる
    # p in [2^(-b), 1.0]

    # 二分探索の境界
    low = 2.0 ** (-b)
    high = 1.0

    # 期待値の高速計算 (事前計算キャッシュ)
    # v が大きい場合、G(z) は sum_{u=1}^L log2(u) * z^2 * (1-z)^(u-1) に漸近する
    def compute_g(z: float) -> float:
        if z <= 0.0:
            return 0.0
        if z >= 1.0:
            return 0.0
        # u=1..50 程度で級数が急速に減衰 ( (1-z)^u -> 0 )
        total = 0.0
        one_minus_z = 1.0 - z
        prob = z * z
        for u in range(1, min(2000, num_blocks + 1)):
            if u > 1:
                prob *= one_minus_z
                if prob < 1e-15:
                    break
            total += math.log2(u) * prob
        return total

    def expected_x_maurer(p_val: float) -> float:
        q_val = (1.0 - p_val) / ((1 << b) - 1)
        return compute_g(p_val) + ((1 << b) - 1) * compute_g(q_val)

    # 境界チェック
    x_at_low = expected_x_maurer(low)
    x_at_high = expected_x_maurer(high)

    if x_bar_prime >= x_at_low:
        return 1.0
    if x_bar_prime <= x_at_high:
        return 0.0

    for _ in range(50):
        mid = (low + high) / 2.0
        val = expected_x_maurer(mid)
        if val > x_bar_prime:
            low = mid
        else:
            high = mid

    p_sol = (low + high) / 2.0
    h = -math.log2(p_sol) / b
    return min(1.0, max(0.0, h))
