"""
情報理論・拡張エントロピー指標計算モジュール
- シャノンエントロピー (Shannon Entropy: H_1)
- レニーエントロピー (Rényi Entropy: H_alpha)
- 衝突エントロピー (Collision Entropy / Rényi order 2: H_2)
- 最小エントロピー理論値 (Min-Entropy Theoretical: H_infinity = -log2(max p_i))
- エントロピー効率 (Entropy Efficiency: H_shannon / n)
"""

import math
from typing import Dict, List, Optional
import numpy as np


def compute_shannon_entropy(data: np.ndarray) -> float:
    """
    シャノンエントロピー (Shannon Entropy):
    H = - sum_{i} p_i * log2(p_i)
    """
    length = len(data)
    if length == 0:
        return 0.0

    _, counts = np.unique(data, return_counts=True)
    probs = counts / length
    # p_i * log2(p_i)
    shannon = -np.sum(probs * np.log2(probs))
    return float(shannon)


def compute_renyi_entropy(data: np.ndarray, alpha: float) -> float:
    """
    レニーエントロピー (Rényi Entropy of order alpha):
    H_alpha = 1 / (1 - alpha) * log2( sum_{i} p_i^alpha )
    alpha = 1 の極限はシャノンエントロピー。
    alpha = 2 は衝突エントロピー。
    alpha -> infinity は最小エントロピー。
    """
    length = len(data)
    if length == 0:
        return 0.0

    if math.isclose(alpha, 1.0, rel_tol=1e-5):
        return compute_shannon_entropy(data)

    _, counts = np.unique(data, return_counts=True)
    probs = counts / length

    if alpha == float("inf"):
        return -math.log2(np.max(probs))

    sum_p_alpha = np.sum(probs ** alpha)
    if sum_p_alpha <= 0.0:
        return 0.0

    renyi = (1.0 / (1.0 - alpha)) * math.log2(sum_p_alpha)
    return float(renyi)


def compute_collision_entropy(data: np.ndarray) -> float:
    """
    衝突エントロピー (Collision Entropy / Rényi order 2: H_2):
    H_2 = -log2( sum_{i} p_i^2 )
    """
    return compute_renyi_entropy(data, alpha=2.0)


def compute_theoretical_min_entropy(data: np.ndarray) -> float:
    """
    経験分布に対する理論上の最小エントロピー (Min-Entropy: H_inf):
    H_inf = -log2( max_i p_i )
    ※ 信頼区間上限 (Z=2.576) を適用する前の純粋な最頻値エントロピー
    """
    length = len(data)
    if length == 0:
        return 0.0
    _, counts = np.unique(data, return_counts=True)
    max_p = np.max(counts) / length
    return -math.log2(max_p) if max_p > 0.0 else 0.0


def compute_extended_entropies(data: np.ndarray, bits_per_symbol: int) -> Dict[str, float]:
    """
    全情報理論エントロピー指標を一括計算する
    """
    max_entropy = float(bits_per_symbol)
    h_shannon = compute_shannon_entropy(data)
    h_collision = compute_collision_entropy(data)
    h_renyi_0_5 = compute_renyi_entropy(data, alpha=0.5)
    h_renyi_3_0 = compute_renyi_entropy(data, alpha=3.0)
    h_min_raw = compute_theoretical_min_entropy(data)

    efficiency = (h_shannon / max_entropy) if max_entropy > 0 else 0.0

    return {
        "shannon": h_shannon,
        "collision_h2": h_collision,
        "renyi_alpha_0_5": h_renyi_0_5,
        "renyi_alpha_3_0": h_renyi_3_0,
        "min_entropy_raw": h_min_raw,
        "efficiency": efficiency,
    }
