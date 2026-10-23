"""
時系列・スライディングウィンドウによるエントロピー推移分析モジュール
時系列で乱数データをブロック（窓）ごとに分割し、
エントロピーの経時変化・ドリフト・局所的劣化を検出・追跡する。
"""

import math
from typing import Dict, List
import numpy as np

from sp800_90b.estimators.extended import (
    compute_shannon_entropy,
    compute_collision_entropy,
    compute_theoretical_min_entropy,
)
from sp800_90b.estimators.mcv import estimate_mcv


def analyze_sliding_window_entropy(
    data: np.ndarray,
    num_windows: int = 50,
) -> Dict[str, List[float]]:
    """
    データ全体を num_windows 個の連続区間に分割し、
    各区間におけるシャノンエントロピー、衝突エントロピー、MCV最小エントロピーの推移を計算する。
    """
    length = len(data)
    if length < num_windows:
        return {"timestamps": [], "shannon": [], "collision": [], "mcv": []}

    window_size = length // num_windows
    shannon_series = []
    collision_series = []
    mcv_series = []
    window_labels = []

    for w in range(num_windows):
        start = w * window_size
        end = start + window_size
        slice_data = data[start:end]

        sh_val = compute_shannon_entropy(slice_data)
        col_val = compute_collision_entropy(slice_data)
        mcv_val = estimate_mcv(slice_data)

        shannon_series.append(round(sh_val, 4))
        collision_series.append(round(col_val, 4))
        mcv_series.append(round(mcv_val, 4))
        window_labels.append(f"W{w+1} ({start:,}-{end:,})")

    return {
        "labels": window_labels,
        "shannon": shannon_series,
        "collision": collision_series,
        "mcv": mcv_series,
    }
