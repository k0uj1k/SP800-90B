"""
テスト用乱数生成モジュール (Synthetic Data Generator)
テスト・動作確認用のデータセット（完全乱数、バイアス、マルコフ、リスタート行列など）を生成する。
"""

import numpy as np


def generate_uniform_data(length: int = 1_000_000, bits_per_symbol: int = 8, seed: int = 42) -> np.ndarray:
    """一様乱数（理想的なエントロピー源）を生成する"""
    rng = np.random.default_rng(seed)
    max_val = 1 << bits_per_symbol
    return rng.integers(0, max_val, size=length, dtype=np.uint32)


def generate_biased_data(
    length: int = 1_000_000,
    bits_per_symbol: int = 8,
    p_bias: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """特定のシンボルに偏り（バイアス）のあるデータを生成する"""
    rng = np.random.default_rng(seed)
    max_val = 1 << bits_per_symbol
    # シンボル0を確率 p_bias で生成し、残りを均等に配分
    p_rest = (1.0 - p_bias) / (max_val - 1)
    probs = np.full(max_val, p_rest)
    probs[0] = p_bias
    return rng.choice(max_val, size=length, p=probs).astype(np.uint32)


def generate_markov_data(
    length: int = 1_000_000,
    p00: float = 0.7,
    p11: float = 0.7,
    seed: int = 42,
) -> np.ndarray:
    """相関を持つバイナリマルコフ系列を生成する"""
    rng = np.random.default_rng(seed)
    data = np.zeros(length, dtype=np.uint8)
    curr = rng.integers(0, 2)
    data[0] = curr

    for i in range(1, length):
        r = rng.random()
        if curr == 0:
            curr = 0 if r < p00 else 1
        else:
            curr = 1 if r < p11 else 0
        data[i] = curr

    return data


def generate_restart_matrix(
    rows: int = 1000,
    cols: int = 1000,
    bits_per_symbol: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """1000 x 1000 のリスタート行列を生成する"""
    rng = np.random.default_rng(seed)
    max_val = 1 << bits_per_symbol
    return rng.integers(0, max_val, size=(rows, cols), dtype=np.uint32)
