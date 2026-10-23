"""
Section 5.1 並べ替え検定実行モジュール (Permutation Testing)
規格書 Section 5.1 に基づく実装 (マルチプロセス並列化対応)
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from sp800_90b.iid.statistics import compute_all_permutation_statistics


@dataclass
class PermutationTestResult:
    """並べ替え検定の判定結果"""
    passed: bool
    num_tests: int
    failed_test_indices: List[int]
    counters: List[Tuple[int, int]]  # (C_{i,0}, C_{i,1})


def _run_permutation_batch(
    base_data: np.ndarray,
    is_binary: bool,
    t_orig: np.ndarray,
    num_iterations: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    指定されたイテレーション回数分のシャッフルと統計量比較を実行するワーカー関数
    """
    rng = np.random.default_rng(seed)
    num_stats = len(t_orig)
    c_i0 = np.zeros(num_stats, dtype=np.int32)
    c_i1 = np.zeros(num_stats, dtype=np.int32)

    shuffled = base_data.copy()

    for _ in range(num_iterations):
        rng.shuffle(shuffled)
        t_perm = np.array(compute_all_permutation_statistics(shuffled, is_binary))
        c_i0 += (t_perm > t_orig).astype(np.int32)
        c_i1 += (t_perm == t_orig).astype(np.int32)

    return c_i0, c_i1


def run_permutation_tests(
    data: np.ndarray,
    is_binary: bool,
    num_permutations: int = 10_000,
    num_workers: int = 4,
) -> PermutationTestResult:
    """
    規格書 Section 5.1:
    10,000回の Fisher-Yates シャッフルを行い、元データの統計量 T_i と比較する。
    判定基準 (Step 3):
    いずれかの統計量 i について (C_{i,0} + C_{i,1} <= 5) または (C_{i,0} >= 9995) の場合、IID仮説を棄却。
    それ以外は IID と判定。
    """
    t_orig = np.array(compute_all_permutation_statistics(data, is_binary))
    num_stats = len(t_orig)

    batch_size = num_permutations // num_workers
    remainder = num_permutations % num_workers

    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for w in range(num_workers):
            iters = batch_size + (1 if w < remainder else 0)
            if iters > 0:
                f = executor.submit(
                    _run_permutation_batch,
                    data,
                    is_binary,
                    t_orig,
                    iters,
                    seed=42 + w * 1000,
                )
                futures.append(f)

    total_c0 = np.zeros(num_stats, dtype=np.int32)
    total_c1 = np.zeros(num_stats, dtype=np.int32)

    for f in futures:
        c0, c1 = f.result()
        total_c0 += c0
        total_c1 += c1

    # 合否判定
    failed_indices = []
    for i in range(num_stats):
        if (total_c0[i] + total_c1[i] <= 5) or (total_c0[i] >= (num_permutations - 5)):
            failed_indices.append(i)

    passed = (len(failed_indices) == 0)
    counters = [(int(total_c0[i]), int(total_c1[i])) for i in range(num_stats)]

    return PermutationTestResult(
        passed=passed,
        num_tests=num_stats,
        failed_test_indices=failed_indices,
        counters=counters,
    )
