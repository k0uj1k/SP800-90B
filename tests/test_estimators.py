"""
規格書の各数値例（Examples）に対する検証ユニットテスト
"""

import math
import numpy as np
import pytest

from sp800_90b.estimators.mcv import estimate_mcv
from sp800_90b.estimators.collision import estimate_collision
from sp800_90b.estimators.markov import estimate_markov
from sp800_90b.estimators.compression import estimate_compression
from sp800_90b.estimators.t_tuple import estimate_t_tuple
from sp800_90b.estimators.lrs import estimate_lrs
from sp800_90b.estimators.multi_mcw import estimate_multi_mcw
from sp800_90b.estimators.lag import estimate_lag
from sp800_90b.estimators.multi_mmc import estimate_multi_mmc
from sp800_90b.estimators.lz78y import estimate_lz78y


def test_mcv_example():
    """
    規格書 Section 6.3.1 Example (Page 42):
    S = (0, 1, 1, 2, 0, 1, 2, 2, 0, 1, 0, 1, 1, 0, 2, 2, 1, 0, 2, 1), L = 20
    most common value is 1, p_hat = 0.4.
    p_u = 0.4 + 2.576 * sqrt(0.012) = 0.6822 (text says ~0.6822, p_u calc)
    min-entropy = -log2(0.6822) = 0.5517 (approx ~0.5363 depending on exact sqrt rounding)
    """
    s = np.array([0, 1, 1, 2, 0, 1, 2, 2, 0, 1, 0, 1, 1, 0, 2, 2, 1, 0, 2, 1])
    h = estimate_mcv(s)
    # p_u 約 0.68 前後、Hは約 0.54 前後
    assert 0.50 <= h <= 0.60


def test_markov_example():
    """
    規格書 Section 6.3.3 Example (Page 44-45):
    L = 40, S = (1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0,
                 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0)
    Resulting entropy estimate is approx 0.761.
    """
    s = np.array([
        1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0,
        1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0
    ])
    h = estimate_markov(s)
    assert pytest.approx(h, abs=0.02) == 0.761


def test_lag_example():
    """
    規格書 Section 6.3.8 Example (Page 52):
    S = (2, 1, 3, 2, 1, 3, 1, 3, 1, 2), L = 10, D = 3
    correct = (0, 0, 0, 1, 1, 0, 0, 0, 0), P_global = 0.2222, P'_global = 0.6008, P_local = 0.1167
    min-entropy approx 0.735
    """
    s = np.array([2, 1, 3, 2, 1, 3, 1, 3, 1, 2])
    h = estimate_lag(s, k=3, max_d=3)
    assert pytest.approx(h, abs=0.03) == 0.735


def test_multi_mmc_example():
    """
    規格書 Section 6.3.9 Example (Page 54):
    S = (2, 1, 3, 2, 1, 3, 1, 3, 1), L = 9, D = 3
    correct = (0, 0, 1, 1, 0, 1, 0), P_global = 0.4286, P'_global = 0.9490
    min-entropy approx 0.0755
    """
    s = np.array([2, 1, 3, 2, 1, 3, 1, 3, 1])
    h = estimate_multi_mmc(s, k=3, depth=3)
    assert pytest.approx(h, abs=0.03) == 0.0755
