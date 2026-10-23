"""
Section 4 ヘルステスト (Health Tests) モジュール
規格書 Section 4.4 に基づく実装
  - 4.4.1 Repetition Count Test (RCT)
  - 4.4.2 Adaptive Proportion Test (APT)
"""

import math
from typing import Generator, Tuple
import numpy as np
from scipy.stats import binom


def calc_rct_cutoff(h: float, alpha: float = 2.0 ** -20) -> int:
    """
    4.4.1 Repetition Count Test (RCT) のカットオフ値 C を算出する。
    C = 1 + ceil(-log2(alpha) / H)
    """
    if h <= 0.0:
        return 100_000
    log2_alpha = -math.log2(alpha)
    c = 1 + math.ceil(log2_alpha / h)
    return int(c)


def calc_apt_cutoff(h: float, is_binary: bool, alpha: float = 2.0 ** -20) -> Tuple[int, int]:
    """
    4.4.2 Adaptive Proportion Test (APT) の窓長 W とカットオフ値 C を算出する。
    W = 1024 (バイナリ), 512 (非バイナリ)
    C は Pr(B >= C) <= alpha を満たす最小の整数。
    CRITBINOM(W, 2^(-H), 1 - alpha) + 1 に対応。
    """
    w = 1024 if is_binary else 512
    p = 2.0 ** (-h)
    # 二項分布 B ~ Binomial(W, p) の上側確率点
    # scipy.stats.binom.ppf(1 - alpha, W, p)
    c = 1 + int(binom.ppf(1.0 - alpha, w, p))
    return w, int(c)


class HealthTester:
    """
    ストリームデータに対して RCT および APT を逐次評価するクラス
    """

    def __init__(self, h: float, is_binary: bool, alpha: float = 2.0 ** -20):
        self.h = h
        self.is_binary = is_binary
        self.rct_c = calc_rct_cutoff(h, alpha=alpha)
        self.apt_w, self.apt_c = calc_apt_cutoff(h, is_binary, alpha=alpha)

        # RCT 内部状態
        self.rct_last_a = None
        self.rct_count_b = 0

        # APT 内部状態
        self.apt_target_a = None
        self.apt_count_b = 0
        self.apt_window_idx = 0

    def test_sample(self, sample: int) -> Tuple[bool, bool]:
        """
        1つのサンプルを入力し、(rct_passed, apt_passed) の真偽値を返す。
        """
        # --- RCT ---
        rct_passed = True
        if self.rct_last_a is None:
            self.rct_last_a = sample
            self.rct_count_b = 1
        else:
            if sample == self.rct_last_a:
                self.rct_count_b += 1
                if self.rct_count_b >= self.rct_c:
                    rct_passed = False
            else:
                self.rct_last_a = sample
                self.rct_count_b = 1

        # --- APT ---
        apt_passed = True
        if self.apt_window_idx == 0:
            self.apt_target_a = sample
            self.apt_count_b = 1
            self.apt_window_idx = 1
        else:
            if sample == self.apt_target_a:
                self.apt_count_b += 1
                if self.apt_count_b >= self.apt_c:
                    apt_passed = False

            # バイナリ拡張: W - B >= C のチェック (Section 4.4.2)
            if self.is_binary and (self.apt_w - self.apt_count_b >= self.apt_c):
                apt_passed = False

            self.apt_window_idx += 1
            if self.apt_window_idx >= self.apt_w:
                self.apt_window_idx = 0

        return rct_passed, apt_passed
