"""
Section 6 最小エントロピー推定器の統合モジュール
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

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
from sp800_90b.estimators.symbol_extended import (
    estimate_symbol_collision,
    estimate_symbol_markov,
    estimate_symbol_compression,
)


@dataclass
class EntropyAssessmentResult:
    """エントロピー推定結果を格納するデータクラス"""
    is_binary: bool
    k: int
    mcv: float
    collision: Optional[float] = None
    markov: Optional[float] = None
    compression: Optional[float] = None
    t_tuple: Optional[float] = None
    lrs: Optional[float] = None
    multi_mcw: Optional[float] = None
    lag: Optional[float] = None
    multi_mmc: Optional[float] = None
    lz78y: Optional[float] = None
    min_entropy: float = 0.0
    # 非バイナリ用拡張推定器
    symbol_collision: Optional[float] = None
    symbol_markov: Optional[float] = None
    symbol_compression: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "mcv": self.mcv,
            "collision": self.collision,
            "markov": self.markov,
            "compression": self.compression,
            "t_tuple": self.t_tuple,
            "lrs": self.lrs,
            "multi_mcw": self.multi_mcw,
            "lag": self.lag,
            "multi_mmc": self.multi_mmc,
            "lz78y": self.lz78y,
            "symbol_collision": self.symbol_collision,
            "symbol_markov": self.symbol_markov,
            "symbol_compression": self.symbol_compression,
            "min_entropy": self.min_entropy,
        }


def assess_non_iid_entropy(data: np.ndarray, k: int) -> EntropyAssessmentResult:
    """
    規格書 Section 6.2: Non-IID トラックの全推定器を実行し最小値を求める。
    バイナリ（k=2）の場合は10種全てを実行。
    非バイナリ（k>2）の場合はバイナリ専用を除外した7種に加え、拡張シンボル推定器も計算。
    """
    is_binary = (k == 2)
    estimates = {}

    # 1. MCV
    h_mcv = estimate_mcv(data)
    estimates["mcv"] = h_mcv

    # 2. Collision (バイナリ専用)
    h_col = None
    h_sym_col = None
    if is_binary:
        h_col = estimate_collision(data)
        estimates["collision"] = h_col
    else:
        h_sym_col = estimate_symbol_collision(data, k=k)

    # 3. Markov (バイナリ専用)
    h_markov = None
    h_sym_markov = None
    if is_binary:
        h_markov = estimate_markov(data)
        estimates["markov"] = h_markov
    else:
        h_sym_markov = estimate_symbol_markov(data, k=k)

    # 4. Compression (バイナリ専用)
    h_comp = None
    h_sym_comp = None
    if is_binary:
        h_comp = estimate_compression(data)
        estimates["compression"] = h_comp
    else:
        h_sym_comp = estimate_symbol_compression(data, k=k)

    # 5. t-Tuple
    h_tuple = estimate_t_tuple(data)
    estimates["t_tuple"] = h_tuple

    # 6. LRS
    h_lrs = estimate_lrs(data)
    estimates["lrs"] = h_lrs

    # 7. MultiMCW
    h_mcw = estimate_multi_mcw(data, k=k)
    estimates["multi_mcw"] = h_mcw

    # 8. Lag
    h_lag = estimate_lag(data, k=k)
    estimates["lag"] = h_lag

    # 9. MultiMMC
    h_mmc = estimate_multi_mmc(data, k=k)
    estimates["multi_mmc"] = h_mmc

    # 10. LZ78Y
    h_lz = estimate_lz78y(data, k=k)
    estimates["lz78y"] = h_lz

    # 有効な規格値の最小値を取得 (SP 800-90B 準拠)
    valid_values = [v for v in estimates.values() if v is not None and not math.isnan(v)]
    min_h = min(valid_values) if valid_values else 0.0

    return EntropyAssessmentResult(
        is_binary=is_binary,
        k=k,
        mcv=h_mcv,
        collision=h_col,
        markov=h_markov,
        compression=h_comp,
        t_tuple=h_tuple,
        lrs=h_lrs,
        multi_mcw=h_mcw,
        lag=h_lag,
        multi_mmc=h_mmc,
        lz78y=h_lz,
        min_entropy=min_h,
        symbol_collision=h_sym_col,
        symbol_markov=h_sym_markov,
        symbol_compression=h_sym_comp,
    )
