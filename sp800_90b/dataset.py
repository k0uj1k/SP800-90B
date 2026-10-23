"""
NIST SP 800-90B データ構造および前処理モジュール
"""

from typing import List, Tuple
import numpy as np


class Dataset:
    """
    乱数データセットを保持し、シンボル列やビット列への変換・抽出を提供するクラス
    """

    def __init__(self, data: np.ndarray, bits_per_symbol: int = 8):
        """
        :param data: 1次元の整数配列（uint8等）
        :param bits_per_symbol: 1シンボルあたりの有効ビット長 (1 <= n <= 8)
        """
        self.raw_data = np.asarray(data, dtype=np.uint32)
        self.bits_per_symbol = bits_per_symbol
        self.length = len(self.raw_data)

        # ユニークシンボル数 k
        unique_vals = np.unique(self.raw_data)
        self.alphabet = unique_vals
        self.k = len(unique_vals)

    @classmethod
    def from_bytes(cls, raw_bytes: bytes, bits_per_symbol: int = 8) -> "Dataset":
        """バイト列からデータセットを構築する"""
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        return cls(arr, bits_per_symbol=bits_per_symbol)

    @classmethod
    def from_file(cls, path: str, bits_per_symbol: int = 8, max_samples: int = None) -> "Dataset":
        """バイナリファイルから読み込む"""
        with open(path, "rb") as f:
            data = f.read(max_samples if max_samples is not None else -1)
        return cls.from_bytes(data, bits_per_symbol=bits_per_symbol)

    def to_bitstring(self, max_bits: int = 1_000_000) -> np.ndarray:
        """
        規格書 Section 3.1.3:
        n-bitの各シンボルを最上位ビットから順にビット列（0 or 1）に展開する。
        先頭 max_bits ビット（通常1,000,000ビット）を抽出して返す。
        """
        bits = []
        for val in self.raw_data:
            for b in range(self.bits_per_symbol - 1, -1, -1):
                bits.append((val >> b) & 1)
                if len(bits) >= max_bits:
                    return np.array(bits, dtype=np.uint8)
        return np.array(bits, dtype=np.uint8)

    def to_conversion_i(self) -> np.ndarray:
        """
        規格書 Section 5.1 Conversion I:
        バイナリ入力を8ビットごとの非重複ブロックに分割し、各ブロック内の1の個数を数えた系列を出力する。
        """
        bits = self.raw_data
        remainder = len(bits) % 8
        if remainder != 0:
            padded_bits = np.pad(bits, (0, 8 - remainder), "constant", constant_values=0)
        else:
            padded_bits = bits
        blocks = padded_bits.reshape(-1, 8)
        return np.sum(blocks, axis=1, dtype=np.uint32)

    def to_conversion_ii(self) -> np.ndarray:
        """
        規格書 Section 5.1 Conversion II:
        バイナリ入力を8ビットごとの非重複ブロックに分割し、各ブロックの整数値（0〜255）を出力する。
        最上位ビットから順に解釈。
        """
        bits = self.raw_data
        remainder = len(bits) % 8
        if remainder != 0:
            padded_bits = np.pad(bits, (0, 8 - remainder), "constant", constant_values=0)
        else:
            padded_bits = bits
        blocks = padded_bits.reshape(-1, 8)
        # 8ビットを整数に集約
        powers = 2 ** np.arange(7, -1, -1, dtype=np.uint32)
        return np.dot(blocks, powers)
