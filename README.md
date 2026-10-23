# NIST SP 800-90B 乱数検定・エントロピー評価ツール

米国国立標準技術研究所 (NIST) の乱数源エントロピー評価規格 **SP 800-90B (*Recommendation for the Entropy Sources Used for Random Bit Generation*)** に準拠したPython実装です。

## 主な機能

- **Section 6 最小エントロピー推定 (Min-Entropy Estimation)**
  - 最頻値推定 (Most Common Value: MCV)
  - 衝突推定 (Collision Estimate, バイナリ専用)
  - マルコフ推定 (Markov Estimate, バイナリ専用)
  - 圧縮推定 (Compression Estimate, Maurer統計量, バイナリ専用)
  - $t$-タプル推定 ($t$-Tuple Estimate)
  - 最長重複部分列推定 (Longest Repeated Substring: LRS Estimate)
  - 4つの予測器推定 (MultiMCW, Lag, MultiMMC, LZ78Y)
  - ビット列展開評価 ($H_{\\text{bitstring}}$) および総合初期エントロピー $H_I$ の決定
- **Section 5 IID仮説検定 (Testing the IID Assumption)**
  - 10,000回の Fisher-Yates 並べ替え検定 (Permutation Tests)
  - カイ二乗検定 (独立性検定 & 10分割適合度検定)
  - LRS検定
- **Section 3.1.4 リスタートテスト (Restart Tests)**
  - $1000 \\times 1000$ リスタート行列に対する二項サニティチェック
  - 行・列データセットのエントロピー評価 ($H_r, H_c$)
- **Section 4 ヘルステスト (Health Tests)**
  - Repetition Count Test (RCT)
  - Adaptive Proportion Test (APT)
  - 許容誤検知率 $\\alpha = 2^{-20}$ に基づくカットオフ閾値 $C$ の自動計算
- **合成テストデータ生成機能**
  - 完全一様乱数、偏り（バイアス）データ、相関マルコフデータ、リスタート行列の自動生成

## インストールと環境構築

プロジェクト直下に仮想環境 `.venv` を作成して実行します。

```powershell
# 仮想環境の作成とパッケージインストール (uv または pip)
uv venv .venv
uv pip install numpy scipy bz2file pytest
```

## 使い方 (CLI)

### 1. 乱数データファイルのエントロピー推定 (HTML / JSON 出力)

```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe sp800_90b/cli.py estimate --file your_random_data.bin --bits 8 --html-out report.html --json-out result.json
```

### 2. 内蔵の合成データを用いた即時テスト

手元に実データがない場合でも、`--samples` を指定して即座に評価を試すことができます。

```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe sp800_90b/cli.py estimate --samples 20000 --bits 8 --html-out report.html --json-out result.json
```

### 3. ユニットテストの実行

規格書 Section 6 に掲載されている数値例 (Examples) と照合するテストが含まれています。

```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\pytest.exe tests
```
