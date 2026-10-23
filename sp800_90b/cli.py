"""
NIST SP 800-90B CLI エントリポイント
"""

import argparse
import json
import os
import sys
import numpy as np

from sp800_90b.dataset import Dataset
from sp800_90b.estimators import assess_non_iid_entropy
from sp800_90b.health import calc_rct_cutoff, calc_apt_cutoff, HealthTester
from sp800_90b.validation import run_restart_tests, compute_output_entropy
from sp800_90b.generator import generate_uniform_data, generate_biased_data, generate_markov_data, generate_restart_matrix
from sp800_90b.report import generate_html_report


def run_assessment(args):
    # データセットのロードまたは自動生成
    if args.file:
        print(f"[*] ファイルを読み込み中: {args.file}")
        dataset = Dataset.from_file(args.file, bits_per_symbol=args.bits)
    else:
        print("[*] 入力ファイルが指定されていないため、検証用の一様乱数データを生成します (100,000 サンプル)")
        sample_data = generate_uniform_data(length=args.samples, bits_per_symbol=args.bits)
        dataset = Dataset(sample_data, bits_per_symbol=args.bits)

    print(f"[*] サンプル長 L: {dataset.length:,}")
    print(f"[*] 1シンボルあたりのビット長 n: {dataset.bits_per_symbol}")
    print(f"[*] アルファベットサイズ k: {dataset.k}")

    # Non-IID トラックによるエントロピー評価
    print("\n[*] Section 6 エントロピー推定器を実行中...")
    result_orig = assess_non_iid_entropy(dataset.raw_data, k=dataset.k)

    h_original = result_orig.min_entropy
    print("\n--- [エントロピー推定結果 (H_original)] ---")
    for k, v in result_orig.to_dict().items():
        if v is not None:
            print(f"  - {k:<15}: {v:.4f} bits/sample")

    # 非バイナリの場合はビット列に対する評価も実施 (Section 3.1.3)
    h_bitstring_scaled = None
    if not result_orig.is_binary:
        print("\n[*] Section 3.1.3 ビット列に対する評価 (H_bitstring) を実行中...")
        bitstring = dataset.to_bitstring(max_bits=1_000_000)
        res_bit = assess_non_iid_entropy(bitstring, k=2)
        h_bitstring = res_bit.min_entropy
        h_bitstring_scaled = dataset.bits_per_symbol * h_bitstring
        print(f"  - H_bitstring (per bit)   : {h_bitstring:.4f} bits/bit")
        print(f"  - Scaled (n * H_bitstring): {h_bitstring_scaled:.4f} bits/sample")

    # 初期エントロピー H_I の決定
    h_sub = args.h_submitter if args.h_submitter is not None else float(dataset.bits_per_symbol)
    if h_bitstring_scaled is not None:
        h_i = min(h_original, h_bitstring_scaled, h_sub)
    else:
        h_i = min(h_original, h_sub)

    print(f"\n[+] 確定したノイズ源初期エントロピー H_I: {h_i:.4f} bits/sample (H_sub={h_sub:.4f})")

    # リスタートテスト (オプション)
    h_final = h_i
    if args.restart_file:
        print(f"\n[*] リスタートデータ行列を読み込み中: {args.restart_file}")
        # バイナリまたは npy から 1000x1000 行列を想定
        r_matrix = np.fromfile(args.restart_file, dtype=np.uint8).reshape(1000, 1000)
        r_res = run_restart_tests(r_matrix, h_i=h_i, k=dataset.k)
        print(f"  - リスタート Sanity Check : {'合格 (PASS)' if r_res.sanity_passed else '不合格 (FAIL)'} (p={r_res.p_value:.6f})")
        print(f"  - 行エントロピー H_r      : {r_res.h_r:.4f}")
        print(f"  - 列エントロピー H_c      : {r_res.h_c:.4f}")
        print(f"  - リスタート総合判定      : {'合格 (PASS)' if r_res.passed else '不合格 (FAIL)'}")
        if r_res.passed:
            h_final = r_res.h_final
    elif args.run_synthetic_restart:
        print("\n[*] 検証用合成リスタート行列 (1000x1000) を生成してテスト中...")
        r_matrix = generate_restart_matrix(bits_per_symbol=args.bits)
        r_res = run_restart_tests(r_matrix, h_i=h_i, k=dataset.k)
        print(f"  - リスタート Sanity Check : {'合格 (PASS)' if r_res.sanity_passed else '不合格 (FAIL)'} (p={r_res.p_value:.6f})")
        print(f"  - リスタート総合判定      : {'合格 (PASS)' if r_res.passed else '不合格 (FAIL)'}")
        if r_res.passed:
            h_final = r_res.h_final

    # ヘルステスト推奨閾値の算出 (Section 4)
    rct_c = calc_rct_cutoff(h_final)
    apt_w, apt_c = calc_apt_cutoff(h_final, is_binary=result_orig.is_binary)

    print("\n--- [推奨ヘルステスト閾値 (Section 4.4, alpha = 2^-20)] ---")
    print(f"  - Repetition Count Test (RCT) カットオフ C : {rct_c}")
    print(f"  - Adaptive Proportion Test (APT) 窓長 W    : {apt_w}")
    print(f"  - Adaptive Proportion Test (APT) カットオフ C: {apt_c}")

    # ヒストグラム用の出現回数集計 (非バイナリかつ256以下の場合)
    hist_labels = []
    hist_counts = []
    if dataset.k <= 256:
        unique_vals, counts = np.unique(dataset.raw_data, return_counts=True)
        count_map = {int(v): int(c) for v, c in zip(unique_vals, counts)}
        max_possible = 1 << dataset.bits_per_symbol
        for val in range(min(max_possible, 256)):
            hist_labels.append(str(val))
            hist_counts.append(count_map.get(val, 0))

    # 結果データ構造
    summary_data = {
        "samples": dataset.length,
        "bits_per_symbol": dataset.bits_per_symbol,
        "alphabet_size": dataset.k,
        "h_original": h_original,
        "h_bitstring_scaled": h_bitstring_scaled,
        "h_initial": h_i,
        "h_final": h_final,
        "histogram_labels": hist_labels,
        "histogram_counts": hist_counts,
        "health_test_cutoffs": {
            "rct_cutoff": rct_c,
            "apt_window": apt_w,
            "apt_cutoff": apt_c,
        },
        "estimates": result_orig.to_dict(),
    }

    # JSON出力
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"\n[+] 結果を JSON 形式で保存しました: {args.json_out}")

    # HTML出力
    if args.html_out:
        html_content = generate_html_report(summary_data)
        with open(args.html_out, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[+] 見やすい HTML レポートを保存しました: {args.html_out}")


def main():
    parser = argparse.ArgumentParser(description="NIST SP 800-90B 乱数エントロピー検定ツール")
    subparsers = parser.add_subparsers(dest="command")

    # estimate コマンド
    p_est = subparsers.add_parser("estimate", help="乱数データのエントロピー推定と検定を実行")
    p_est.add_argument("--file", "-f", type=str, help="入力データファイルパス (未指定時は合成データを使用)")
    p_est.add_argument("--bits", "-b", type=int, default=8, help="1シンボルあたりのビット長 (1〜8, デフォルト: 8)")
    p_est.add_argument("--samples", "-s", type=int, default=100_000, help="合成データ時のサンプル数 (デフォルト: 100,000)")
    p_est.add_argument("--h-submitter", type=float, help="申告最小エントロピー H_submitter")
    p_est.add_argument("--restart-file", type=str, help="1000x1000 リスタートデータファイル")
    p_est.add_argument("--run-synthetic-restart", action="store_true", help="合成データでリスタートテストを試行")
    p_est.add_argument("--json-out", "-o", type=str, help="評価結果の JSON 出力パス")
    p_est.add_argument("--html-out", type=str, help="評価結果の HTML レポート出力パス")

    args = parser.parse_args()
    if args.command == "estimate" or args.command is None:
        if args.command is None:
            # 引数なしの場合はヘルプまたはデフォルト動作
            run_assessment(argparse.Namespace(
                file=None, bits=8, samples=10_000, h_submitter=None,
                restart_file=None, run_synthetic_restart=False, json_out=None
            ))
        else:
            run_assessment(args)


if __name__ == "__main__":
    main()
