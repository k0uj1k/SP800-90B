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
from sp800_90b.estimators.extended import compute_extended_entropies
from sp800_90b.estimators.timeseries import analyze_sliding_window_entropy
from sp800_90b.health import calc_rct_cutoff, calc_apt_cutoff, HealthTester
from sp800_90b.validation import run_restart_tests, compute_output_entropy
from sp800_90b.generator import generate_uniform_data, generate_biased_data, generate_markov_data, generate_restart_matrix
from sp800_90b.report import generate_html_report
from sp800_90b.iid.permutation import run_permutation_tests
from sp800_90b.iid.chi_square import (
    test_independence_non_binary,
    test_goodness_of_fit_non_binary,
    test_independence_binary,
    test_goodness_of_fit_binary,
    test_lrs_iid,
)


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

    # IID仮説検定 (Option または Auto)
    iid_testing_result = None
    iid_verified = False
    if args.track in ["iid", "auto"]:
        print("\n[*] Section 5 IID仮説検定 (Permutation & Chi-Square) を実行中...")
        is_binary = (dataset.k == 2)
        # カイ二乗検定
        if is_binary:
            indep_res = test_independence_binary(dataset.raw_data)
            gof_res = test_goodness_of_fit_binary(dataset.raw_data)
        else:
            indep_res = test_independence_non_binary(dataset.raw_data, k=dataset.k)
            gof_res = test_goodness_of_fit_non_binary(dataset.raw_data, k=dataset.k)

        lrs_pass = test_lrs_iid(dataset.raw_data)
        chi_passed = indep_res.passed and gof_res.passed and lrs_pass

        # 並べ替え検定 (10,000回シャッフル)
        num_perms = 1000 if args.fast else 10000
        perm_res = run_permutation_tests(dataset.raw_data, is_binary=is_binary, num_permutations=num_perms)
        iid_verified = chi_passed and perm_res.passed

        iid_testing_result = {
            "verified": iid_verified,
            "permutation_passed": perm_res.passed,
            "failed_permutation_tests": perm_res.failed_test_indices,
            "independence_passed": indep_res.passed,
            "goodness_of_fit_passed": gof_res.passed,
            "lrs_passed": lrs_pass,
        }
        print(f"  - IID仮説判定結果: {'合格 (IID承認)' if iid_verified else '不合格 (Non-IIDへフォールバック)'}")

    # Section 6 エントロピー推定器
    print("\n[*] Section 6 エントロピー推定器を実行中...")
    result_orig = assess_non_iid_entropy(dataset.raw_data, k=dataset.k)

    h_original = result_orig.mcv if (args.track == "iid" or (args.track == "auto" and iid_verified)) else result_orig.min_entropy
    print(f"\n--- [SP 800-90B エントロピー推定結果 (H_original)] ---")
    for k_name, v in result_orig.to_dict().items():
        if v is not None:
            print(f"  - {k_name:<18}: {v:.4f} bits/sample")

    # 案1 & 案2: 拡張情報理論エントロピーの計算
    print("\n[*] 案1/案2 拡張エントロピー指標 (Shannon / Renyi / Collision) を計算中...")
    extended_metrics = compute_extended_entropies(dataset.raw_data, bits_per_symbol=dataset.bits_per_symbol)
    print(f"  - シャノンエントロピー (H_1) : {extended_metrics['shannon']:.4f} bits/sample")
    print(f"  - 衝突エントロピー (H_2)     : {extended_metrics['collision_h2']:.4f} bits/sample")
    print(f"  - レニーエントロピー (H_0.5) : {extended_metrics['renyi_alpha_0_5']:.4f} bits/sample")
    print(f"  - レニーエントロピー (H_3.0) : {extended_metrics['renyi_alpha_3_0']:.4f} bits/sample")
    print(f"  - エントロピー効率 (Ratio)   : {extended_metrics['efficiency']:.2%}")

    # 案4: 時系列・スライディングウィンドウ分析
    print("\n[*] 案4 スライディングウィンドウによるエントロピー推移を分析中...")
    timeseries_data = analyze_sliding_window_entropy(dataset.raw_data, num_windows=min(30, max(5, dataset.length // 1000)))

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
        "track_used": "IID" if (args.track == "iid" or (args.track == "auto" and iid_verified)) else "Non-IID",
        "iid_test_result": iid_testing_result,
        "h_original": h_original,
        "h_bitstring_scaled": h_bitstring_scaled,
        "h_initial": h_i,
        "h_final": h_final,
        "histogram_labels": hist_labels,
        "histogram_counts": hist_counts,
        "extended_metrics": extended_metrics,
        "timeseries_data": timeseries_data,
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
    p_est.add_argument("--track", type=str, choices=["non-iid", "iid", "auto"], default="non-iid", help="評価トラック (non-iid, iid, auto)")
    p_est.add_argument("--fast", action="store_true", help="高速検証モード (並べ替え検定の回数を短縮)")
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
