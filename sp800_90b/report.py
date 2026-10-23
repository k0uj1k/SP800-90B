"""
NIST SP 800-90B グラフィカル＆インタラクティブ HTML レポート生成モジュール
Chart.js を埋め込み、シンボル度数分布、推定器別エントロピー比較棒グラフ、
エントロピーゲージメーター、ヘルステスト解説ビジュアルを直感的に表示する。
"""

import datetime
import json
from typing import Any, Dict


def generate_html_report(data: Dict[str, Any], title: str = "NIST SP 800-90B 乱数エントロピー評価レポート") -> str:
    """
    評価結果の辞書データから Chart.js によるグラフ付きのモダン HTML を構築する
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    samples = data.get("samples", 0)
    bits_per_symbol = data.get("bits_per_symbol", 8)
    alphabet_size = data.get("alphabet_size", 256)
    max_entropy_possible = float(bits_per_symbol)
    h_original = data.get("h_original", 0.0)
    h_bitstring_scaled = data.get("h_bitstring_scaled")
    h_initial = data.get("h_initial", 0.0)
    h_final = data.get("h_final", 0.0)

    entropy_rate = (h_final / max_entropy_possible) * 100.0 if max_entropy_possible > 0 else 0.0

    estimates = data.get("estimates", {})
    health = data.get("health_test_cutoffs", {})

    # 推定器の表示ラベルと値
    estimator_labels = [
        ("mcv", "6.3.1 最頻値 (MCV)"),
        ("collision", "6.3.2 衝突 (Collision)"),
        ("markov", "6.3.3 マルコフ (Markov)"),
        ("compression", "6.3.4 圧縮 (Compression)"),
        ("t_tuple", "6.3.5 t-タプル (t-Tuple)"),
        ("lrs", "6.3.6 最長重複 (LRS)"),
        ("multi_mcw", "6.3.7 MultiMCW 予測器"),
        ("lag", "6.3.8 Lag 予測器"),
        ("multi_mmc", "6.3.9 MultiMMC 予測器"),
        ("lz78y", "6.3.10 LZ78Y 予測器"),
    ]

    chart_labels = []
    chart_values = []
    chart_bg_colors = []
    chart_border_colors = []

    table_rows = []
    min_val = estimates.get("min_entropy", float("inf"))

    for key, display_name in estimator_labels:
        val = estimates.get(key)
        if val is not None:
            chart_labels.append(display_name.split(" ")[1])  # 短縮名
            chart_values.append(round(val, 4))
            is_min = abs(val - min_val) < 1e-5
            if is_min:
                chart_bg_colors.append("rgba(239, 68, 68, 0.75)")      # 赤ハイライト（最小ボトルネック）
                chart_border_colors.append("rgba(185, 28, 28, 1)")
            else:
                chart_bg_colors.append("rgba(59, 130, 246, 0.65)")     # 青
                chart_border_colors.append("rgba(29, 78, 216, 1)")

            badge = "<span class='badge badge-bottleneck'>最小ボトルネック</span>" if is_min else ""
            val_str = f"<strong>{val:.4f}</strong> bits/sample {badge}"
            row_class = "table-highlight" if is_min else ""
        else:
            val_str = "<span class='badge badge-secondary'>対象外 (非バイナリ)</span>"
            row_class = ""

        table_rows.append(f"""
            <tr class="{row_class}">
                <td>{display_name}</td>
                <td class="text-right">{val_str}</td>
            </tr>
        """)

    table_html = "\n".join(table_rows)

    # ヒストグラムデータ（存在する場合）
    hist_labels = data.get("histogram_labels", [])
    hist_counts = data.get("histogram_counts", [])
    has_hist = len(hist_counts) > 0

    # 評価ステータス判定バッジ
    if entropy_rate >= 80.0:
        status_badge = "<span class='status-pill status-good'>良好 (High Entropy)</span>"
    elif entropy_rate >= 50.0:
        status_badge = "<span class='status-pill status-warn'>中程度 (Medium)</span>"
    else:
        status_badge = "<span class='status-pill status-danger'>エントロピー低下 (Low)</span>"

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- Chart.js CDN (オフライン時はローカル表示にも耐えうるフォールバック付き) -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --bg: #f8fafc;
            --surface: #ffffff;
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --text-main: #0f172a;
            --text-sub: #64748b;
            --border: #e2e8f0;
            --danger: #ef4444;
            --warning: #f59e0b;
            --success: #10b981;
        }}
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 30px 16px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1080px;
            margin: 0 auto;
        }}
        /* ヘッダー */
        .header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 28px 32px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
            margin-bottom: 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .header h1 {{
            margin: 0 0 6px 0;
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.02em;
        }}
        .header .meta {{
            font-size: 13px;
            opacity: 0.85;
        }}
        /* ステータスピル */
        .status-pill {{
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}
        .status-good {{
            background-color: #dcfce7;
            color: #166534;
        }}
        .status-warn {{
            background-color: #fef3c7;
            color: #92400e;
        }}
        .status-danger {{
            background-color: #fee2e2;
            color: #991b1b;
        }}

        /* サマリカードグリッド */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 18px;
            margin-bottom: 28px;
        }}
        .kpi-card {{
            background: var(--surface);
            padding: 22px 24px;
            border-radius: 14px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
        }}
        .kpi-card::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--primary);
        }}
        .kpi-card.kpi-highlight::before {{
            background: var(--danger);
        }}
        .kpi-card .kpi-title {{
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-sub);
            margin-bottom: 8px;
        }}
        .kpi-card .kpi-value {{
            font-size: 32px;
            font-weight: 800;
            color: var(--text-main);
            letter-spacing: -0.02em;
        }}
        .kpi-card.kpi-highlight .kpi-value {{
            color: #dc2626;
        }}
        .kpi-card .kpi-sub {{
            font-size: 13px;
            color: var(--text-sub);
            margin-top: 6px;
        }}

        /* ゲージ進捗バー */
        .gauge-container {{
            margin-top: 10px;
            background-color: #e2e8f0;
            border-radius: 9999px;
            height: 8px;
            overflow: hidden;
        }}
        .gauge-bar {{
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #10b981);
            border-radius: 9999px;
        }}

        /* チャート＆セクション */
        .section-card {{
            background: var(--surface);
            padding: 26px 28px;
            border-radius: 14px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            border: 1px solid var(--border);
            margin-bottom: 28px;
        }}
        .section-title {{
            font-size: 19px;
            font-weight: 700;
            margin-top: 0;
            margin-bottom: 18px;
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .chart-box {{
            position: relative;
            height: 340px;
            width: 100%;
            margin-top: 10px;
        }}

        /* テーブル */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 13px 16px;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: #f8fafc;
            font-weight: 600;
            color: var(--text-sub);
            text-align: left;
        }}
        .text-right {{
            text-align: right;
        }}
        .table-highlight {{
            background-color: #fef2f2 !important;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
        }}
        .badge-bottleneck {{
            background-color: #fee2e2;
            color: #b91c1c;
        }}
        .badge-secondary {{
            background-color: #f1f5f9;
            color: #64748b;
        }}

        /* ヘルステストビジュアル */
        .health-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 14px;
        }}
        @media (max-width: 768px) {{
            .health-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .health-box {{
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }}
        .health-box h3 {{
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 16px;
            color: var(--primary-dark);
        }}
        .health-metric {{
            font-size: 28px;
            font-weight: 800;
            color: var(--text-main);
            margin: 10px 0;
        }}
        .health-desc {{
            font-size: 13px;
            color: var(--text-sub);
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- ヘッダー -->
        <div class="header">
            <div>
                <h1>NIST SP 800-90B 乱数エントロピー評価レポート</h1>
                <div class="meta">生成日時: {now_str} &bull; 規格: NIST SP 800-90B (2018)</div>
            </div>
            <div>
                {status_badge}
            </div>
        </div>

        <!-- KPI グリッド -->
        <div class="kpi-grid">
            <div class="kpi-card kpi-highlight">
                <div class="kpi-title">確定最小エントロピー (H)</div>
                <div class="kpi-value">{h_final:.4f}</div>
                <div class="kpi-sub">bits / sample (最悪ケース予測耐性)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">エントロピー率 (Entropy Rate)</div>
                <div class="kpi-value">{entropy_rate:.1f}%</div>
                <div class="gauge-container">
                    <div class="gauge-bar" style="width: {min(100.0, entropy_rate):.1f}%;"></div>
                </div>
                <div class="kpi-sub">{max_entropy_possible:.1f} bits 中 {h_final:.2f} bits</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">評価サンプル数 (L)</div>
                <div class="kpi-value">{samples:,}</div>
                <div class="kpi-sub">1シンボル: {bits_per_symbol} bits</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">ユニークシンボル数 (k)</div>
                <div class="kpi-value">{alphabet_size}</div>
                <div class="kpi-sub">最大可能シンボル: {1 << bits_per_symbol}</div>
            </div>
        </div>

        <!-- グラフ1: 各推定器のエントロピー比較棒グラフ -->
        <div class="section-card">
            <div class="section-title">
                <span>📈 推定器別エントロピー比較チャート (Section 6)</span>
                <span style="font-size: 12px; font-weight: normal; color: var(--text-sub);">
                    赤色バーが安全性の下限（最小エントロピー）
                </span>
            </div>
            <div class="chart-box">
                <canvas id="estimatorsChart"></canvas>
            </div>
        </div>

        <!-- グラフ2: 出現頻度ヒストグラム (データがある場合) -->
        {"<!-- Histogram Section -->" if not has_hist else f'''
        <div class="section-card">
            <div class="section-title">
                <span>📊 シンボル出現頻度分布 (Histogram)</span>
                <span style="font-size: 12px; font-weight: normal; color: var(--text-sub);">
                    理想的な乱数源ではフラットな一様分布になります
                </span>
            </div>
            <div class="chart-box">
                <canvas id="histChart"></canvas>
            </div>
        </div>
        '''}

        <!-- 推定器詳細テーブル -->
        <div class="section-card">
            <div class="section-title">
                <span>📋 推定器の内訳と個別スコア詳細</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>推定アルゴリズム</th>
                        <th class="text-right">推定エントロピー値</th>
                    </tr>
                </thead>
                <tbody>
                    {table_html}
                </tbody>
            </table>
        </div>

        <!-- ヘルステストビジュアル解説 -->
        <div class="section-card">
            <div class="section-title">
                <span>🛡️ ヘルステスト推奨閾値 (Health Test Cutoffs, α = 2⁻²⁰)</span>
            </div>
            <div class="health-grid">
                <div class="health-box">
                    <h3>4.4.1 Repetition Count Test (RCT)</h3>
                    <div class="health-metric">C = {health.get('rct_cutoff', '-')} <span style="font-size: 14px; font-weight: normal; color: var(--text-sub);">回</span></div>
                    <div class="health-desc">
                        ノイズ源が特定の値に固着（スタック）する致命的故障を即座に検出します。<br>
                        同一シンボルが連続して <strong>{health.get('rct_cutoff', '-')} 回</strong> 現れた瞬間にエラーを出力します。
                    </div>
                </div>
                <div class="health-box">
                    <h3>4.4.2 Adaptive Proportion Test (APT)</h3>
                    <div class="health-metric">C = {health.get('apt_cutoff', '-')} / W = {health.get('apt_window', '-')}</div>
                    <div class="health-desc">
                        環境変化や劣化によるエントロピー低下・偏りを検出します。<br>
                        <strong>{health.get('apt_window', '-')} サンプル</strong> の監視窓内で同一値が <strong>{health.get('apt_cutoff', '-')} 回以上</strong> 出現した場合にエラーを出力します。
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Chart.js スクリプト -->
    <script>
        const estCtx = document.getElementById('estimatorsChart');
        if (estCtx && window.Chart) {{
            new Chart(estCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(chart_labels, ensure_ascii=False)},
                    datasets: [{{
                        label: '推定エントロピー (bits/sample)',
                        data: {json.dumps(chart_values)},
                        backgroundColor: {json.dumps(chart_bg_colors)},
                        borderColor: {json.dumps(chart_border_colors)},
                        borderWidth: 1.5,
                        borderRadius: 6,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    return context.parsed.y.toFixed(4) + ' bits/sample';
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: {max_entropy_possible},
                            title: {{
                                display: true,
                                text: 'Min-Entropy (bits/sample)'
                            }},
                            grid: {{
                                color: '#f1f5f9'
                            }}
                        }},
                        x: {{
                            grid: {{
                                display: false
                            }}
                        }}
                    }}
                }}
            }});
        }}

        {f'''
        const histCtx = document.getElementById('histChart');
        if (histCtx && window.Chart) {{
            new Chart(histCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(hist_labels)},
                    datasets: [{{
                        label: '出現回数',
                        data: {json.dumps(hist_counts)},
                        backgroundColor: 'rgba(99, 102, 241, 0.15)',
                        borderColor: 'rgba(99, 102, 241, 1)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: true,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{ display: true, text: 'カウント数' }}
                        }},
                        x: {{
                            title: {{ display: true, text: 'シンボル値' }}
                        }}
                    }}
                }}
            }});
        }}
        ''' if has_hist else ""}
    </script>
</body>
</html>
"""
    return html
