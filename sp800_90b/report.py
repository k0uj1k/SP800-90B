"""
NIST SP 800-90B ビビッド＆クール (Cyberpunk/Dark Neon) HTML ダッシュボード
- 日本語 / English のワンクリック表示切り替え (i18n)
- ネオンサイバーパンク/ダークモードのビビッドなビジュアル
- Chart.js による動的チャート (Shannon/Rényi比較, SP 800-90B推定器比較, スライディングウィンドウ推移, ヒストグラム)
"""

import datetime
import json
from typing import Any, Dict


def generate_html_report(data: Dict[str, Any], title: str = "NIST SP 800-90B Entropy Assessment Dashboard") -> str:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    samples = data.get("samples", 0)
    bits_per_symbol = data.get("bits_per_symbol", 8)
    alphabet_size = data.get("alphabet_size", 256)
    max_entropy_possible = float(bits_per_symbol)
    h_final = data.get("h_final", 0.0)
    track_used = data.get("track_used", "Non-IID")

    entropy_rate = (h_final / max_entropy_possible) * 100.0 if max_entropy_possible > 0 else 0.0

    estimates = data.get("estimates", {})
    health = data.get("health_test_cutoffs", {})
    extended = data.get("extended_metrics", {})
    timeseries = data.get("timeseries_data", {})
    iid_test = data.get("iid_test_result")

    # 推定器定義
    estimator_keys = [
        ("mcv", "6.3.1 最頻値 (MCV)", "6.3.1 Most Common Value (MCV)"),
        ("collision", "6.3.2 衝突 (Collision)", "6.3.2 Collision Estimate"),
        ("markov", "6.3.3 マルコフ (Markov)", "6.3.3 Markov Estimate"),
        ("compression", "6.3.4 圧縮 (Compression)", "6.3.4 Compression Estimate"),
        ("t_tuple", "6.3.5 t-タプル (t-Tuple)", "6.3.5 t-Tuple Estimate"),
        ("lrs", "6.3.6 最長重複 (LRS)", "6.3.6 LRS Estimate"),
        ("multi_mcw", "6.3.7 MultiMCW 予測器", "6.3.7 MultiMCW Predictor"),
        ("lag", "6.3.8 Lag 予測器", "6.3.8 Lag Predictor"),
        ("multi_mmc", "6.3.9 MultiMMC 予測器", "6.3.9 MultiMMC Predictor"),
        ("lz78y", "6.3.10 LZ78Y 予測器", "6.3.10 LZ78Y Predictor"),
        ("symbol_collision", "拡張シンボル衝突 (Symbol Collision)", "Extended Symbol Collision"),
        ("symbol_markov", "拡張シンボルマルコフ (Symbol Markov)", "Extended Symbol Markov"),
        ("symbol_compression", "拡張シンボル圧縮 (Symbol Compression)", "Extended Symbol Compression"),
    ]

    sp_chart_labels_ja = []
    sp_chart_labels_en = []
    sp_chart_values = []
    sp_chart_bg = []
    sp_chart_border = []
    table_rows = []

    min_val = estimates.get("min_entropy", float("inf"))

    for key, name_ja, name_en in estimator_keys:
        val = estimates.get(key)
        if val is not None:
            sp_chart_labels_ja.append(name_ja.split(" ")[1] if " " in name_ja else name_ja)
            sp_chart_labels_en.append(name_en.split(" ")[1] if " " in name_en else name_en)
            sp_chart_values.append(round(val, 4))
            is_min = abs(val - min_val) < 1e-5 and "拡張" not in name_ja
            if is_min:
                sp_chart_bg.append("rgba(255, 0, 85, 0.85)")       # ビビッドネオンピンク/レッド
                sp_chart_border.append("#ff0055")
            elif "拡張" in name_ja:
                sp_chart_bg.append("rgba(168, 85, 247, 0.8)")      # ネオンパープル
                sp_chart_border.append("#c084fc")
            else:
                sp_chart_bg.append("rgba(0, 240, 255, 0.75)")      # ネオンシアン
                sp_chart_border.append("#00f0ff")

            badge_ja = "<span class='badge badge-danger'>最小ボトルネック</span>" if is_min else ""
            badge_en = "<span class='badge badge-danger'>Bottleneck Min</span>" if is_min else ""
            if "拡張" in name_ja:
                badge_ja = "<span class='badge badge-purple'>拡張指標</span>"
                badge_en = "<span class='badge badge-purple'>Extended</span>"

            val_str = f"<strong>{val:.4f}</strong> <span class='unit'>bits/sample</span> {badge_ja}"
            val_str_en = f"<strong>{val:.4f}</strong> <span class='unit'>bits/sample</span> {badge_en}"
            row_class = "table-highlight" if is_min else ""
        else:
            val_str = "<span class='badge badge-secondary'>対象外 (非バイナリ)</span>"
            val_str_en = "<span class='badge badge-secondary'>N/A (Non-binary)</span>"
            row_class = ""

        table_rows.append(f"""
            <tr class="{row_class}">
                <td><span class="lang-ja">{name_ja}</span><span class="lang-en">{name_en}</span></td>
                <td class="text-right"><span class="lang-ja">{val_str}</span><span class="lang-en">{val_str_en}</span></td>
            </tr>
        """)

    table_html = "\n".join(table_rows)

    # 情報理論エントロピー指標
    info_labels_ja = ["シャノン (H₁)", "レニー α=0.5", "衝突/レニー α=2", "レニー α=3", "理論最頻値 (H_inf)", "SP 800-90B (H)"]
    info_labels_en = ["Shannon (H₁)", "Rényi α=0.5", "Collision α=2", "Rényi α=3", "Max-Prob (H_inf)", "SP 800-90B (H)"]
    info_values = [
        round(extended.get("shannon", 0.0), 4),
        round(extended.get("renyi_alpha_0_5", 0.0), 4),
        round(extended.get("collision_h2", 0.0), 4),
        round(extended.get("renyi_alpha_3_0", 0.0), 4),
        round(extended.get("min_entropy_raw", 0.0), 4),
        round(h_final, 4),
    ]

    hist_labels = data.get("histogram_labels", [])
    hist_counts = data.get("histogram_counts", [])
    has_hist = len(hist_counts) > 0

    ts_labels = timeseries.get("labels", [])
    ts_shannon = timeseries.get("shannon", [])
    ts_collision = timeseries.get("collision", [])
    ts_mcv = timeseries.get("mcv", [])
    has_ts = len(ts_labels) > 0

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #090d16;
            --surface: #111827;
            --surface-card: #182235;
            --border: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(0, 240, 255, 0.3);
            --neon-cyan: #00f0ff;
            --neon-pink: #ff0055;
            --neon-purple: #a855f7;
            --neon-green: #00ff88;
            --neon-yellow: #facc15;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --font-mono: 'JetBrains Mono', monospace;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 30px 16px;
            line-height: 1.6;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(0, 240, 255, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(255, 0, 85, 0.05) 0%, transparent 40%);
            min-height: 100vh;
        }}
        .container {{ max-width: 1140px; margin: 0 auto; }}

        /* 多言語切り替え */
        body.lang-mode-ja .lang-en {{ display: none !important; }}
        body.lang-mode-en .lang-ja {{ display: none !important; }}

        .lang-switch-btn {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--neon-cyan);
            color: var(--neon-cyan);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 700;
            font-family: var(--font-mono);
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);
        }}
        .lang-switch-btn:hover {{
            background: var(--neon-cyan);
            color: #000;
            box-shadow: 0 0 25px rgba(0, 240, 255, 0.5);
        }}

        /* ヘッダー */
        .header {{
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(24, 34, 53, 0.9) 100%);
            border: 1px solid var(--border);
            border-left: 5px solid var(--neon-cyan);
            padding: 26px 32px;
            border-radius: 16px;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            backdrop-filter: blur(10px);
        }}
        .header h1 {{
            margin: 0 0 6px 0;
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #fff 0%, var(--neon-cyan) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header .meta {{
            font-size: 13px;
            color: var(--text-sub);
            font-family: var(--font-mono);
        }}

        /* KPI グリッド */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--surface);
            padding: 22px 24px;
            border-radius: 14px;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.2);
        }}
        .kpi-card.cyan {{ border-top: 4px solid var(--neon-cyan); }}
        .kpi-card.pink {{ border-top: 4px solid var(--neon-pink); }}
        .kpi-card.purple {{ border-top: 4px solid var(--neon-purple); }}
        .kpi-card.green {{ border-top: 4px solid var(--neon-green); }}

        .kpi-title {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-sub);
            margin-bottom: 6px;
            font-family: var(--font-mono);
        }}
        .kpi-value {{
            font-size: 34px;
            font-weight: 800;
            color: #fff;
            font-family: var(--font-mono);
            letter-spacing: -0.03em;
        }}
        .kpi-card.pink .kpi-value {{
            color: var(--neon-pink);
            text-shadow: 0 0 20px rgba(255, 0, 85, 0.4);
        }}
        .kpi-card.cyan .kpi-value {{
            color: var(--neon-cyan);
            text-shadow: 0 0 20px rgba(0, 240, 255, 0.4);
        }}
        .kpi-card.purple .kpi-value {{
            color: var(--neon-purple);
            text-shadow: 0 0 20px rgba(168, 85, 247, 0.4);
        }}
        .kpi-sub {{
            font-size: 12px;
            color: var(--text-sub);
            margin-top: 6px;
        }}

        /* ゲージバー */
        .gauge-container {{
            margin-top: 10px;
            background-color: rgba(255, 255, 255, 0.08);
            border-radius: 9999px;
            height: 8px;
            overflow: hidden;
        }}
        .gauge-bar {{
            height: 100%;
            background: linear-gradient(90deg, var(--neon-cyan), var(--neon-green));
            border-radius: 9999px;
            box-shadow: 0 0 10px var(--neon-green);
        }}

        /* セクションカード */
        .section-card {{
            background: var(--surface);
            padding: 24px 26px;
            border-radius: 14px;
            border: 1px solid var(--border);
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 700;
            margin-top: 0;
            margin-bottom: 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #fff;
        }}
        .chart-box {{ position: relative; height: 330px; width: 100%; }}

        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        @media (max-width: 800px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

        .subcard {{
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            transition: all 0.2s ease;
        }}
        .subcard:hover {{ border-color: var(--neon-cyan); }}
        .subcard-title {{ font-size: 13px; font-weight: 700; color: var(--text-sub); font-family: var(--font-mono); }}
        .subcard-val {{ font-size: 26px; font-weight: 800; margin: 8px 0; color: #fff; font-family: var(--font-mono); }}
        .subcard-desc {{ font-size: 12px; color: var(--text-sub); }}

        /* テーブル */
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--border); }}
        th {{ background: rgba(255, 255, 255, 0.02); font-weight: 700; color: var(--text-sub); font-family: var(--font-mono); text-align: left; }}
        .text-right {{ text-align: right; }}
        .unit {{ font-size: 11px; color: var(--text-sub); font-family: var(--font-mono); }}
        .table-highlight {{
            background-color: rgba(255, 0, 85, 0.08) !important;
            border-left: 3px solid var(--neon-pink);
        }}

        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            font-family: var(--font-mono);
        }}
        .badge-danger {{
            background-color: rgba(255, 0, 85, 0.2);
            color: #ff3377;
            border: 1px solid var(--neon-pink);
        }}
        .badge-purple {{
            background-color: rgba(168, 85, 247, 0.2);
            color: #c084fc;
            border: 1px solid var(--neon-purple);
        }}
        .badge-secondary {{
            background-color: rgba(255, 255, 255, 0.05);
            color: var(--text-sub);
        }}
    </style>
</head>
<body class="lang-mode-ja">
    <div class="container">
        <!-- ヘッダー -->
        <div class="header">
            <div>
                <h1>
                    <span class="lang-ja">NIST SP 800-90B 乱数エントロピー評価ダッシュボード</span>
                    <span class="lang-en">NIST SP 800-90B Entropy Assessment Dashboard</span>
                </h1>
                <div class="meta">
                    <span class="lang-ja">生成日時: {now_str} &bull; モード: <strong>{track_used} Track</strong> &bull; サンプル数: {samples:,}</span>
                    <span class="lang-en">Generated: {now_str} &bull; Mode: <strong>{track_used} Track</strong> &bull; Samples: {samples:,}</span>
                </div>
            </div>
            <button class="lang-switch-btn" onclick="toggleLanguage()">
                <span class="lang-ja">🌐 English</span>
                <span class="lang-en">🌐 日本語</span>
            </button>
        </div>

        <!-- KPI グリッド -->
        <div class="kpi-grid">
            <div class="kpi-card pink">
                <div class="kpi-title">
                    <span class="lang-ja">SP 800-90B 確定最小エントロピー (H)</span>
                    <span class="lang-en">Final Assessed Min-Entropy (H)</span>
                </div>
                <div class="kpi-value">{h_final:.4f}</div>
                <div class="kpi-sub">
                    <span class="lang-ja">最悪ケースの攻撃耐性 (bits/sample)</span>
                    <span class="lang-en">Worst-case Guessing Work (bits/sample)</span>
                </div>
            </div>

            <div class="kpi-card purple">
                <div class="kpi-title">
                    <span class="lang-ja">シャノンエントロピー (Shannon H₁)</span>
                    <span class="lang-en">Shannon Entropy (H₁)</span>
                </div>
                <div class="kpi-value">{extended.get('shannon', 0.0):.4f}</div>
                <div class="kpi-sub">
                    <span class="lang-ja">平均情報量 (bits/sample)</span>
                    <span class="lang-en">Average Information Rate (bits/sample)</span>
                </div>
            </div>

            <div class="kpi-card cyan">
                <div class="kpi-title">
                    <span class="lang-ja">衝突エントロピー (Collision H₂)</span>
                    <span class="lang-en">Collision Entropy (H₂)</span>
                </div>
                <div class="kpi-value">{extended.get('collision_h2', 0.0):.4f}</div>
                <div class="kpi-sub">
                    <span class="lang-ja">Rényi α=2 エントロピー</span>
                    <span class="lang-en">Rényi Entropy of Order 2</span>
                </div>
            </div>

            <div class="kpi-card green">
                <div class="kpi-title">
                    <span class="lang-ja">エントロピー効率 (Efficiency)</span>
                    <span class="lang-en">Entropy Efficiency (Rate)</span>
                </div>
                <div class="kpi-value">{entropy_rate:.1f}%</div>
                <div class="gauge-container">
                    <div class="gauge-bar" style="width: {min(100.0, entropy_rate):.1f}%;"></div>
                </div>
                <div class="kpi-sub">{max_entropy_possible:.1f} bits 中 {h_final:.2f} bits</div>
            </div>
        </div>

        <!-- チャートグリッド -->
        <div class="grid-2">
            <!-- グラフ1: エントロピー指標比較 -->
            <div class="section-card">
                <div class="section-title">
                    <span>
                        <span class="lang-ja">📐 情報理論エントロピー比較 (Shannon vs Rényi vs Min)</span>
                        <span class="lang-en">📐 Information Theoretic Entropies (Shannon vs Rényi)</span>
                    </span>
                </div>
                <div class="chart-box">
                    <canvas id="infoChart"></canvas>
                </div>
            </div>

            <!-- グラフ2: SP 800-90B 推定器別比較 -->
            <div class="section-card">
                <div class="section-title">
                    <span>
                        <span class="lang-ja">📊 規格推定器・拡張推定器の比較</span>
                        <span class="lang-en">📊 SP 800-90B & Extended Estimators</span>
                    </span>
                </div>
                <div class="chart-box">
                    <canvas id="spChart"></canvas>
                </div>
            </div>
        </div>

        <!-- グラフ3: 時系列スライディングウィンドウ -->
        {"<!-- TimeSeries Section -->" if not has_ts else f'''
        <div class="section-card">
            <div class="section-title">
                <span>
                    <span class="lang-ja">⏱️ スライディングウィンドウによるエントロピー経時推移</span>
                    <span class="lang-en">⏱️ Sliding Window Entropy Dynamics</span>
                </span>
                <span style="font-size: 12px; font-weight: normal; color: var(--text-sub); font-family: var(--font-mono);">
                    <span class="lang-ja">時間経過に伴う局所的劣化・揺らぎを検出</span>
                    <span class="lang-en">Detecting localized degradation and drift over time</span>
                </span>
            </div>
            <div class="chart-box">
                <canvas id="tsChart"></canvas>
            </div>
        </div>
        '''}

        <!-- グラフ4: ヒストグラム -->
        {"<!-- Histogram Section -->" if not has_hist else f'''
        <div class="section-card">
            <div class="section-title">
                <span>
                    <span class="lang-ja">📈 シンボル出現頻度分布 (Histogram)</span>
                    <span class="lang-en">📈 Symbol Frequency Distribution</span>
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
                <span>
                    <span class="lang-ja">📋 全推定アルゴリズムの個別スコア内訳</span>
                    <span class="lang-en">📋 Individual Estimator Score Breakdown</span>
                </span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>
                            <span class="lang-ja">推定アルゴリズム</span>
                            <span class="lang-en">Algorithm</span>
                        </th>
                        <th class="text-right">
                            <span class="lang-ja">推定エントロピー値</span>
                            <span class="lang-en">Assessed Value</span>
                        </th>
                    </tr>
                </thead>
                <tbody>
                    {table_html}
                </tbody>
            </table>
        </div>

        <!-- ヘルステストビジュアル -->
        <div class="section-card">
            <div class="section-title">
                <span>
                    <span class="lang-ja">🛡️ ヘルステスト推奨設定 (Section 4.4, α = 2⁻²⁰)</span>
                    <span class="lang-en">🛡️ Health Test Cutoffs (Section 4.4, α = 2⁻²⁰)</span>
                </span>
            </div>
            <div class="grid-2">
                <div class="subcard">
                    <div class="subcard-title">4.4.1 Repetition Count Test (RCT)</div>
                    <div class="subcard-val" style="color: var(--neon-pink);">C = {health.get('rct_cutoff', '-')} <span class="unit">回 / reps</span></div>
                    <div class="subcard-desc">
                        <span class="lang-ja">ノイズ源のスタック（固着）故障を即座に検出。同一値が連続 C 回現れたら異常。</span>
                        <span class="lang-en">Detects catastrophic stuck faults immediately. Signals failure on C consecutive identical samples.</span>
                    </div>
                </div>
                <div class="subcard">
                    <div class="subcard-title">4.4.2 Adaptive Proportion Test (APT)</div>
                    <div class="subcard-val" style="color: var(--neon-cyan);">C = {health.get('apt_cutoff', '-')} / W = {health.get('apt_window', '-')}</div>
                    <div class="subcard-desc">
                        <span class="lang-ja">環境変化や劣化による偏りを検出。窓長 W サンプル内で同一値が C 回以上現れたら異常。</span>
                        <span class="lang-en">Detects gradual degradation. Signals failure if a sample appears C or more times within window W.</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Chart.js スクリプト -->
    <script>
        let currentLang = 'ja';

        function toggleLanguage() {{
            const body = document.body;
            if (currentLang === 'ja') {{
                currentLang = 'en';
                body.classList.remove('lang-mode-ja');
                body.classList.add('lang-mode-en');
            }} else {{
                currentLang = 'ja';
                body.classList.remove('lang-mode-en');
                body.classList.add('lang-mode-ja');
            }}
            updateChartLabels();
        }}

        // チャート共通ダークテーマ
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'JetBrains Mono', monospace";
        Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';

        // 1. 情報理論エントロピー比較
        const infoCtx = document.getElementById('infoChart');
        let infoChart = null;
        if (infoCtx && window.Chart) {{
            infoChart = new Chart(infoCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(info_labels_ja, ensure_ascii=False)},
                    datasets: [{{
                        data: {json.dumps(info_values)},
                        backgroundColor: [
                            'rgba(168, 85, 247, 0.8)',
                            'rgba(192, 132, 252, 0.7)',
                            'rgba(0, 240, 255, 0.75)',
                            'rgba(56, 189, 248, 0.7)',
                            'rgba(250, 204, 21, 0.75)',
                            'rgba(255, 0, 85, 0.85)'
                        ],
                        borderRadius: 6,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        y: {{ beginAtZero: true, max: {max_entropy_possible}, title: {{ display: true, text: 'bits/sample' }} }},
                        x: {{ ticks: {{ autoSkip: false, font: {{ size: 10 }} }} }}
                    }}
                }}
            }});
        }}

        // 2. SP 800-90B 各推定器比較
        const spCtx = document.getElementById('spChart');
        let spChart = null;
        if (spCtx && window.Chart) {{
            spChart = new Chart(spCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(sp_chart_labels_ja, ensure_ascii=False)},
                    datasets: [{{
                        data: {json.dumps(sp_chart_values)},
                        backgroundColor: {json.dumps(sp_chart_bg)},
                        borderColor: {json.dumps(sp_chart_border)},
                        borderWidth: 1,
                        borderRadius: 6,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        y: {{ beginAtZero: true, max: {max_entropy_possible}, title: {{ display: true, text: 'bits/sample' }} }},
                        x: {{ ticks: {{ autoSkip: false, font: {{ size: 9 }} }} }}
                    }}
                }}
            }});
        }}

        // 3. 時系列推移
        {f'''
        const tsCtx = document.getElementById('tsChart');
        if (tsCtx && window.Chart) {{
            new Chart(tsCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(ts_labels)},
                    datasets: [
                        {{
                            label: 'Shannon H₁',
                            data: {json.dumps(ts_shannon)},
                            borderColor: '#a855f7',
                            backgroundColor: 'transparent',
                            tension: 0.2,
                        }},
                        {{
                            label: 'Collision H₂',
                            data: {json.dumps(ts_collision)},
                            borderColor: '#00f0ff',
                            backgroundColor: 'transparent',
                            tension: 0.2,
                        }},
                        {{
                            label: 'Min-Entropy (MCV)',
                            data: {json.dumps(ts_mcv)},
                            borderColor: '#ff0055',
                            backgroundColor: 'transparent',
                            tension: 0.2,
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ beginAtZero: false, max: {max_entropy_possible}, title: {{ display: true, text: 'bits/sample' }} }}
                    }}
                }}
            }});
        }}
        ''' if has_ts else ""}

        // 4. ヒストグラム
        {f'''
        const histCtx = document.getElementById('histChart');
        if (histCtx && window.Chart) {{
            new Chart(histCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(hist_labels)},
                    datasets: [{{
                        label: 'Frequency',
                        data: {json.dumps(hist_counts)},
                        backgroundColor: 'rgba(0, 240, 255, 0.1)',
                        borderColor: '#00f0ff',
                        fill: true,
                        pointRadius: 0,
                        borderWidth: 1.5,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});
        }}
        ''' if has_hist else ""}

        function updateChartLabels() {{
            if (infoChart) {{
                infoChart.data.labels = currentLang === 'ja'
                    ? {json.dumps(info_labels_ja, ensure_ascii=False)}
                    : {json.dumps(info_labels_en, ensure_ascii=False)};
                infoChart.update();
            }}
            if (spChart) {{
                spChart.data.labels = currentLang === 'ja'
                    ? {json.dumps(sp_chart_labels_ja, ensure_ascii=False)}
                    : {json.dumps(sp_chart_labels_en, ensure_ascii=False)};
                spChart.update();
            }}
        }}
    </script>
</body>
</html>
"""
    return html
