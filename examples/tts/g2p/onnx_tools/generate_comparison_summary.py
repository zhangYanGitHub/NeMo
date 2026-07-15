#!/usr/bin/env python3
"""生成 G2P 多验证集、多模型对比汇总页（单文件、双击即可打开，无需 HTTP 服务）。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from g2p_evaluate import (  # noqa: E402
    aggregate_metrics,
    aggregate_metrics_excluding_letter_a,
    prepare_results,
)


DATASET_LABELS = {
    "nav_template": "导航模版指令",
    "core_navigation": "核心导航指令",
    "navigation_extension": "导航扩展",
    "long_tail_generalization": "长尾泛化",
    "generate": "通用领域泛化",
}


def _parse_report(report_path: Path) -> Dict:
    html = report_path.read_text(encoding="utf-8")
    kpis = re.findall(
        r'<div class="kpi"><div class="lbl">([^<]+)</div><div class="val">([^<]+)</div></div>',
        html,
    )
    kpi = dict(kpis)
    m = re.search(r'summaries-json">(\[.*?\])</script>', html, re.DOTALL)
    summary = json.loads(m.group(1))[0] if m else {}
    gen_m = re.search(r"生成时间 ([^<]+)</span>", html)
    model_m = re.search(r"G2P 模型 ([^<]+)</span>", html)
    return {
        "n": int(kpi.get("总样本数", summary.get("n", 0))),
        "per_str": kpi.get("加权音素 PER", kpi.get("加权平均 PER", "—")),
        "per": float(summary.get("avg_per", 0)),
        "exact_str": kpi.get("音素完全匹配率", kpi.get("全局完全匹配率", "—")),
        "exact": float(summary.get("exact_match_rate", 0)) * 100,
        "warn": summary.get("warn_count", 0),
        "fail": summary.get("fail_count", 0),
        "space": int(summary.get("space_error_count", 0) or 0),
        "space_rate": float(summary.get("space_error_rate", 0) or 0) * 100,
        "pure_space": int(summary.get("pure_space_error_count", 0) or 0),
        "pure_space_rate": float(summary.get("pure_space_error_rate", 0) or 0) * 100,
        "generated_at": gen_m.group(1).strip() if gen_m else "",
        "model": model_m.group(1).strip() if model_m else "",
    }


def _fmt_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _delta_per(fp32: float, int8: float) -> str:
    d = int8 - fp32
    sign = "+" if d > 0 else ""
    cls = "bad" if d > 0.01 else ("good" if d < -0.01 else "neutral")
    return f'<span class="delta {cls}">{sign}{d:.2f}%</span>'


def _delta_exact(fp32: float, int8: float) -> str:
    d = int8 - fp32
    sign = "+" if d > 0 else ""
    cls = "good" if d > 0.1 else ("bad" if d < -0.1 else "neutral")
    return f'<span class="delta {cls}">{sign}{d:.2f}%</span>'


def _weighted_avg(rows: List[Dict], key: str) -> float:
    total_n = sum(r["n"] for r in rows)
    if total_n == 0:
        return 0.0
    return sum(r["n"] * r[key] for r in rows) / total_n


def _sum_field(rows: Dict[str, Dict], field: str) -> int:
    return sum(int(r.get(field, 0) or 0) for r in rows.values())


def _space_cell(row: Dict) -> str:
    if not row:
        return "—"
    return f"{row.get('space', 0)} ({row.get('space_rate', 0):.1f}%)"


def _resolve_int8_dir(fp32_dir: Path, int8_dir: Path | None) -> Path | None:
    """解析 INT8 输出目录，兼容 _int8 / _in8 等命名。"""
    candidates: List[Path] = []
    if int8_dir:
        candidates.append(int8_dir)
    parent, stem = fp32_dir.parent, fp32_dir.name
    for suffix in ("_int8", "_in8", "_INT8"):
        candidates.append(parent / f"{stem}{suffix}")
    seen: set[str] = set()
    for c in candidates:
        key = str(c.resolve())
        if key in seen:
            continue
        seen.add(key)
        if c.exists():
            return c
    return None


def _relative_url(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path, base_dir)).as_posix()


def _empty_report_url() -> str:
    return ""


def generate_summary_html(
    *,
    output_path: Path,
    fp32_dir: Path,
    int8_dir: Path | None,
    fp32_model_path: Path | None,
    int8_model_path: Path | None,
    title: str,
) -> None:
    int8_dir = _resolve_int8_dir(fp32_dir, int8_dir)
    int8_missing = int8_dir is None
    if int8_missing:
        int8_dir = fp32_dir  # placeholder，避免后续路径报错

    datasets: List[Tuple[str, str]] = []
    for ds_id, ds_label in DATASET_LABELS.items():
        fp32_ok = (fp32_dir / ds_id / "report.html").exists()
        int8_ok = not int8_missing and (int8_dir / ds_id / "report.html").exists()
        if fp32_ok or int8_ok:
            datasets.append((ds_id, ds_label))

    fp32_rows: Dict[str, Dict] = {}
    int8_rows: Dict[str, Dict] = {}
    for ds_id, _ in datasets:
        fp32_report = fp32_dir / ds_id / "report.html"
        int8_report = int8_dir / ds_id / "report.html"
        if fp32_report.exists():
            fp32_rows[ds_id] = _parse_report(fp32_report)
        if not int8_missing and int8_report.exists():
            int8_rows[ds_id] = _parse_report(int8_report)

    fp32_size = _fmt_size(fp32_model_path.stat().st_size) if fp32_model_path and fp32_model_path.exists() else "—"
    int8_size = _fmt_size(int8_model_path.stat().st_size) if int8_model_path and int8_model_path.exists() else "—"
    size_ratio = ""
    if fp32_model_path and int8_model_path and fp32_model_path.exists() and int8_model_path.exists():
        ratio = int8_model_path.stat().st_size / fp32_model_path.stat().st_size * 100
        size_ratio = f"（约为 FP32 的 {ratio:.0f}%）"

    fp32_all = list(fp32_rows.values())
    int8_all = list(int8_rows.values())
    fp32_w_per = _weighted_avg(fp32_all, "per")
    int8_w_per = _weighted_avg(int8_all, "per")
    fp32_w_exact = _weighted_avg(fp32_all, "exact")
    int8_w_exact = _weighted_avg(int8_all, "exact")
    total_n = sum(r["n"] for r in fp32_all) or sum(r["n"] for r in int8_all)
    fp32_warn = _sum_field(fp32_rows, "warn")
    fp32_fail = _sum_field(fp32_rows, "fail")
    int8_warn = _sum_field(int8_rows, "warn")
    int8_fail = _sum_field(int8_rows, "fail")
    fp32_space = _sum_field(fp32_rows, "space")
    fp32_space_rate = fp32_space / total_n * 100 if total_n else 0.0
    int8_space = _sum_field(int8_rows, "space")
    int8_space_rate = int8_space / total_n * 100 if total_n and int8_all else 0.0

    report_urls: Dict[str, Dict[str, str]] = {"fp32": {}, "int8": {}}
    output_dir = output_path.parent
    for ds_id, _ in datasets:
        fp32_report = fp32_dir / ds_id / "report.html"
        int8_report = int8_dir / ds_id / "report.html"
        if fp32_report.exists():
            report_urls["fp32"][ds_id] = _relative_url(fp32_report, output_dir)
        if not int8_missing and int8_report.exists():
            report_urls["int8"][ds_id] = _relative_url(int8_report, output_dir)

    table_rows = []
    for ds_id, ds_label in datasets:
        f = fp32_rows.get(ds_id, {})
        i = int8_rows.get(ds_id, {})
        f_wf = f"{f.get('warn', 0)} / {f.get('fail', 0)}" if f else "—"
        i_wf = f"{i.get('warn', 0)} / {i.get('fail', 0)}" if i else "—"
        table_rows.append(
            f"""<tr data-ds="{ds_id}">
              <td><strong>{ds_label}</strong><div class="ds-id">{ds_id}</div></td>
              <td>{f.get('n') or i.get('n') or '—'}</td>
              <td>{f.get('per_str', '—')}</td>
              <td>{f.get('exact_str', '—')}</td>
              <td>{_space_cell(f)}</td>
              <td>{f_wf}</td>
              <td>{i.get('per_str', '—')}</td>
              <td>{i.get('exact_str', '—')}</td>
              <td>{_space_cell(i)}</td>
              <td>{i_wf}</td>
              <td>{_delta_per(f.get('per', 0), i.get('per', 0)) if f and i else '—'}</td>
              <td>{_delta_exact(f.get('exact', 0), i.get('exact', 0)) if f and i else '—'}</td>
              <td><button type="button" class="link-btn" data-goto="{ds_id}">查看报告 →</button></td>
            </tr>"""
        )

    int8_per_cell = f"{int8_w_per:.2f}%" if int8_all else "—"
    int8_exact_cell = f"{int8_w_exact:.2f}%" if int8_all else "—"
    int8_wf_cell = f"{int8_warn} / {int8_fail}" if int8_all else "—"
    int8_space_cell = f"{int8_space} ({int8_space_rate:.1f}%)" if int8_all else "—"
    delta_per_cell = _delta_per(fp32_w_per, int8_w_per) if int8_all else "—"
    delta_exact_cell = _delta_exact(fp32_w_exact, int8_w_exact) if int8_all else "—"

    table_footer = f"""<tr class="totals-row">
              <td><strong>合计</strong></td>
              <td>{total_n}</td>
              <td>{fp32_w_per:.2f}%</td>
              <td>{fp32_w_exact:.2f}%</td>
              <td>{fp32_space} ({fp32_space_rate:.1f}%)</td>
              <td>{fp32_warn} / {fp32_fail}</td>
              <td>{int8_per_cell}</td>
              <td>{int8_exact_cell}</td>
              <td>{int8_space_cell}</td>
              <td>{int8_wf_cell}</td>
              <td>{delta_per_cell}</td>
              <td>{delta_exact_cell}</td>
              <td></td>
            </tr>"""

    int8_dir_note = ""
    if int8_missing:
        int8_dir_note = (
            '<div class="alert-banner">⚠️ 未找到 INT8 验证输出目录，INT8 列数据为空。'
            f"请确认目录存在（如 {fp32_dir.name}_int8 或 {fp32_dir.name}_in8）后重新生成。</div>"
        )
    elif int8_dir:
        int8_dir_note = f'<div class="info-banner">INT8 数据来源：<code>{int8_dir.name}</code></div>'

    detail_panels = []
    for ds_id, ds_label in datasets:
        fp32_src = report_urls["fp32"].get(ds_id, _empty_report_url())
        int8_src = report_urls["int8"].get(ds_id, _empty_report_url())
        detail_panels.append(
            f"""<div class="panel detail-panel" id="panel-{ds_id}" hidden>
              <div class="panel-head">
                <h2>{ds_label}</h2>
                <div class="model-tabs" data-ds="{ds_id}">
                  <button type="button" class="model-tab active" data-model="fp32">FP32 · model.onnx</button>
                  <button type="button" class="model-tab" data-model="int8">INT8 · model_int8.onnx</button>
                </div>
              </div>
              <div class="iframe-wrap">
                <iframe class="report-frame" src="about:blank" data-src="{fp32_src}" data-ds="{ds_id}" data-model="fp32" title="{ds_label} FP32"></iframe>
                <iframe class="report-frame" src="about:blank" data-src="{int8_src}" data-ds="{ds_id}" data-model="int8" title="{ds_label} INT8" hidden></iframe>
              </div>
            </div>"""
        )

    nav_tabs = ['<button type="button" class="main-tab active" data-tab="overview">汇总对比</button>']
    for ds_id, ds_label in datasets:
        nav_tabs.append(f'<button type="button" class="main-tab" data-tab="{ds_id}">{ds_label}</button>')

    fp32_model_name = fp32_all[0]["model"] if fp32_all else "model.onnx"
    int8_model_name = int8_all[0]["model"] if int8_all else "model_int8.onnx"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{title}</title>
  <style>
    :root{{
      --bg:#0f172a; --card:#fff; --muted:#64748b; --line:#e2e8f0;
      --ok:#10b981; --warn:#f59e0b; --fail:#ef4444; --accent:#3b82f6;
      --fp32:#2563eb; --int8:#7c3aed;
    }}
    *{{box-sizing:border-box;}}
    body{{margin:0;font-family:system-ui,-apple-system,sans-serif;background:#f1f5f9;color:#0f172a;}}
    .hero{{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);color:#f8fafc;padding:28px 24px 20px;}}
    .hero-inner{{max-width:1400px;margin:0 auto;}}
    .hero h1{{margin:0 0 6px;font-size:24px;font-weight:700;}}
    .hero .sub{{opacity:.85;font-size:13px;line-height:1.5;max-width:900px;}}
    .hero-meta{{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px;font-size:12px;}}
    .hero-meta span{{background:rgba(255,255,255,.12);padding:5px 10px;border-radius:8px;}}
    .model-cards{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px;}}
    .model-card{{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);border-radius:12px;padding:14px 16px;}}
    .model-card .tag{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;opacity:.8;}}
    .model-card.fp32 .tag{{color:#93c5fd;}}
    .model-card.int8 .tag{{color:#c4b5fd;}}
    .model-card .name{{font-size:14px;font-weight:600;margin:4px 0;}}
    .model-card .size{{font-size:22px;font-weight:700;}}
    .layout{{max-width:1400px;margin:0 auto;padding:16px 24px 32px;}}
    .main-nav{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px;position:sticky;top:0;z-index:10;background:#f1f5f9;padding:8px 0;}}
    .main-tab{{font:inherit;cursor:pointer;font-size:13px;font-weight:600;padding:8px 14px;border-radius:999px;background:#fff;color:#334155;border:1px solid var(--line);box-shadow:0 1px 3px rgba(15,23,42,.06);}}
    .main-tab:hover{{border-color:#93c5fd;}}
    .main-tab.active{{background:var(--accent);color:#fff;border-color:var(--accent);}}
    .panel{{background:var(--card);border-radius:14px;box-shadow:0 4px 20px rgba(15,23,42,.08);overflow:hidden;}}
    .panel-head{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;padding:16px 20px;border-bottom:1px solid var(--line);}}
    .panel-head h2{{margin:0;font-size:17px;}}
    .kpi-row{{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;padding:16px 20px;border-bottom:1px solid var(--line);}}
    .kpi{{padding:12px 14px;background:#f8fafc;border-radius:10px;}}
    .kpi .lbl{{font-size:11px;color:var(--muted);font-weight:500;}}
    .kpi .val{{font-size:20px;font-weight:700;margin-top:4px;}}
    table{{width:100%;border-collapse:collapse;font-size:12px;}}
    th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);}}
    th{{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);background:#f8fafc;position:sticky;top:0;}}
    tr:hover td{{background:#f8fafc;}}
    .ds-id{{font-size:10px;color:var(--muted);margin-top:2px;}}
    .delta{{font-weight:700;font-size:11px;padding:2px 6px;border-radius:4px;}}
    .delta.good{{background:#d1fae5;color:#065f46;}}
    .delta.bad{{background:#fee2e2;color:#991b1b;}}
    .delta.neutral{{background:#f1f5f9;color:#475569;}}
    .link-btn{{font:inherit;cursor:pointer;font-size:12px;font-weight:600;color:var(--accent);background:none;border:none;padding:0;}}
    .link-btn:hover{{text-decoration:underline;}}
    .alert-banner{{background:#fef2f2;border:1px solid #fecaca;color:#991b1b;padding:10px 14px;border-radius:10px;font-size:13px;margin-bottom:12px;}}
    .info-banner{{background:#eff6ff;border:1px solid #bfdbfe;color:#1e40af;padding:8px 14px;border-radius:10px;font-size:12px;margin-bottom:12px;}}
    .info-banner code{{font-family:ui-monospace,Menlo,monospace;font-size:11px;background:rgba(255,255,255,.6);padding:2px 6px;border-radius:4px;}}
    .totals-row td{{background:#f8fafc;font-weight:700;border-top:2px solid var(--line);}}
    .col-group{{text-align:center;font-size:10px;}}
    .col-fp32{{background:#eff6ff!important;}}
    .col-int8{{background:#f5f3ff!important;}}
    .table-wrap{{overflow:auto;max-height:none;}}
    .model-tabs{{display:flex;gap:6px;}}
    .model-tab{{font:inherit;cursor:pointer;font-size:12px;font-weight:600;padding:6px 12px;border-radius:8px;background:#f1f5f9;color:#475569;border:1px solid var(--line);}}
    .model-tab.active[data-model="fp32"]{{background:var(--fp32);color:#fff;border-color:var(--fp32);}}
    .model-tab.active[data-model="int8"]{{background:var(--int8);color:#fff;border-color:var(--int8);}}
    .iframe-wrap{{height:calc(100vh - 220px);min-height:500px;}}
    .report-frame{{width:100%;height:100%;border:none;display:block;}}
    .footer{{text-align:center;font-size:11px;color:var(--muted);padding:16px;}}
    @media(max-width:768px){{
      .model-cards{{grid-template-columns:1fr;}}
      .iframe-wrap{{height:calc(100vh - 280px);}}
    }}
  </style>
</head>
<body>
  <div class="hero">
    <div class="hero-inner">
      <h1>{title}</h1>
      <div class="sub">FP32 与 INT8 量化版本在 5 份验证集上的 G2P 发音一致性对比 · 汇总页轻量打开 · 点击下方页签按需加载各数据集详细报告</div>
      <div class="hero-meta">
        <span>生成时间 {generated}</span>
        <span>总样本 {total_n}</span>
        <span>验证集 {len(datasets)} 份</span>
      </div>
      <div class="model-cards">
        <div class="model-card fp32">
          <div class="tag">FP32</div>
          <div class="name">{fp32_model_name}</div>
          <div class="size">{fp32_size}</div>
        </div>
        <div class="model-card int8">
          <div class="tag">INT8 量化</div>
          <div class="name">{int8_model_name}</div>
          <div class="size">{int8_size} {size_ratio}</div>
        </div>
      </div>
    </div>
  </div>

  <div class="layout">
    {int8_dir_note}
    <div class="main-nav" id="main-nav">
      {''.join(nav_tabs)}
    </div>

    <div class="panel" id="panel-overview">
      <div class="kpi-row">
        <div class="kpi"><div class="lbl">FP32 加权音素 PER</div><div class="val">{fp32_w_per:.2f}%</div></div>
        <div class="kpi"><div class="lbl">INT8 加权音素 PER</div><div class="val">{int8_per_cell}</div></div>
        <div class="kpi"><div class="lbl">PER 变化</div><div class="val">{delta_per_cell}</div></div>
        <div class="kpi"><div class="lbl">FP32 音素匹配率</div><div class="val">{fp32_w_exact:.2f}%</div></div>
        <div class="kpi"><div class="lbl">INT8 音素匹配率</div><div class="val">{int8_exact_cell}</div></div>
        <div class="kpi"><div class="lbl">匹配率变化</div><div class="val">{delta_exact_cell}</div></div>
        <div class="kpi"><div class="lbl">FP32 空格错误</div><div class="val">{fp32_space} ({fp32_space_rate:.1f}%)</div></div>
        <div class="kpi"><div class="lbl">INT8 空格错误</div><div class="val">{int8_space_cell}</div></div>
        <div class="kpi"><div class="lbl">FP32 Warn / Fail</div><div class="val">{fp32_warn} / {fp32_fail}</div></div>
        <div class="kpi"><div class="lbl">INT8 Warn / Fail</div><div class="val">{int8_wf_cell}</div></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th rowspan="2">验证集</th>
              <th rowspan="2">样本数</th>
              <th colspan="4" class="col-group col-fp32">FP32 (model.onnx)</th>
              <th colspan="4" class="col-group col-int8">INT8 (model_int8.onnx)</th>
              <th colspan="2" class="col-group">量化影响</th>
              <th rowspan="2"></th>
            </tr>
            <tr>
              <th class="col-fp32">音素 PER</th><th class="col-fp32">音素匹配</th><th class="col-fp32">空格错误</th><th class="col-fp32">Warn/Fail</th>
              <th class="col-int8">音素 PER</th><th class="col-int8">音素匹配</th><th class="col-int8">空格错误</th><th class="col-int8">Warn/Fail</th>
              <th>Δ PER</th><th>Δ 匹配率</th>
            </tr>
          </thead>
          <tbody>
            {''.join(table_rows)}
            {table_footer}
          </tbody>
        </table>
      </div>
    </div>

    {''.join(detail_panels)}

    <div class="footer">PER = phone-level 编辑距离 / 参考长度 · 明细报告内可点击表头排序 · 明细按需加载，无需把全部数据内联进汇总页</div>
  </div>

  <script>
  (function(){{
    const mainNav = document.getElementById('main-nav');
    const panels = {{
      overview: document.getElementById('panel-overview'),
    }};
    document.querySelectorAll('.detail-panel').forEach(p => {{
      panels[p.id.replace('panel-','')] = p;
    }});

    const emptyDoc = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/></head><body style="font-family:system-ui,sans-serif;padding:32px;color:#64748b;text-align:center"><p>该模型/验证集暂无报告数据</p></body></html>';

    function mountFrame(frame){{
      const ds = frame.dataset.ds;
      const model = frame.dataset.model;
      const key = model + ':' + ds;
      if(frame.dataset.loaded === key) return;

      const src = frame.dataset.src;
      if(src){{
        frame.removeAttribute('srcdoc');
        frame.src = src;
      }} else {{
        frame.srcdoc = emptyDoc;
      }}
      frame.dataset.loaded = key;
    }}

    function loadVisibleFrames(tabId){{
      if(tabId === 'overview') return;
      const panel = panels[tabId];
      if(!panel) return;
      panel.querySelectorAll('.report-frame:not([hidden])').forEach(mountFrame);
    }}

    function showTab(tabId){{
      Object.entries(panels).forEach(([id, el]) => {{
        if(!el) return;
        el.hidden = (id !== tabId);
      }});
      document.querySelectorAll('.main-tab').forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.tab === tabId);
      }});
      loadVisibleFrames(tabId);
    }}

    mainNav.addEventListener('click', e => {{
      const btn = e.target.closest('.main-tab');
      if(!btn) return;
      showTab(btn.dataset.tab);
    }});

    document.querySelectorAll('.link-btn').forEach(btn => {{
      btn.addEventListener('click', () => showTab(btn.dataset.goto));
    }});

    document.querySelectorAll('.model-tabs').forEach(group => {{
      const ds = group.dataset.ds;
      group.addEventListener('click', e => {{
        const tab = e.target.closest('.model-tab');
        if(!tab) return;
        const model = tab.dataset.model;
        group.querySelectorAll('.model-tab').forEach(b => b.classList.toggle('active', b === tab));
        document.querySelectorAll(`.report-frame[data-ds="${{ds}}"]`).forEach(frame => {{
          const show = frame.dataset.model === model;
          frame.hidden = !show;
          if(show) mountFrame(frame);
        }});
      }});
    }});

    showTab('overview');
  }})();
  </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"已生成汇总报告: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 G2P 多模型对比汇总 HTML")
    parser.add_argument("--fp32-dir", default="examples/tts/g2p/output/0705")
    parser.add_argument("--int8-dir", default="examples/tts/g2p/output/0705_int8")
    parser.add_argument("--fp32-model", default="examples/tts/g2p/model/0705/model.onnx")
    parser.add_argument("--int8-model", default="examples/tts/g2p/model/0705/model_int8.onnx")
    parser.add_argument(
        "--output",
        "-o",
        default="examples/tts/g2p/output/0705_comparison_summary.html",
    )
    parser.add_argument("--title", default="0705 · FP32 vs INT8 验证汇总")
    args = parser.parse_args()

    generate_summary_html(
        output_path=Path(args.output),
        fp32_dir=Path(args.fp32_dir),
        int8_dir=Path(args.int8_dir),
        fp32_model_path=Path(args.fp32_model),
        int8_model_path=Path(args.int8_model),
        title=args.title,
    )


if __name__ == "__main__":
    main()
