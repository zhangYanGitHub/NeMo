#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_report_pipeline.py

单语言一键流水线：从「生成」到「汇总」一步完成。

对同一份验证集（``data/val_datasets/<locale>/*.csv``）分别用 **FP32**（``model.onnx``）
与 **INT8**（``model_int8.onnx``）跑纯 ONNX 推理，以 espeak-ng（piper_phonemize）产出的 IPA 为
参考（target），逐验证集算 PER 并生成 ``report.html``，最后汇总成一张 FP32 vs INT8 对比页。

目录约定（可用参数覆盖）：
  - 模型: data/model/<locale>/<run>/{model.onnx, model_int8.onnx, model.g2p_export_meta.json}
  - 输入: data/val_datasets/<locale>/*.csv（列: text,data_type）
  - 输出: data/output/<locale>/<run>/<dataset>/report.html          （FP32）
          data/output/<locale>/<run>_int8/<dataset>/report.html     （INT8）
          data/output/<locale>/<run>_comparison_summary.html        （汇总）

用法（在本目录内执行）：
  python run_report_pipeline.py --locale en_US
  python run_report_pipeline.py --locale en_US --limit 100
  python run_report_pipeline.py --locale de_DE \
      --model-dir data/model/de_DE/model_0715_epoch_205 \
      --val-dir   data/val_datasets/de_DE \
      --output-dir data/output
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from espeak_ng_client import piper_ipa_raw_batch, read_texts_from_csv, resolve_piper_espeak_voice  # noqa: E402
from g2p_evaluate import aggregate_metrics, prepare_results, render_multi_locale_report  # noqa: E402
from g2p_nemo_client import get_nemo_g2p_client  # noqa: E402
from g2p_text_frontend import number_locale_for, text_to_segments  # noqa: E402
from generate_comparison_summary import DATASET_LABELS, generate_summary_html  # noqa: E402

# 验证集文件名 → 汇总页数据集 id（须与 generate_comparison_summary.DATASET_LABELS 的 key 对齐）。
# 基础文件 <locale>.csv → nav_template；其余按文件名子串匹配。
_DATASET_ID_BY_SUBSTR: Tuple[Tuple[str, str], ...] = (
    ("core_navigation", "core_navigation"),
    ("navigation_extension", "navigation_extension"),
    ("long_tail_generalization", "long_tail_generalization"),
    ("random_ood_mixed", "generate"),
    ("generate", "generate"),
)


def _map_csv_to_dataset_id(csv_path: Path, locale: str) -> Optional[str]:
    stem = csv_path.stem
    if stem == locale:
        return "nav_template"
    for substr, ds_id in _DATASET_ID_BY_SUBSTR:
        if substr in stem:
            return ds_id
    return None


def _discover_model_dir(model_dir: Path) -> Path:
    """接受叶子目录或其父目录（若父目录下恰好只有一个含 model.onnx 的子目录则自动下钻）。"""
    if (model_dir / "model.onnx").is_file():
        return model_dir
    subdirs = [d for d in sorted(model_dir.glob("*")) if d.is_dir() and (d / "model.onnx").is_file()]
    if len(subdirs) == 1:
        return subdirs[0]
    if not subdirs:
        raise FileNotFoundError(f"在 {model_dir} 下未找到含 model.onnx 的目录")
    raise ValueError(f"{model_dir} 下有多个候选模型目录，请用 --model-dir 明确指定其一: {[str(s) for s in subdirs]}")


def _join_segment_ipas(seg_lists: List[List[str]], flat_ipas: List[str]) -> List[str]:
    """把按片段展平的 IPA 结果按原句重新聚合，词间单空格拼接（与训练一致）。"""
    out: List[str] = []
    idx = 0
    for segs in seg_lists:
        k = len(segs)
        out.append(" ".join(x for x in flat_ipas[idx : idx + k] if x))
        idx += k
    return out


class _DatasetInput:
    """一个验证集的输入与共享的 espeak 参考（FP32/INT8 复用，避免重复调用 espeak）。"""

    def __init__(self, ds_id: str, label: str, csv_path: Path, texts: List[str], seg_lists: List[List[str]],
                 flat_segments: List[str], espeak_ipas: List[str]) -> None:
        self.ds_id = ds_id
        self.label = label
        self.csv_path = csv_path
        self.texts = texts
        self.seg_lists = seg_lists
        self.flat_segments = flat_segments
        self.espeak_ipas = espeak_ipas


def _prepare_dataset_inputs(
    val_dir: Path, locale: str, limit: Optional[int]
) -> List[_DatasetInput]:
    voice = resolve_piper_espeak_voice(locale)
    num_loc = number_locale_for(locale)
    csv_files = sorted(p for p in val_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{val_dir} 下没有 *.csv 验证集")

    inputs: List[_DatasetInput] = []
    for csv_path in csv_files:
        ds_id = _map_csv_to_dataset_id(csv_path, locale)
        if ds_id is None:
            print(f"[跳过] 无法映射到已知数据集 id: {csv_path.name}", flush=True)
            continue
        label = DATASET_LABELS.get(ds_id, ds_id)
        texts = read_texts_from_csv(csv_path)
        if limit is not None:
            texts = texts[:limit]
        seg_lists = [text_to_segments(t, num_loc) for t in texts]
        flat_segments = [s for segs in seg_lists for s in segs]
        print(
            f"[{label}] {csv_path.name} · {len(texts)} 条 → {len(flat_segments)} 片段 · espeak voice={voice}",
            flush=True,
        )
        espeak_flat = piper_ipa_raw_batch(flat_segments, locale, locale=locale) if flat_segments else []
        espeak_ipas = _join_segment_ipas(seg_lists, espeak_flat)
        inputs.append(_DatasetInput(ds_id, label, csv_path, texts, seg_lists, flat_segments, espeak_ipas))
    return inputs


def _run_variant(
    *,
    variant_name: str,
    onnx_path: Path,
    meta_path: Path,
    inputs: List[_DatasetInput],
    locale: str,
    out_dir: Path,
    batch_size: int,
) -> None:
    """对一个模型变体（FP32 或 INT8）跑全部验证集并写出各自的 report.html。"""
    g2p_lang_tag = locale.replace("_", "-")
    print(f"\n=== 变体 {variant_name}: {onnx_path.name} ===", flush=True)
    client = get_nemo_g2p_client(onnx=onnx_path, meta=meta_path, batch_size=batch_size, show_startup_progress=True)
    model_name = f"G2P-ONNX ({onnx_path.name})"

    for din in inputs:
        t0 = time.perf_counter()
        g2p_flat = (
            client.phonemize_ipa_batch(din.flat_segments, g2p_lang_tag, locale=locale, show_progress=True)
            if din.flat_segments
            else []
        )
        g2p_ipas = _join_segment_ipas(din.seg_lists, g2p_flat)

        non_empty = sum(1 for x in g2p_ipas if (x or "").strip())
        if g2p_ipas and non_empty == 0:
            raise RuntimeError(
                f"[{variant_name}/{din.label}] G2P 推理结果全为空，通常是 ONNX/meta/checkpoint 组合异常。"
                f" sample_text={din.texts[0]!r}" if din.texts else ""
            )

        raw_rows = [
            {"text": t, "target": tgt, "predict": pred}
            for t, tgt, pred in zip(din.texts, din.espeak_ipas, g2p_ipas)
        ]
        results = prepare_results(raw_rows)
        stats = aggregate_metrics(results)
        stats.update(
            {
                "locale": din.ds_id,
                "label": din.label,
                "espeak_voice": resolve_piper_espeak_voice(locale),
                "g2p_lang_tag": g2p_lang_tag,
                "g2p_supported": True,
            }
        )
        report_path = out_dir / din.ds_id / "report.html"
        render_multi_locale_report(
            report_path,
            summaries=[stats],
            rows_by_locale={din.ds_id: results},
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            g2p_model=model_name,
            elapsed_s=round(time.perf_counter() - t0, 1),
        )
        print(
            f"  [{variant_name}/{din.label}] n={stats['n']} · PER={stats['avg_per']:.2f}% · "
            f"exact={stats['exact_match_rate'] * 100:.1f}% → {report_path}",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="单语言一键：生成各验证集报告并汇总 FP32 vs INT8 对比页")
    ap.add_argument("--locale", required=True, help="语言，如 en_US / de_DE / fr_FR")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="含 model.onnx / model_int8.onnx / model.g2p_export_meta.json 的目录；"
                         "默认 data/model/<locale>（若其下仅一个模型目录则自动下钻）")
    ap.add_argument("--val-dir", type=Path, default=None, help="验证集目录，默认 data/val_datasets/<locale>")
    ap.add_argument("--output-dir", type=Path, default=None, help="输出根目录，默认 data/output")
    ap.add_argument("--limit", type=int, default=None, help="每个验证集最多处理条数")
    ap.add_argument("--batch-size", type=int, default=1, help="单次 ORT 前向合并条数（默认 1，与端侧导出 profile 一致）")
    ap.add_argument("--skip-int8", action="store_true", help="只跑 FP32（无 INT8 模型时使用）")
    args = ap.parse_args()

    locale = args.locale
    data_dir = _PKG_DIR / "data"
    model_dir = _discover_model_dir(args.model_dir or (data_dir / "model" / locale))
    val_dir = args.val_dir or (data_dir / "val_datasets" / locale)
    output_root = args.output_dir or (data_dir / "output")

    onnx_fp32 = model_dir / "model.onnx"
    onnx_int8 = model_dir / "model_int8.onnx"
    meta_path = model_dir / "model.g2p_export_meta.json"
    for p in (onnx_fp32, meta_path):
        if not p.is_file():
            raise FileNotFoundError(f"缺少文件: {p}")
    has_int8 = onnx_int8.is_file() and not args.skip_int8
    if not onnx_int8.is_file() and not args.skip_int8:
        print(f"[提示] 未找到 {onnx_int8.name}，将只生成 FP32 报告（可加 --skip-int8 消除本提示）", flush=True)

    run_name = model_dir.name
    locale_out = output_root / locale
    fp32_dir = locale_out / run_name
    int8_dir = locale_out / f"{run_name}_int8"
    summary_path = locale_out / f"{run_name}_comparison_summary.html"

    print(
        f"语言: {locale}\n模型目录: {model_dir}\n验证集: {val_dir}\n输出: {locale_out}",
        flush=True,
    )

    inputs = _prepare_dataset_inputs(val_dir, locale, args.limit)
    if not inputs:
        raise RuntimeError("没有可用的验证集输入")

    _run_variant(
        variant_name="FP32", onnx_path=onnx_fp32, meta_path=meta_path,
        inputs=inputs, locale=locale, out_dir=fp32_dir, batch_size=args.batch_size,
    )
    if has_int8:
        _run_variant(
            variant_name="INT8", onnx_path=onnx_int8, meta_path=meta_path,
            inputs=inputs, locale=locale, out_dir=int8_dir, batch_size=args.batch_size,
        )

    generate_summary_html(
        output_path=summary_path,
        fp32_dir=fp32_dir,
        int8_dir=int8_dir if has_int8 else None,
        fp32_model_path=onnx_fp32,
        int8_model_path=onnx_int8 if has_int8 else None,
        title=f"{locale} · {run_name} · FP32 vs INT8 验证汇总",
    )
    print(f"\n完成。汇总页: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
