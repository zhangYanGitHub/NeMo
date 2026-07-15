# G2P Conformer-CTC ONNX 工具链

NeMo 训练出的 **Conformer-CTC G2P**：**导出 → 纯 ONNX 推理 → INT8 量化 → 验证报告**。

推理侧为纯 ONNX（`onnxruntime` + `*.g2p_export_meta.json`），不依赖 NeMo / torch 前向。

**命令约定**：下文均在 **NeMo 仓库根目录**执行（先 `cd` 到仓库根；Python 环境如 `conda activate tntts`）。

路径前缀（下文用 `$TOOLS` 代指）：

```text
TOOLS=examples/tts/g2p/onnx_tools
```

## 快速开始：单语言一键报告（推荐）

每次只跑**一种语言**：对该语言下全部验证集生成 FP32/INT8 分集报告，并汇总成一张对比 HTML。

```bash
conda activate tntts

# 先小样本自检（可选）
python examples/tts/g2p/onnx_tools/run_report_pipeline.py --locale en_US --limit 50

# 三种语言各执行一次（每次独立生成该语言的完整报告）
python examples/tts/g2p/onnx_tools/run_report_pipeline.py --locale en_US
python examples/tts/g2p/onnx_tools/run_report_pipeline.py --locale de_DE
python examples/tts/g2p/onnx_tools/run_report_pipeline.py --locale fr_FR
```

脚本按脚本文件位置解析 `data/`，**与当前工作目录无关**，可从仓库根直接运行。会自动定位 `data/model/<locale>/` 下唯一模型子目录，无需手写 `--model-dir`。

### 输出位置（以 en_US 为例）

模型目录为 `examples/tts/g2p/onnx_tools/data/model/en_US/model_0709_epoch_381/` 时，生成：

| 路径 | 说明 |
|------|------|
| `$TOOLS/data/output/en_US/model_0709_epoch_381/<dataset>/report.html` | FP32 各验证集报告 |
| `$TOOLS/data/output/en_US/model_0709_epoch_381_int8/<dataset>/report.html` | INT8 各验证集报告 |
| `$TOOLS/data/output/en_US/model_0709_epoch_381_comparison_summary.html` | **汇总对比页（双击打开）** |

`<dataset>` 对应： `nav_template` / `core_navigation` / `navigation_extension` / `long_tail_generalization` / `generate`。

### 目录约定（`data/` 已 gitignore）

```text
examples/tts/g2p/onnx_tools/
├── data/
│   ├── model/<locale>/<run>/
│   │   ├── model.onnx
│   │   ├── model_int8.onnx
│   │   └── model.g2p_export_meta.json
│   ├── val_datasets/<locale>/*.csv    # 列: text,data_type
│   └── output/<locale>/               # 报告输出（按语言分子目录）
│       ├── <run>/                     # FP32 各验证集 report.html
│       ├── <run>_int8/                # INT8 各验证集 report.html
│       └── <run>_comparison_summary.html
├── run_report_pipeline.py             # 单语言一键：生成 + 汇总
└── ...
```

当前已配置的模型目录（自动下钻）：

| 语言 | 模型目录（相对 `$TOOLS`） |
|------|---------------------------|
| `en_US` | `data/model/en_US/model_0709_epoch_381` |
| `de_DE` | `data/model/de_DE/model_0715_epoch_205` |
| `fr_FR` | `data/model/fr_FR/model_0715_epoch_199` |

若某语言下存在多个 `<run>` 子目录，需显式指定：

```bash
python examples/tts/g2p/onnx_tools/run_report_pipeline.py --locale de_DE \
  --model-dir examples/tts/g2p/onnx_tools/data/model/de_DE/model_0715_epoch_205
```

其它常用参数：

- `--limit N`：每个验证集只跑前 N 条（调试用）
- `--skip-int8`：无 INT8 模型时只出 FP32
- `--batch-size 1`：默认与端侧导出 profile 一致

## 安装依赖

```bash
pip install -r examples/tts/g2p/onnx_tools/requirements.txt
```

另需系统安装 **espeak-ng**（报告流水线用 piper 生成参考 IPA）：

```bash
brew install espeak-ng   # macOS
```

报告/推理/量化不需要 `torch`、`nemo_toolkit`；**仅导出 ONNX** 时需要。

## 脚本一览

| 脚本 | 作用 |
|------|------|
| **`run_report_pipeline.py`** | **单语言一键**：espeak 参考 + FP32/INT8 推理 → 各集 `report.html` → 汇总对比页 |
| `export_nemo_g2p_ctc_onnx.py` | 从 `.nemo` / `.ckpt` 导出 ONNX + `*.g2p_export_meta.json` |
| `g2p_nemo_client.py` | 纯 ONNX 推理客户端 |
| `g2p_text_frontend.py` | 推理文本前端（与训练 `text_normalize.py` 对齐） |
| `quantize_g2p_onnx.py` | INT8 静态量化 + CTC 一致率验证 |
| `g2p_evaluate.py` | PER 计算与 HTML 报告（已有 `text,target,predict` CSV 时用） |
| `generate_comparison_summary.py` | 仅汇总已有报告目录（分步进阶） |
| `espeak_ng_client.py` | espeak-ng 参考 IPA（`run_report_pipeline` 内部使用） |

## 其它流程（仓库根目录执行）

### 导出 ONNX（需 NeMo + torch）

`--nemo` 与 `--ckpt` 二选一：

```bash
python examples/tts/g2p/onnx_tools/export_nemo_g2p_ctc_onnx.py \
  --ckpt /path/to/last.ckpt \
  --out  examples/tts/g2p/onnx_tools/data/model/en_US/my_run/model.onnx \
  --device cpu
```

默认 `--profile mobile_dynamic_seq`（batch=1、序列长度动态）。

### 纯 ONNX 单句推理

```bash
python examples/tts/g2p/onnx_tools/g2p_nemo_client.py "hello world" \
  --onnx examples/tts/g2p/onnx_tools/data/model/en_US/model_0709_epoch_381/model.onnx \
  --meta examples/tts/g2p/onnx_tools/data/model/en_US/model_0709_epoch_381/model.g2p_export_meta.json
```

### INT8 量化

```bash
python examples/tts/g2p/onnx_tools/quantize_g2p_onnx.py \
  --onnx examples/tts/g2p/onnx_tools/data/model/en_US/model_0709_epoch_381/model.onnx \
  --json examples/tts/g2p/onnx_tools/data/model/en_US/model_0709_epoch_381/model.g2p_export_meta.json \
  --out  examples/tts/g2p/onnx_tools/data/model/en_US/model_0709_epoch_381/model_int8.onnx \
  --calib-csv examples/tts/g2p/onnx_tools/data/val_datasets/en_US/en_US_validation_core_navigation.csv
```

## 说明

- **PER**：忽略词边界空格后的 phone-level 编辑距离 / 参考音素长度；参考 IPA 来自 espeak-ng（GPL，仅评测用）。
- **train == serve**：`g2p_text_frontend.py` 须与 `examples/dataset/text_normalize.py` 保持同步。
- 纯 ONNX 推理需**同一次导出**的 `.onnx` + `.g2p_export_meta.json`；meta 过旧请重新导出。
