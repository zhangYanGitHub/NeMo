# G2P 数据集工具

本目录提供从 HuggingFace 下载 IPA-CHILDES 数据集、并用 `piper-phonemize` 批量生成 NeMo G2P 训练 manifest 的脚本。

整体流程：

```
下载 CSV 原始数据  →  批量 G2P 音素化  →  train.json + vocab.txt
```

## 推荐流程（ipa-childes-split → `dataset/release/en-US`，Conformer-CTC）

新增脚本：`examples/dataset/preprocess_ipa_childes_split.py`

- 只读取 CSV 两列：语言码（默认 `espeak_lang_code`）和原始文本（默认 `sentence`）
- 不使用 CSV 里的现成 IPA 字段；`text` 通过 **`piper-phonemize` 进程内 espeak backend** 重新生成（不再频繁启动 `espeak-ng` 子进程）
- 默认按 CPU 核数多线程并发；依赖当前 Python 环境里可导入 `piper_phonemize`。可在同一终端、同一 conda 环境里自检：

  ```bash
  python -c "import piper_phonemize; print('ok')"
  ```

  若导入失败，先安装依赖：`pip install -r examples/dataset/requirements.txt`
- 默认开启 **`tqdm` 进度条**（`--no-show-progress` 可关）；进度输出在 **stderr**，且 **`disable=False`**，即使用 `2>&1 | tee` 也会写入日志。实现上采用**有上限的并发提交 + 保序回收**，避免旧版 `executor.map` 先读完整个 CSV 再等工作、导致长时间停在 `0batch` 的问题。全表很大时开头仍会先扫一遍 CSV 统计换行（用于 `total`），终端会先出现一行 `Preprocess: counting newlines…`。
- 输出 manifest 里 `text` 为**空格分隔的 IPA 单元**（与 `examples/tts/g2p/conf/g2p_conformer_ctc.yaml` 中的 `ipa_symbol` tokenizer 一致）
- 输出文件（均在 `--output-dir` 下）：
  - `train.json` / `test.json`（由 `--manifest-name` 指定；每行 `text_graphemes` + `text`）
  - `phoneme_vocab.txt`、`grapheme_vocab.txt`
  - `vocab.txt`：音素与字符合集，**G2P 训练时 `model.tokenizer.dir` 指向的目录内需有该文件**
- 性能默认值已内置按 CPU 自动调优：`workers=max(1, cpu-2)`（上限 24），`batch-size` 随 CPU 自动取 `2048/4096/8192`；直接用下方命令即可，无需再额外传并行参数。

**词表策略：**先跑 **train**，写出全套词表；再跑 **test** 且必须加 `--no-write-vocab`，只生成 `test.json`。若对 test 也开启写词表，会按测试集重新统计并**覆盖** `vocab.txt`，通常不符合预期。

预处理命令（仓库根目录执行；en-US 相对路径示例如下。若在其它机器上，把 CSV / `--output-dir` 换成你的绝对路径即可，例如输入 `/path/to/NeMo/dataset/ipa-childes-split/.../data.csv`，输出 `/path/to/NeMo/dataset/release/en-US`）：

```bash
# 1) train：生成 train.json + vocab.txt（及 phoneme / grapheme 词表）
python examples/dataset/preprocess_ipa_childes_split.py \
  --input-csv dataset/ipa-childes-split/train/en-US/data.csv \
  --output-dir dataset/release/en-US \
  --manifest-name train.json \
  --merged-vocab-name vocab.txt \
  --phoneme-vocab-name phoneme_vocab.txt \
  --grapheme-vocab-name grapheme_vocab.txt

# 2) test：只生成 test.json，不覆盖 train 统计的词表
python examples/dataset/preprocess_ipa_childes_split.py \
  --input-csv dataset/ipa-childes-split/test/en-US/data.csv \
  --output-dir dataset/release/en-US \
  --manifest-name test.json \
  --no-write-vocab
```

可选：若下游需要 JSON 形式的音素表（例如与 `IPASymbolTokenizer.from_meta` 的 `phoneme_labels` 对齐），或复制字素词表为 `vocab_origin.txt`，在仓库根目录执行：

```bash
cd dataset/release/en-US

python3 -c "import json; p='phoneme_vocab.txt'; labels=[l.rstrip('\n') for l in open(p,encoding='utf-8') if l.strip()]; print(json.dumps({'phoneme_labels': labels}, ensure_ascii=False, indent=2))" > vocab_phoneme.json

# 若需单独保留「字素侧」词表副本（命名自定）
cp grapheme_vocab.txt vocab_origin.txt
```

训练命令：**模型与优化器、trainer、`exp_manager` 等均采用** `examples/tts/g2p/conf/g2p_conformer_ctc.yaml` **默认值**；命令行只覆盖数据路径、tokenizer 目录与实验输出目录。请在仓库根目录执行：

```bash
mkdir -p logs exp

python3 examples/tts/g2p/g2p_train_and_evaluate.py \
  --config-name=g2p_conformer_ctc \
  train_manifest=$PWD/dataset/release/en-US/train.json \
  validation_manifest=$PWD/dataset/release/en-US/train.json \
  test_manifest=$PWD/dataset/release/en-US/test.json \
  model.tokenizer.dir=$PWD/dataset/release/en-US \
  exp_manager.exp_dir=$PWD/exp/g2p_conformer_ctc_en_us_release \
  do_training=true \
  do_testing=true \
  2>&1 | tee "$PWD/logs/g2p_conformer_ctc_en_us_release_$(date +%F_%H-%M-%S).log"
```

说明：

- 配置见 `examples/tts/g2p/conf/g2p_conformer_ctc.yaml`（含 `model.tokenizer_grapheme`、`model.encoder`、`optim` / `sched`、`trainer`、`exp_manager` 等）；**无需**在命令行重复 `model.embedding.d_model`、`+optim.lr` 等与 YAML 相同的项。
- `validation_manifest` 与 `train_manifest` 指向同一文件与 YAML 默认用法一致；若你有单独验证集，改为对应 `val.json` 路径即可。
- `exp_manager.exp_dir` 在 YAML 中为 `null`，训练时必须通过命令行设为可写目录（上例为 `$PWD/exp/g2p_conformer_ctc_en_us_release`）。
- 导出 `.ckpt` / `.nemo` 后，下文「G2P 训练 checkpoint 与 manifest 后处理」中的路径请按你的 `exp_manager.exp_dir` 与实际 run 目录替换。

---

## 环境准备

建议使用独立的 conda 虚拟环境，避免与 NeMo 主环境冲突。

### 国内镜像（临时使用，不改配置文件）

以下方式仅对**当前命令/当前终端会话**生效，不会修改 `~/.condarc`、`pip config` 或 shell 配置文件。

#### Conda（清华源）

```bash
conda create -n g2p python=3.10 -y \
  --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
```

#### pip（清华源）

```bash
pip install -r examples/dataset/requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install "huggingface_hub>=0.24" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

阿里云镜像（临时）：`-i https://mirrors.aliyun.com/pypi/simple/`

#### HuggingFace（hf-mirror，仅当前终端）

```bash
export HF_ENDPOINT=https://hf-mirror.com

python examples/dataset/download_ipa_childes_split.py \
  --languages en-US \
  --splits train
```

或单行（不 export，只对这一条命令生效）：

```bash
HF_ENDPOINT=https://hf-mirror.com python examples/dataset/download_ipa_childes_split.py \
  --languages en-US \
  --splits train
```

---

### 创建并激活 `g2p` 环境

NeMo 训练环境要求 **Python 3.10+**（见仓库 `pyproject.toml` / `AGENTS.md`），CI 覆盖 3.10 / 3.11 / 3.12。  
建议 **`g2p` 环境与 NeMo 训练环境使用相同的 Python 小版本**，避免后续在同一机器上装 NeMo 时出现兼容问题。

在仓库根目录执行：

```bash
# 推荐与 NeMo 默认目标一致：Python 3.10
# 若你的 NeMo 训练环境是 3.11 或 3.12，把下面版本号改成一致即可
conda create -n g2p python=3.10 -y
conda activate g2p

# 安装本目录依赖
pip install -r examples/dataset/requirements.txt

# 下载脚本还需要 huggingface_hub
pip install "huggingface_hub>=0.24"
```

`piper-phonemize` 提供 Python 3.10–3.12 的预编译 wheel；与 NeMo 一样，**不建议使用 3.9 及以下**。

之后每次使用前激活环境：

```bash
conda activate g2p
```

#### `conda activate` 报错怎么办？

若出现类似错误：

```text
conda: error: argument COMMAND: invalid choice: 'activate'
```

**原因：** `activate` 不是 conda 的子命令，而是 conda 注入到 shell 里的**函数**。当前终端还没加载 conda 的 shell 钩子（常见于新开的终端、IDE 内置终端、或未执行过 `conda init`）。

**当前终端临时修复（推荐，不改配置文件）：**

```bash
# bash / zsh 通用：先 source，再 activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate g2p
```

**一劳永逸（需改 shell 配置，执行一次后重开终端）：**

```bash
conda init bash    # 若用 bash
# conda init zsh   # 若用 zsh
```

然后**关闭并重新打开终端**，再执行 `conda activate g2p`。

**不 activate 也能跑（适合脚本/CI）：**

```bash
conda run -n g2p python examples/dataset/generate_g2p_manifest_espeak.py \
  --input-csv dataset/ipa-childes-split/train/en-US/data.csv \
  --output-vocab dataset/release/en-US/vocab.txt \
  --voice en-us \
  --num-samples 100
```

### 依赖说明

`requirements.txt` 内容：

- `piper-phonemize`：进程内调用 espeak-ng，速度快、不播放音频
- `tqdm`：进度条

下载脚本额外需要 `huggingface_hub`。

---

## 1. 下载数据集

脚本：`download_ipa_childes_split.py`

数据来源：[fdemelo/ipa-childes-split](https://huggingface.co/datasets/fdemelo/ipa-childes-split)  
对应模型：[g2p-mbyt5-12l-ipa-childes-espeak](https://huggingface.co/fdemelo/g2p-mbyt5-12l-ipa-childes-espeak)  
许可证：CC-BY-4.0

### 基本用法

```bash
# 下载全部语言 train + test（约 3.58 GB）
python examples/dataset/download_ipa_childes_split.py

# 只下载 en-US
python examples/dataset/download_ipa_childes_split.py --languages en-US

# 只下载 en-US 训练集
python examples/dataset/download_ipa_childes_split.py \
  --languages en-US \
  --splits train
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `<repo_root>/dataset/ipa-childes-split` | 保存目录 |
| `--languages` | 全部 31 种语言 | 语言标签，如 `en-US zh-CN` |
| `--splits` | `train test` | 下载哪些划分 |
| `--hf-token` | 无 | HuggingFace token（公开数据集一般不需要） |

### 下载后目录结构

```
dataset/ipa-childes-split/
├── README.md
├── train/
│   └── en-US/
│       └── data.csv
└── test/
    └── en-US/
        └── data.csv
```

CSV 主要字段：

- `sentence`：原始文本（grapheme）
- `ipa_espeak`：数据集中预存的 espeak-ng 音素（仅供参考，本工具会重新生成）
- `espeak_lang_code`：espeak voice，如 `en-us`

---

## 2. 生成 G2P Manifest

脚本：`generate_g2p_manifest_espeak.py`

读取 CSV 中的原始文本，通过 `piper-phonemize` 调用 espeak-ng 生成 IPA 音素，输出 NeMo T5 G2P 格式的 JSONL manifest，并同步收集音素字符表 `vocab.txt`。

### 必填参数

| 参数 | 说明 |
|------|------|
| `--input-csv` | 输入 CSV 路径 |
| `--output-vocab` | 输出 vocab.txt 路径 |
| `--voice` | espeak-ng voice，如 `en-us` |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-json` | `<output-vocab 同目录>/train.json` | 输出 JSONL 路径 |
| `--text-field` | `sentence` | CSV 中文本列名 |
| `--num-samples` | `-1` | 处理条数，`-1` 表示全量 |
| `--num-workers` | 自动（`os.cpu_count()`） | 并行进程数，不传则按 CPU 核数 |
| `--batch-size` | `2048` | 每个 worker 任务的句子数 |
| `--resume` / `--no-resume` | 默认 `--resume` | 若 `train.json` 已存在则断点续传 |

### 断点续传

默认开启 `--resume`：

- 统计已有 `train.json` 行数，跳过 CSV 里对应条数，**追加**写入
- 词表从 `vocab.vocab_cache` 快速恢复（每批 flush 时更新），跑完后写 `vocab.txt` 并删除缓存
- 中断后直接**再跑同一条命令**即可继续，无需改参数

强制从头覆盖：

```bash
python examples/dataset/generate_g2p_manifest_espeak.py \
  --input-csv dataset/ipa-childes-split/train/en-US/data.csv \
  --output-vocab dataset/release/en-US/vocab.txt \
  --voice en-us \
  --no-resume
```

### 速度说明

脚本已做以下优化（230 万条场景）：

- `piper-phonemize` 进程内 C++ 调用，无子进程、无音频
- 多进程 `ProcessPoolExecutor`，默认 CPU 核数
- 默认 `batch-size=2048`，减少进程间通信
- `csv.reader` 按列索引读，避免 `DictReader` 开销
- 批量写 JSON（256 行缓冲），减少磁盘 I/O
- 不再为全量数据预先扫一遍 CSV 计数（去掉 `count_rows` 全表遍历）

下载脚本 `download_ipa_childes_split.py` 使用 HuggingFace `snapshot_download`，**已自带断点续传**（本地 cache 命中已下载文件）。

### 最简用法

```bash
python examples/dataset/generate_g2p_manifest_espeak.py \
  --input-csv dataset/ipa-childes-split/train/en-US/data.csv \
  --output-vocab dataset/release/en-US/vocab.txt \
  --voice en-us
```

自动生成：`dataset/release/en-US/train.json`

### 试跑 100 条

```bash
python examples/dataset/generate_g2p_manifest_espeak.py \
  --input-csv dataset/ipa-childes-split/train/en-US/data.csv \
  --output-vocab dataset/release/en-US/vocab.txt \
  --output-json dataset/release/en-US/train.json \
  --voice en-us \
  --num-samples 100
```

### 全量处理（约 230 万条）

`--num-workers` 无需指定，默认自动使用 CPU 核数并行：

```bash
python examples/dataset/generate_g2p_manifest_espeak.py \
  --input-csv dataset/ipa-childes-split/train/en-US/data.csv \
  --output-vocab dataset/release/en-US/vocab.txt \
  --output-json dataset/release/en-US/train.json \
  --voice en-us
```

如需手动限制并行度（例如留核给系统），再显式传入：

```bash
python examples/dataset/generate_g2p_manifest_espeak.py \
  --input-csv dataset/ipa-childes-split/train/en-US/data.csv \
  --output-vocab dataset/release/en-US/vocab.txt \
  --output-json dataset/release/en-US/train.json \
  --voice en-us \
  --num-workers 8
```

---

## 输出格式

### train.json（JSONL，每行一条）

```json
{"text_graphemes": "just like your book at home", "text": "dʒˈʌst lˈaɪk jʊɹ bˈʊk æt hˈoʊm"}
```

字段说明：

- `text_graphemes`：输入 grapheme 文本
- `text`：espeak-ng 生成的 IPA；**`generate_g2p_manifest_espeak.py` 常为连续 IPA 字符串**；**`preprocess_ipa_childes_split.py` 则为空格分隔的 IPA 单元**（与 `examples/tts/g2p/conf/g2p_conformer_ctc.yaml` 中的 `ipa_symbol` tokenizer 一致）

可直接用于 NeMo G2P 的 `train_manifest` 等（例如 T5 用 `examples/tts/g2p/conf/g2p_t5.yaml`，Conformer-CTC 用 `examples/tts/g2p/conf/g2p_conformer_ctc.yaml`）。

### vocab.txt

对 **`generate_g2p_manifest_espeak.py`**：`--output-vocab` 从 manifest 每条记录的 **`text_graphemes`** 与 **`text`** 收集全部出现过的字符。

若使用 **`preprocess_ipa_childes_split.py`**，词表由脚本单独统计并写出，格式见上文「推荐流程」：`vocab.txt` 为音素单元与字符合集，行首为 `<pad>` / `<unk>`，**不含**下文示例中的 `<bos>` / `<eos>` 固定行。

下文「固定头部 + 字符表」示意仅针对 **`generate_g2p_manifest_espeak.py`** 产物：

文件格式（固定头部 + 数据里出现过的 IPA 字符）：

```
<pad>
<unk>
<bos>
<eos>
 
ʒ
ˈ
ʌ
...
```

- 前 4 行：NeMo 特殊 token
- 第 5 行：空格（单独一行）
- 之后：本次数据两个字段里出现过的所有字符（字母、`_`、`'`、IPA 等）

---

## G2P 训练 checkpoint 与 manifest 后处理

### 将 Lightning `.ckpt` 导出为 NeMo `.nemo`

训练得到的 `.ckpt` 不是 NeMo 推理入口直接使用的 `.nemo` 包。在**已安装 NeMo** 的环境中，于仓库根目录执行（按需修改 `CKPT`、`OUT_NEMO`）：

```bash
CKPT="exp/g2p_en_us_260w/conformer_ctc_en_us_260w/2026-06-12_09-48-39/checkpoints/conformer_ctc_en_us_260w--val_per=0.0008-epoch=64.ckpt" \
OUT_NEMO="exp/g2p_en_us_260w/conformer_ctc_en_us_260w/2026-06-12_09-48-39/conformer_ctc_en_us_260w--val_per=0.0008-epoch=64.nemo" \
python -c "
import os
from nemo.collections.tts.g2p.models.ctc import CTCG2PModel
ckpt = os.environ['CKPT']
out = os.environ['OUT_NEMO']
model = CTCG2PModel.load_from_checkpoint(ckpt, map_location='cpu')
model.save_to(out)
print('saved:', out)
"
```

若希望单行内联路径（不设环境变量），等价写法为：

```bash
python -c "
from nemo.collections.tts.g2p.models.ctc import CTCG2PModel
m = CTCG2PModel.load_from_checkpoint(
    'exp/g2p_en_us_260w/conformer_ctc_en_us_260w/2026-06-12_09-48-39/checkpoints/conformer_ctc_en_us_260w--val_per=0.0008-epoch=64.ckpt',
    map_location='cpu',
)
m.save_to('exp/g2p_en_us_260w/conformer_ctc_en_us_260w/2026-06-12_09-48-39/conformer_ctc_en_us_260w--val_per=0.0008-epoch=64.nemo')
"
```

导出后的 `.nemo` 可用于 `examples/tts/g2p/g2p_inference.py` 的 `pretrained_model=` 参数（见该脚本文件头用法）。

### 导出 ONNX（Conformer CTC G2P，`.nemo` 与 `.ckpt`）

`CTCG2PModel` 实现 `Exportable`，加载权重后调用 `model.export(...)` 即可得到 ONNX。需要环境已安装 NeMo，并具备 ONNX 相关依赖（常见为 `pip install onnx`；导出时若缺包按报错补装即可）。

**说明：** 同一次训练若 `.ckpt` 与 `.nemo` 只是同一权重的两种存储形式，**任选一种加载再导出即可**，得到的部署图应一致。只有在你想对比两种加载路径是否一致时，才需要各导出一个 ONNX（例如不同后缀）。

下面假设 checkpoint 与 `.nemo` 均在同一实验目录下（按你的实际文件名修改 `NEMO`、`CKPT`）：

```bash
DIR="exp/g2p_en_us_260w/conformer_ctc_en_us_260w/2026-06-12_09-48-39"
CKPT="${DIR}/checkpoints/conformer_ctc_en_us_260w--val_per=0.0008-epoch=64.ckpt"
NEMO="${DIR}/conformer_ctc_en_us_260w--val_per=0.0008-epoch=64.nemo"
OUT_FROM_NEMO="${DIR}/conformer_ctc_en_us_260w_from_nemo.onnx"
OUT_FROM_CKPT="${DIR}/conformer_ctc_en_us_260w_from_ckpt.onnx"

# 1) 从 .nemo 导出
NEMO="$NEMO" OUT_ONNX="$OUT_FROM_NEMO" python -c "
import os
from nemo.collections.tts.g2p.models.ctc import CTCG2PModel
path = os.environ['NEMO']
out = os.environ['OUT_ONNX']
m = CTCG2PModel.restore_from(path, map_location='cpu')
m.eval()
m.export(out)
print('saved:', out)
"

# 2) 从 .ckpt 导出（与上一步二选一即可，不必两条都跑）
CKPT="$CKPT" OUT_ONNX="$OUT_FROM_CKPT" python -c "
import os
from nemo.collections.tts.g2p.models.ctc import CTCG2PModel
path = os.environ['CKPT']
out = os.environ['OUT_ONNX']
m = CTCG2PModel.load_from_checkpoint(path, map_location='cpu')
m.eval()
m.export(out)
print('saved:', out)
"
```

有 GPU 时可将 `map_location='cpu'` 改为 `'cuda:0'` 并把模型 `m.to('cuda')` 后再 `export`（部分算子 GPU 上追踪更省事，视环境而定）。

ONNX 的输入/输出与 `nemo/collections/tts/g2p/models/ctc.py` 中 `forward_for_export` 一致：**输入** `input_ids`、`input_len`；**输出** `log_probs`、`encoded_len`。部署侧仍需用与训练一致的 grapheme tokenizer 将文本编成 `input_ids`，并用与原模型相同的 CTC 解码（词表在 `.nemo` / checkpoint 内）将 `log_probs` 转为音素串。

### 从 manifest 提取 grapheme 文本行（`extract_graphemes_from_manifest.py`）

脚本路径：`examples/dataset/extract_graphemes_from_manifest.py`。**不加载任何模型**，只读本地 **JSONL**（一行一个 JSON 对象），从每行取出 `text_graphemes`，写入 **UTF-8 纯文本**（与上文的 `.ckpt` / `.nemo` 无关）。

#### 输入需要满足什么

- **文件**：`--input` 指向的 JSONL；编码按实现以 **二进制** 读取，约定为 **UTF-8**（与本目录 `generate_g2p_manifest_espeak.py` 写出的 `train.json` 一致即可）。
- **每行**：应能解析出非空的 `text_graphemes`（取前会做 `strip`，全空则**跳过该行不写输出**）。
- **快速路径（默认优先）**：若一行以字节序列 `{"text_graphemes": "` 开头，且其后能匹配到**第一次**出现的 `", "text":`，则把这两段之间的内容当作 grapheme 串写出（**不做 JSON 字符串转义解码**，与 NeMo 常见「`text_graphemes` 在前、`text` 在后」的 manifest 一致时最快）。若行格式不满足（例如字段顺序相反、或没有紧跟的 `", "text":`），则走 **回退路径**：整行 `json.loads`（若安装了 `orjson` 则优先 `orjson.loads`），再取 `text_graphemes` 键。
- **不会写入输出行的情况**：取不到 `text_graphemes`、值为空、或解析失败（静默跳过）。

#### 输出是什么

- **文件**：`--output` 路径；父目录不存在会自动创建。
- **内容**：每个成功提取的 `text_graphemes` **占一行**；行与行之间为换行符 `\n`（实现里用 `\n` 拼接写入）。
- **结束**：脚本在 stdout 打印 `Wrote <条数> grapheme lines to <output> (<workers> worker(s))`。

#### 命令行参数（与 `parse_args()` 一致）

| 参数 | 必填 | 默认 | 含义 |
|------|------|------|------|
| `--input` | 是 | — | 输入 JSONL manifest 路径 |
| `--output` | 是 | — | 输出 `.txt`（或其它后缀）路径 |
| `--workers` | 否 | `0` | 并行进程数；`0` 表示用 **`os.cpu_count()`**；**`1`** 表示单进程、不分片 |
| `--progress` | 否 | 关 | 显示进度条（需安装 **`tqdm`**） |
| `--write-buffer-lines` | 否 | `8192` | 每个 worker 累计多少行再 flush 一次（调 I/O，一般不用改） |

#### 示例命令（仓库根目录）

与本 README 前文生成的 manifest 一致时：

```bash
python examples/dataset/extract_graphemes_from_manifest.py \
  --input dataset/release/en-US/train.json \
  --output dataset/release/en-US/graphemes_from_manifest.txt \
  --progress
```

强制单进程（便于调试或避免占满 CPU）：

```bash
python examples/dataset/extract_graphemes_from_manifest.py \
  --input dataset/release/en-US/train.json \
  --output dataset/release/en-US/graphemes_from_manifest.txt \
  --workers 1 \
  --progress
```

---

## 完整示例（en-US）

在仓库根目录依次执行：

```bash
# 0. 激活环境（若 conda activate 报错，先 source 再 activate，见上文）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate g2p

# 1. 安装依赖（首次）
pip install -r examples/dataset/requirements.txt
pip install "huggingface_hub>=0.24"

# 2. 下载 en-US 训练数据（国内临时镜像，单行生效）
HF_ENDPOINT=https://hf-mirror.com python examples/dataset/download_ipa_childes_split.py \
  --languages en-US \
  --splits train

# 3. 先试 100 条
python examples/dataset/generate_g2p_manifest_espeak.py \
  --input-csv dataset/ipa-childes-split/train/en-US/data.csv \
  --output-vocab dataset/release/en-US/vocab.txt \
  --output-json dataset/release/en-US/train.json \
  --voice en-us \
  --num-samples 100

# 4. 确认无误后跑全量
python examples/dataset/generate_g2p_manifest_espeak.py \
  --input-csv dataset/ipa-childes-split/train/en-US/data.csv \
  --output-vocab dataset/release/en-US/vocab.txt \
  --output-json dataset/release/en-US/train.json \
  --voice en-us
```

最终产物：

```
dataset/release/en-US/
├── train.json    # G2P manifest
└── vocab.txt     # 从 text_graphemes + text 收集的词表
```

---

## 说明

- 使用 `piper-phonemize` 而非 `espeak-ng` 命令行：避免每条数据 fork 子进程，230 万条数据速度差距极大；且不会播放音频。
- `piper-phonemize` 自带 espeak-ng 数据，**不需要**单独安装 `espeak-ng` 命令行工具。
- 多进程并行基于 `ProcessPoolExecutor`（espeak-ng 非线程安全），`--num-workers` 默认等于 CPU 核数。
- 数据文件默认保存在仓库根目录的 `dataset/` 下，建议加入 `.gitignore`，不要提交到 git。
