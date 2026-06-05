# G2P 数据集工具

本目录提供从 HuggingFace 下载 IPA-CHILDES 数据集、并用 `piper-phonemize` 批量生成 NeMo G2P 训练 manifest 的脚本。

整体流程：

```
下载 CSV 原始数据  →  批量 G2P 音素化  →  train.json + vocab.txt
```

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
  --output-vocab dataset/normal/en-US/vocab.txt \
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
  --output-vocab dataset/normal/en-US/vocab.txt \
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
  --output-vocab dataset/normal/en-US/vocab.txt \
  --voice en-us
```

自动生成：`dataset/normal/en-US/train.json`

### 试跑 100 条

```bash
python examples/dataset/generate_g2p_manifest_espeak.py \
  --input-csv dataset/ipa-childes-split/train/en-US/data.csv \
  --output-vocab dataset/normal/en-US/vocab.txt \
  --output-json dataset/normal/en-US/train.json \
  --voice en-us \
  --num-samples 100
```

### 全量处理（约 230 万条）

`--num-workers` 无需指定，默认自动使用 CPU 核数并行：

```bash
python examples/dataset/generate_g2p_manifest_espeak.py \
  --input-csv dataset/ipa-childes-split/train/en-US/data.csv \
  --output-vocab dataset/normal/en-US/vocab.txt \
  --output-json dataset/normal/en-US/train.json \
  --voice en-us
```

如需手动限制并行度（例如留核给系统），再显式传入：

```bash
python examples/dataset/generate_g2p_manifest_espeak.py \
  --input-csv dataset/ipa-childes-split/train/en-US/data.csv \
  --output-vocab dataset/normal/en-US/vocab.txt \
  --output-json dataset/normal/en-US/train.json \
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
- `text`：espeak-ng 生成的 IPA 音素串

可直接用于 NeMo G2P 训练配置（如 `examples/tts/g2p/conf/g2p_t5.yaml`）中的 `train_manifest`。

### vocab.txt

从 manifest 每条记录的 **`text_graphemes`** 和 **`text`** 两个字段收集**全部**出现过的字符，写入 `--output-vocab`。

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
  --output-vocab dataset/normal/en-US/vocab.txt \
  --output-json dataset/normal/en-US/train.json \
  --voice en-us \
  --num-samples 100

# 4. 确认无误后跑全量
python examples/dataset/generate_g2p_manifest_espeak.py \
  --input-csv dataset/ipa-childes-split/train/en-US/data.csv \
  --output-vocab dataset/normal/en-US/vocab.txt \
  --output-json dataset/normal/en-US/train.json \
  --voice en-us
```

最终产物：

```
dataset/normal/en-US/
├── train.json    # G2P manifest
└── vocab.txt     # 从 text_graphemes + text 收集的词表
```

---

## 说明

- 使用 `piper-phonemize` 而非 `espeak-ng` 命令行：避免每条数据 fork 子进程，230 万条数据速度差距极大；且不会播放音频。
- `piper-phonemize` 自带 espeak-ng 数据，**不需要**单独安装 `espeak-ng` 命令行工具。
- 多进程并行基于 `ProcessPoolExecutor`（espeak-ng 非线程安全），`--num-workers` 默认等于 CPU 核数。
- 数据文件默认保存在仓库根目录的 `dataset/` 下，建议加入 `.gitignore`，不要提交到 git。
