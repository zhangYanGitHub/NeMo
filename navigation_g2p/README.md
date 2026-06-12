# Navigation G2P Dataset Generator

Production-grade pipeline for generating navigation-domain grapheme-to-phoneme (G2P) training data. Output format is JSONL compatible with NeMo TTS G2P fine-tuning (`text_graphemes` / `text` fields).

## Project Goal

Generate 10K–30K high-quality **navigation text → IPA** pairs for training a lightweight on-device G2P model that can replace espeak-ng IPA output in navigation scenarios.

Supported dataset profiles:

| Profile | Train | Val | Test | Purpose |
|---------|------:|----:|-----:|---------|
| `mini_seed` | 50 | 10 | 10 | Smoke test |
| `nav_baseline_v1` | 2,000 | 200 | 200 | Development baseline |
| `nav_prod_v1` | 20,000 | 1,000 | 1,000 | Production training set |
| `nav_entities_v1` | — | — | 1,000 | Long-tail entity test set |
| `nav_replay_v1` | — | — | 400 | Business-style replay test set |

## Directory Structure

```
navigation_g2p/
├── README.md
├── generate_navigation_g2p_dataset.py   # CLI entry point
├── config.py                            # Runtime defaults
├── dataset_profiles.py                  # Profile sizes & quotas
├── templates.py                         # Template registry by category
├── replay_templates.py                  # Business-style replay templates
├── slot_values.py                       # Slot vocabularies & builders
├── entity_sets.py                       # Curated long-tail entity pools
├── ipa_generator.py                     # Pluggable IPA backends
├── normalizer.py                        # Grapheme/IPA normalization
├── dedup.py                             # Dedup keys, canonical entity, generation caps
├── splitter.py                          # Entity-aware train/val/test split
├── validator.py                         # Quality gate, dedup audit, deficits
└── utils.py                             # Shared helpers
```

## Requirements

- Python 3.10+
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) installed and on `PATH`

No external corpus downloads required — all templates and slot values are embedded.

## Install espeak-ng

**macOS (Homebrew):**
```bash
brew install espeak-ng
```

**Ubuntu/Debian:**
```bash
sudo apt-get install espeak-ng
```

**Verify:**
```bash
espeak-ng --version
espeak-ng -v en-us --ipa=3 -q "turn right onto Main Street"
```

## Quick Start

```bash
cd navigation_g2p

# Smoke test (~70 samples, fast)
python generate_navigation_g2p_dataset.py \
  --dataset-profile mini_seed \
  --output-dir output \
  --seed 42 \
  --preview 3
```

## Run Each Dataset Profile

### mini_seed (smoke test)
```bash
python generate_navigation_g2p_dataset.py --dataset-profile mini_seed --output-dir output --seed 42
```

### nav_baseline_v1 (development)
```bash
python generate_navigation_g2p_dataset.py --dataset-profile nav_baseline_v1 --output-dir output --seed 42
```

### nav_prod_v1 (production — ~22K samples)
```bash
# Fastest (recommended): parallel IPA + skip refinement rounds
python generate_navigation_g2p_dataset.py \
  --dataset-profile nav_prod_v1 \
  --output-dir output \
  --seed 42 \
  --fast \
  --workers 16

# Full quality gate + refinement (slower)
python generate_navigation_g2p_dataset.py \
  --dataset-profile nav_prod_v1 \
  --output-dir output \
  --seed 42 \
  --strict
```

### nav_entities_v1 (entity-focused test)
```bash
python generate_navigation_g2p_dataset.py --dataset-profile nav_entities_v1 --output-dir output --seed 42
```

### nav_replay_v1 (business replay test)
```bash
python generate_navigation_g2p_dataset.py --dataset-profile nav_replay_v1 --output-dir output --seed 42
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-profile` | `nav_prod_v1` | Dataset variant to generate |
| `--output-dir` | `output` | Root output directory |
| `--seed` | `42` | Random seed |
| `--locale` | `en-us` | espeak-ng voice locale |
| `--espeak-cmd` | `espeak-ng` | Path to espeak-ng binary |
| `--no-expand-abbrev` | off (expand on) | Keep St./Ave./Blvd. abbreviations instead of full words |
| `--no-spoken-numbers` | off (spoken on) | Keep Arabic digits instead of English number words |
| `--dedupe` / `--no-dedupe` | on | Remove duplicate grapheme strings |
| `--max-text-len` | `300` | Max grapheme length |
| `--max-ipa-len` | `500` | Max IPA length |
| `--preview` | `0` | Print first N samples to stdout |
| `--save-metadata` | on | Write `*_with_meta.jsonl` files |
| `--strict` / `--no-strict` | profile default | Exit 1 if quality gate fails (`nav_prod_v1` default: strict) |
| `--workers` | auto (`cpu×2`) | Parallel espeak-ng threads — **main speed lever** |
| `--ipa-batch-size` | `256` | Graphemes per parallel IPA batch |
| `--fast` | off | Skip refinement, oversample 1.08×, espeak retries 0 |

## Output Files

Each profile writes to `output/<profile_name>/`:

```
output/nav_prod_v1/
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── train_with_meta.jsonl
├── val_with_meta.jsonl
├── test_with_meta.jsonl
└── summary.json
```

### JSONL format

**train.jsonl** (minimal):
```json
{"text_graphemes": "In 300 meters, turn right onto Main Street", "text": "ɪn θɹiː hʌndɹɪd mˈiːtɚz tˈɜːn ɹˈaɪt ˌɑːntʊ mˈeɪn stɹˈiːt"}
```

**train_with_meta.jsonl** (extended):
```json
{
  "text_graphemes": "In 300 meters, turn right onto Main Street",
  "text": "...",
  "template_type": "distance_prefixed",
  "template_id": "dp_002",
  "slots": {"distance_phrase": "300 meters", "direction": "right", "road_name": "Main Street"},
  "split": "train",
  "source_generator": "template_v1"
}
```

IPA is written as UTF-8 directly (not `\uXXXX` escaped).

## Extend Templates

1. Open `templates.py`
2. Add a `Template` to the appropriate category list (e.g. `DISTANCE_PREFIXED`)
3. Use `{slot_name}` placeholders matching entries in `slot_values.py`
4. Set `template_type`, `length_hint` (`short`/`medium`/`long`), and optional `tags`
5. Update quotas in `dataset_profiles.py` if adding a new category

Example:
```python
_t("dp_018", "distance_prefixed",
   "In {distance_phrase}, bear {direction} onto {road_name}",
   ("distance_phrase", "direction", "road_name"))
```

For business-style replay sentences, add templates to `replay_templates.py`.

## Extend Slot Vocabularies

1. Open `slot_values.py`
2. Append values to the relevant list (e.g. `POI_NAME`, `ROAD_CORE_NAME`)
3. For composite slots, update builder functions (`build_road_name`, `build_route_name`, etc.)
4. For entity-test long-tail coverage, also update `entity_sets.py`

## Replace IPA Backend

`ipa_generator.py` defines a pluggable interface:

```python
from ipa_generator import BaseIpaGenerator, create_ipa_generator, DictionaryBackedIpaGenerator

# Default
gen = create_ipa_generator(backend="espeak", espeak_cmd="espeak-ng", locale="en-us")

# Custom lexicon with espeak fallback
gen = DictionaryBackedIpaGenerator(lexicon={"Main St.": "mˈeɪn stɹˈiːt"}, fallback=gen)

# Future neural model (stub)
from ipa_generator import NeuralIpaGenerator
gen = NeuralIpaGenerator(model_path="/path/to/model", fallback=espeak_gen)
```

Wire your backend in `generate_navigation_g2p_dataset.py` → `run()` where `create_ipa_generator` is called.

## Understanding summary.json

`summary.json` is the **training acceptance report**. Key fields:

| Field | Meaning |
|-------|---------|
| `quality_gate` | `{passed, failures[], warnings[]}` — hard/soft rules |
| `dedup_audit` | exact/near-dup rates, effective_unique_ratio, top skeletons |
| `entity_coverage` | Per-split unique canonical entities by slot type |
| `difficulty_balance` | long_sentence_ratio, entity_heavy_ratio, long_tail_ratio per split |
| `deficits` | Remaining gaps with remediation actions |
| `recommended_actions` | Human-readable补样建议 |
| `conflict_sample_count` | Samples with cross-split entity votes (written to `conflict.jsonl`) |
| `leakage.weighted_leakage_ratio` | Weighted entity leakage (address×3, poi×2, road×2) |
| `total_generated` / `total_unique` | Pool size before/after exact dedupe |
| `template_distribution_by_split` | Per-split template balance |

Example `quality_gate` snippet:

```json
{
  "quality_gate": {
    "passed": true,
    "failures": [],
    "warnings": ["WARN_03 basic_actions_share 0.16 > 0.15"]
  },
  "dedup_audit": {
    "exact_duplicate_rate": 0.0,
    "near_duplicate_rate_slot_skeleton": 0.041,
    "effective_unique_ratio": 0.62,
    "per_template_id_max_share": 0.028
  },
  "entity_coverage": {
    "train": {"road_name": 342, "poi_name": 291, "route_name": 256}
  },
  "difficulty_balance": {
    "train": {"long_sentence_ratio": 0.19, "entity_heavy_ratio": 0.31, "long_tail_ratio": 0.13}
  },
  "deficits": [],
  "recommended_actions": []
}
```

## Quality Control (Production Gate)

### Dedup layers (`dedup.py`)

| Key | Generation intercept | Validator audit | Hard fail |
|-----|---------------------|-----------------|-----------|
| `exact_key` | yes | yes | exact_duplicate_rate ≤ 0.1% |
| `normalized_key` | yes | yes | — |
| `slot_skeleton_key` | yes (cap=1) | yes | near_dup_slot ≤ 8% |
| `numeric_skeleton_key` | yes (cap=2 per entity) | yes | — |
| `canonical_entity` | via caps | entity_coverage | unique_* mins |

### nav_prod_v1 hard thresholds (`dataset_profiles.py` → `QualityThresholds`)

| Metric | Threshold |
|--------|-----------|
| effective_unique_ratio | ≥ 0.55 |
| weighted_entity_leakage_ratio | ≤ 0.02 |
| unique_road_name (train) | ≥ 320 |
| unique_poi_name (train) | ≥ 280 |
| unique_route_name (train) | ≥ 240 |
| long_sentence_ratio (train) | 15%–25% |
| entity_heavy_ratio (train) | 25%–40% |
| max_template_id_share | ≤ 3% |

### Automatic refinement loop

If quality gate fails, the generator runs up to 3 refinement rounds:

1. `validator.py` computes `deficits`
2. `DatasetGenerator.set_deficits()` adjusts template/entity targeting
3. Generates an extra 5% batch and re-validates
4. `--strict` (default for `nav_prod_v1`) exits 1 if still failing

### Split safety (`splitter.py`)

- Entity assignment uses `canonical_entity()` (abbrev/compass insensitive)
- **Never borrows** samples across train/val/test to fill deficits
- Entity-conflict samples → `conflict.jsonl` (excluded from train)
- Trim uses stratified ranking (keep rare slot_skeletons, balance templates)

## Generation Strategy

- **Template quotas**: Each `dataset_profile` defines target fractions per `template_type`; the generator biases underrepresented categories.
- **Entity coverage**: Minimum unique entities per slot (`road_name`, `poi_name`, etc.) are tracked and boosted until quotas are met.
- **Dedup**: Five-layer dedup via `DedupTracker` — exact, normalized, slot-skeleton, numeric-scoped, per-entity caps.
- **Length diversity**: Short / medium / long sentence ratios are controlled via `length_hint` and `config.length_distribution`.
- **Long-tail boost**: `nav_entities_v1` uses `entity_sets.py` and higher `long_tail_boost` for difficult names.
- **Entity-aware split**: `splitter.py` hashes canonical entity keys to splits; conflicts isolated.

## License

Internal tooling for NeMo navigation G2P dataset preparation.
