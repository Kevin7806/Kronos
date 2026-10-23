# Kronos on 1 minute ES: local execution, fine-tuning and weekly OOS runbook

**Prepared:** 2026-10-23 · **Status:** runbook and implementation contract; no ES performance results claimed.

**Goal:** run Kronos locally, adapt it to your 1 minute ES data, and evaluate a frozen model on each next OOS week using the existing rolling 24-month framework. Start by measuring forecast quality; position-management experiments follow only if forecast evidence warrants them.

**Recommended first experiment:** Kronos-small with the original tokenizer frozen, 256 completed one-minute bars of context, 15 future one-minute bars, and forecasts scored at completed five-minute boundaries. Compare the untouched checkpoint with a newly fine-tuned copy in every fold, on exactly the same observations. The resolution of the model input remains **one minute**.

This document distinguishes three levels of readiness:

| Label | Meaning |
|---|---|
| **Available now** | Existing upstream interfaces or complete example commands/snippets included here |
| **Engineering smoke only** | Can exercise loading/training, but its results do not satisfy the research protocol |
| **Adapter required** | Proposed interface/behavior to implement before scoring the weekly experiment; not a command that exists today |

The repository can already load weights, generate forecasts and train on CSVs. It does **not** currently contain the exact ES weekly runner specified here. Several upstream training behaviors must be corrected before its validation loss or forecasts are treated as research evidence. Sections 4–6 provide immediate execution steps; sections 7–12 specify that integration precisely.

## 1. Reviewed sources and what this experiment changes

The source review is pinned to:

| Source | Revision / purpose |
|---|---|
| [Kronos paper](https://arxiv.org/html/2508.02739v1) | arXiv:2508.02739v1; methodology and reported pretraining cutoff |
| [Your Kronos fork](https://github.com/Kevin7806/Kronos/tree/67b630e67f6a18c9e9be918d9b4337c960db1e9a) | `master`, `67b630e67f6a18c9e9be918d9b4337c960db1e9a` |
| [Quant Discovery Lab](https://github.com/Kevin7806/quant-discovery-lab/tree/ed86575aa1eddcc8c712a2f3c8746a05d0a8235b) | `strategy/nexus-matrix`, `ed86575aa1eddcc8c712a2f3c8746a05d0a8235b` |
| [Existing Kronos idea](https://github.com/Kevin7806/quant-discovery-lab/blob/ed86575aa1eddcc8c712a2f3c8746a05d0a8235b/docs/research/ideas/KRONOS_ES_RISK_AND_POSITION_MANAGEMENT_IDEA.md) | PR #46, merged; external-pretraining track, path retention, TRUELIGHT comparison |

Kronos represents OHLC, volume and amount as hierarchical tokens, then generates future tokens autoregressively. The paper's pretraining extends through **June 2024**. Its published investment simulation concerns daily equity portfolios; its forecasting tables do not establish a one-minute ES edge. Its volatility benchmark uses close-return variance, which differs from a Garman–Klass label. These facts motivate the ES-specific experiment below, rather than supplying its result. [Paper, §3 and Appendix D]

The existing idea proposed five-minute inputs. **Your current request changes the primary input to one minute.** Retain a five-minute scoring grid so a 15-minute endpoint comparison can later align with TRUELIGHT. Treat a separately aggregated five-minute Kronos arm as an optional, registered comparison, not an automatic replacement for the requested one-minute experiment.

### 1.1 Model choice

| Model | Tokenizer | Published context limit | Role here |
|---|---|---:|---|
| Kronos-small, 24.7M parameters | `NeoQuasar/Kronos-Tokenizer-base` | 512 bars | First experiment |
| Kronos-base, 102.3M parameters | Same base tokenizer | 512 bars | Later size comparison if small passes engineering checks |
| Kronos-mini, 4.1M parameters | `NeoQuasar/Kronos-Tokenizer-2k` | 2,048 bars | Separate smaller-model comparison; not interchangeable with the base tokenizer |

Source: [model card](https://huggingface.co/NeoQuasar/Kronos-small). The starting context of 256 minutes is a research choice, not a published ES optimum. Do not increase `max_context` on small/base to claim a supported longer context.

Your RTX 3060 is a reasonable machine on which to start the small-model feasibility test. Actual VRAM, driver compatibility, throughput and fine-tuning time must be measured. No full-run duration or memory-fit guarantee is made here. Use your local server for all model inference and training; Hugging Face is used only to obtain pinned artifacts.

### 1.2 Pretraining creates two different research tracks

| Track | Initialization every fold | Valid interpretation |
|---|---|---|
| **P, primary here** | Reload the same pinned pretrained tokenizer and predictor; fine-tune only on that fold's permitted ES history | External-pretraining transfer experiment using a 24-month ES adaptation window |
| **F, separate later experiment** | Train tokenizer and predictor from random initialization inside each fold | Architecture trained only on fold data; a different experiment with a different compute budget |

Fine-tuning cannot erase information already present in pretrained weights. Track P cannot support a clean historical OOS claim for 2020–June 2024. Verify the exact model **and tokenizer** training/adaptation cutoff; the paper alone is not a provenance certificate for every downloadable revision. Unknown cutoff means `PROVENANCE_UNVERIFIED`, not clean OOS.

Every origin used for pretrained-model checkpoint selection, probability calibration, blending, or scoring must be later than the verified external-training cutoff. Earlier rows may be ordinary supervised ES **training** examples; they may not masquerade as held-out validation predictions. To claim the system was historically deployable, also establish that the exact artifacts were available before the fold's preparation date. Post-cutoff retrospective evaluation and historical deployability are different claims.

The existing Nexus fresh-fit-only rule is stricter than Track P. This runbook documents the external-pretraining exception **only for this separate Kronos research experiment**, as requested. It does not change the initialization rules for LENS, TRUELIGHT or COMPASS, or promote Kronos into Nexus. Always disclose the unequal total training-data budgets when comparing pretrained and fold-only models.

## 2. Freeze the weekly experiment before running it

### 2.1 Default specification

| Setting | Initial value / rule |
|---|---|
| Asset / source | Continuous ES features, mapped to the OOS active-contract price basis |
| Input resolution | 1 minute OHLCVA |
| Historical input | 256 consecutive, completed one-minute bars |
| Forecast | 15 consecutive future one-minute bars |
| Secondary endpoints | 1 and 5 minutes, taken from those same 15-minute paths |
| Optional later horizon | 60 minutes; separate run and coverage accounting |
| Scored origins | Five-minute completed-bar boundaries, RTH 08:35–14:45 America/Chicago, eligible full sessions |
| Primary event | Future 15-minute close minus current completed close ≤ −5 ES points |
| Primary fine-tuning comparison | Fine-tuned Kronos versus untouched Kronos, paired by origin |
| Required simple references | Persistence for point errors; TRAIN-only empirical endpoint distribution/event rate |
| Training context sessions | All available, eligible weekday exchange-open minutes; report actual RTH/extended-hours coverage |
| Evaluation sampling | 256 paths/origin; `T=1.0`, `top_k=0`, `top_p=1.0` |
| GPU sampling chunk | Start at 8 paths, measure and freeze before scoring |
| Seeds | Start with 42; derive stable per-origin/per-path seeds from a cryptographic hash |
| Weekly initialization | Same frozen original weights; no warm-start from the preceding fine-tuned week |
| Online fitting | None during an OOS week; only the observed input context advances |

The RTH scoring cohort is an initial forecast benchmark. It does not change your broader allowance for weekday overnight positions or the no-weekend/holiday exposure rule. Extended-hours scoring is a separate reported cohort. Do not silently concatenate RTH-only days and call the gap a sequence of one-minute observations.

### 2.2 Exactly how your current 24-month framework works

Your current framework reserves the final **four weeks inside the 24-month development window** for validation. Thus “train 24 months, test one week” currently means **24 months of development history, split into parameter fitting and validation, then one OOS week**. It is not 24 full months of gradient fitting plus an extra validation month.

Reuse the existing date algebra in `nexus_matrix.calendar_es.fold_blocks`:

```text
C                = checkpoint Saturday identifying the completed OOS week
A                = C minus 7 calendar days: preparation Saturday
window_start     = A minus 24 calendar months
validation_start = A minus 28 calendar days

TRAIN    = [window_start, validation_start)
VALIDATE = [validation_start, A)
TEST     = exchange sessions Monday C−5 days through Friday C−1 day
```

All calendar calculations use America/Chicago dates, then convert boundaries to UTC. Use calendar months, not 730 days. Assign bars to exchange session dates before applying the masks. C names the week that has just finished; its training was performed at A, before that week began.

**Worked example:**

| Stage | Checkpoint `2026-02-07` | Next checkpoint `2026-02-14` |
|---|---|---|
| Preparation Saturday | 2026-01-31 | 2026-02-07 |
| Development start | 2024-01-31 | 2024-02-07 |
| Gradient fitting ends before | 2026-01-03 | 2026-01-10 |
| Validation | Jan 3–30, 2026 | Jan 10–Feb 6, 2026 |
| OOS sessions | Feb 2–6, 2026 | Feb 9–13, 2026 |

These dates are illustrative, not a claim that your files contain a complete eligible week. The next fold may legitimately use the preceding week's now-historical observations; the preceding week's frozen predictions and scores must never be rewritten.

Do **not** refit on TRAIN+VALIDATE after calibrating or selecting against VALIDATE in this first protocol. That produces a different model from the one whose calibration was assessed. If you later want gradient updates on every row of the full 24 months, register a full-window refit protocol with an independent or cross-fitted calibration procedure. Do not describe that change as the unchanged existing framework.

### 2.3 Use validation for two jobs without reusing its labels

Split the existing four-week validation block chronologically:

1. **V-select:** first two weeks. Choose the epoch/checkpoint using actual autoregressive forecasts and the frozen selection metric.
2. **V-cal:** final two weeks. With the selected predictor now frozen, fit the prespecified probability calibration map.
3. **TEST:** score once with both predictor and calibrator frozen.

Purge V-select origins whose labels reach V-cal, and V-cal origins whose labels reach A. Retain preceding context when it was observable; the boundary applies to fitted inputs and resolved targets, not an arbitrary loss of every boundary's warmup history.

If V-cal has only one class, skip learned calibration and record the prespecified identity-map fallback. A large-looking one-week result is not a replacement for enough independent weeks.

## 3. ES data contract

### 3.1 Use the current repository conventions

The current `.ai-agent/models/time-series-standards.md` specifies **left-labeled bar-open timestamps**, UTC authoritative timestamps, and additive adjustment. The reviewed legacy Nexus code still uses naive Chicago timestamps internally. Build an explicit conversion boundary; do not infer UTC versus local time from a column name or reuse older right-close assumptions.

| Canonical field | Meaning |
|---|---|
| `bar_open_time` | Timezone-aware UTC opening instant |
| `bar_close_time` | `bar_open_time + 1 minute` for a real one-minute bar |
| `available_time` | Close plus registered feed/publication latency |
| `session_id`, `contract` | Calendar session and actual futures contract |
| `open`, `high`, `low`, `close` | Fold-vintage adjusted OHLC for model input |
| `volume` | Number of contracts traded in the minute; zero only when legitimate/explicitly unavailable |
| `amount` | Explicitly supplied channel; initial convention `0.0` if true turnover is absent |
| `quality_flag`, `is_synthetic` | Preserve outages, halts, imputation and vendor repairs |
| `vintage_id`, source hashes | Reproducible transformation and source identity |

For the stock CSV interface, export `timestamps,open,high,low,close,volume,amount`. Its `timestamps` field is the **bar-open time expressed in Chicago time** for the model's clock features. Keep the authoritative UTC fields and unique IDs in a Parquet sidecar. The research adapter should create clock features from Chicago time directly, while all ordering and joins use UTC.

The reviewed one-minute store exposes this concrete mapping:

| Store column | Kronos export |
|---|---|
| `TimeStamp` | Localize using verified source metadata, preserve UTC identity, export Chicago `timestamps` |
| `Adj Open`, `Adj High`, `Adj Low`, `Adj Close` | Subtract `shift_fold`, then rename to `open,high,low,close` |
| `Volume` | `volume`; preserve contract-count units |
| No observed amount column | Add explicit `amount=0.0` |
| `Asset` | Keep as contract identity in the sidecar; do not feed as a seventh numeric channel |
| `DateTimeKey` | Audit its legacy/local versus UTC encoding; regenerate canonical UTC identity explicitly |

For a development export, first restrict to the fold's permitted history, apply the audited vintage, select a real contiguous segment, and write the seven CSV columns in the stated order. Export labels/quality/session metadata separately. Never export the entire history plus the OOS week into a file that the stock trainer will split by a percentage.

Do not pass naive timestamps through `utc=True` unless the source really is UTC. For legacy naive CT data, localize once with `America/Chicago`, rejecting unresolved ambiguous/nonexistent times. DST fall-back duplicate local clocks require source disambiguation, not deduplication by naive timestamp.

**Timing example:** at decision `10:00:00 CT`, with zero feed latency, the latest available minute is the bar opened at 09:59 and closed at 10:00. The first forecast bar opens at 10:00; the fifteenth opens at 10:14 and closes at 10:15. Never use the observed OHLC of the 10:00 bar as an input at 10:00.

For nonzero feed/model latency, record `forecast_available_at` separately. A forecast may only enter a later economic decision after that instant. Statistical labels are referenced to the declared market origin, not retroactively shifted to make a slower model look timely.

### 3.2 Fold-specific price rollback

Reuse the approved additive vintage construction:

```text
Adj_stored(t) = Raw(t) + offset_stored(contract(t))
shift_fold    = offset_stored(OOS_contract)
Adj_fold(t)   = Adj_stored(t) − shift_fold
```

Use the same scalar for all four adjusted price fields. Resolve the contract and permitted roll records at preparation cutoff A. Do not use a future observed roll price merely because the future contract's identity is known. Reconstruct from past-known roll records; an equivalent final-vintage subtraction is acceptable only with proven future-roll cancellation and provenance.

Required checks:

- `Adj_stored − Raw` is constant within each contract, tolerance 1e−6.
- On the OOS active contract, `Adj_fold == Raw`, tolerance 1e−9.
- Appending later roll records changes neither earlier reconstructed inputs nor predictions.
- Each inner fold uses its own preparation cutoff/vintage.
- Recompute percentage/log-return features on the fold vintage. They are not invariant to an additive shift.
- Keep source prices immutable. Economic fills, if later tested, use raw executable contract prices.

Existing implementations to reuse: `contract_offsets`, `fold_shift`, `apply_vintage`, `assert_offset_constancy` and `assert_oos_raw_eq` in `strategy/Nexus_Matrix/code/nexus_matrix/market_data.py`. Verify function signatures in the pinned version when wiring them.

### 3.3 Amount and volume

For the first experiment use real ES contract volume and **explicit `amount=0.0`** when turnover is unavailable. Supplying the column prevents `KronosPredictor` from silently constructing its `volume × mean(OHLC)` fallback. Use exactly the same convention in training, validation and inference.

A volume-times-price alternative is a separate ablation. It is a proxy, not observed dollar turnover or direct supply/demand. Never multiply volume by the ES $50 multiplier in one stage and omit it in another. If a proxy is used, recompute it on the fold price basis and record its definition.

### 3.4 Calendar, gaps and sample coverage

Check finite values; positive OHLC; `low <= min(open,close) <= max(open,close) <= high`; nonnegative volume; unique UTC keys; one-minute spacing; valid contracts; and declared data coverage. Quarantine bad records with reasons. Never use backward fill, interpolate from future bars, or forward-fill volume.

Segment model windows at maintenance closures, weekends, holidays, unverified outages and contract boundaries. Do not create nonexistent exchange minutes. Use the existing versioned calendar and roll-week exclusion for the primary comparable cohort; preserve eligible historical roll segments for training. Record excluded weeks rather than dropping them from the manifest.

If your dataset already contains specially approved circuit-breaker carry-forward bars, preserve the flag. The initial scored cohort excludes origins whose input or target windows touch synthetic bars; report this attrition. Testing those bars as inputs requires a declared separate policy, and they must never be counted as executable market prints. Do not mistake a true halt for a normal quiet market.

A 256-minute context cannot cover the RTH open if only today's RTH data exists. With RTH-only observations and no gaps, the first full context ends at 12:46; the first eligible five-minute grid origin is 12:50. With genuine premarket data, 08:35 can be eligible. Report that difference and compare all models on matching origins; never silently bridge the previous close to today's open.

## 4. Install and pin a local environment — available now

Run on the Linux workstation, in a new research checkout. The paths below are examples to bind once in your local configuration.

```bash
git clone https://github.com/Kevin7806/Kronos.git
cd Kronos
git checkout 67b630e67f6a18c9e9be918d9b4337c960db1e9a
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
nvidia-smi
```

One reproducible starting choice is PyTorch 2.7.1 with CUDA 12.6 wheels, supported by the official [previous-version installation instructions](https://pytorch.org/get-started/previous-versions/). Use it only with a compatible driver; a different validated wheel/driver combination must be recorded as a different environment, not silently substituted.

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install PyYAML pyarrow scipy pytest
python -m pip check
mkdir -p artifacts/engineering
python -m pip freeze > artifacts/engineering/requirements.lock.txt
python - <<'PY'
import torch
print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable: resolve before GPU benchmark'
print('device:', torch.cuda.get_device_name(0))
print('VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
from model import Kronos, KronosTokenizer, KronosPredictor
print('Kronos imports OK')
PY
```

`PyYAML` is needed by the CSV config loader but is absent from the reviewed root requirements. Qlib/Comet are not needed for the CSV route. Freeze the complete working environment after the smoke test, including Python, OS, GPU, driver and source commit. Keep it separate from existing QDL model environments if dependency requirements differ.

### 4.1 Download once and record immutable model revisions

This script resolves a revision, downloads that exact snapshot, and creates a lock file. Resolving the latest revision is acceptable only when constructing a **new** experiment lock; all retries and folds must reuse the lock and local files. Provenance remains a separate review.

```bash
python - <<'PY'
from pathlib import Path
import hashlib, json
from huggingface_hub import HfApi, snapshot_download

root = Path('pretrained').resolve()
root.mkdir(exist_ok=True)
lock = root / 'artifacts.lock.json'
if lock.exists():
    raise SystemExit('Lock already exists: reuse it; do not silently refresh revisions')
records = []
for repo in ['NeoQuasar/Kronos-Tokenizer-base', 'NeoQuasar/Kronos-small']:
    info = HfApi().model_info(repo)
    target = root / repo.split('/')[-1]
    snapshot_download(repo_id=repo, revision=info.sha,
                      local_dir=str(target),
                      allow_patterns=['*.json', '*.safetensors', 'README.md'])
    hashes = {}
    for p in sorted(target.rglob('*')):
        if p.is_file() and '.cache' not in p.parts:
            with p.open('rb') as stream:
                hashes[str(p.relative_to(target))] = hashlib.file_digest(
                    stream, 'sha256').hexdigest()
    assert list(target.glob('*.safetensors')), f'No safe weights in {target}'
    records.append({'repo': repo, 'revision': info.sha,
                    'local_path': str(target), 'sha256': hashes,
                    'training_cutoff_verified': False,
                    'training_cutoff': None, 'availability_evidence': None})
lock.write_text(json.dumps(records, indent=2))
print(lock)
PY
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
```

The false provenance flag is intentional: a successful download does not establish training dates. Verify both components before claiming clean historical performance. Use local paths in every subsequent `from_pretrained` call. [Hugging Face download documentation](https://huggingface.co/docs/huggingface_hub/v0.33.1/en/guides/download)

## 5. First local forecast — available now

First export a **development-only**, fold-vintage CSV satisfying section 3. The example below is deliberately strict: it refuses timestamps that would require inventing missing minutes. Do not use a protected or reserved OOS week to debug this script.

Save this snippet as `scripts/smoke_es.py` in the Kronos checkout. It expects an already-prepared CSV whose Chicago bar-open timestamps are unambiguous; source mapping and fold rollback must occur before this step.

```python
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model import Kronos, KronosTokenizer, KronosPredictor

ap = argparse.ArgumentParser()
ap.add_argument('--csv', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
L, H = 256, 15
cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
df = pd.read_csv(args.csv)
assert set(['timestamps'] + cols).issubset(df.columns)
assert len(df) >= L + H
df = df.iloc[:L + H].copy()
ts = pd.to_datetime(df['timestamps'])
if ts.dt.tz is None:
    ts = ts.dt.tz_localize('America/Chicago', ambiguous='raise', nonexistent='raise')
else:
    ts = ts.dt.tz_convert('America/Chicago')
utc = ts.dt.tz_convert('UTC')
assert utc.is_monotonic_increasing and not utc.duplicated().any()
assert (utc.diff().dropna() == pd.Timedelta(minutes=1)).all()
assert np.isfinite(df[cols].to_numpy()).all()
assert (df[['open','high','low','close']] > 0).all().all()
assert (df['low'] <= df[['open','close']].min(axis=1)).all()
assert (df['high'] >= df[['open','close']].max(axis=1)).all()
assert (df[['volume','amount']] >= 0).all().all()

torch.manual_seed(42)
np.random.seed(42)
tokenizer = KronosTokenizer.from_pretrained('pretrained/Kronos-Tokenizer-base')
model = Kronos.from_pretrained('pretrained/Kronos-small')
tokenizer.eval()
model.eval()
predictor = KronosPredictor(model, tokenizer, device='cuda:0', max_context=512)

# Only these 256 observed bars enter the predictor.
history = df.iloc[:L][cols].reset_index(drop=True)
x_ts = ts.iloc[:L].reset_index(drop=True)
# Calendar times only: no future OHLC enters generation.
y_ts = pd.Series(pd.date_range(x_ts.iloc[-1] + pd.Timedelta(minutes=1),
                              periods=H, freq='min'))
assert list(y_ts) == list(ts.iloc[L:L+H])
with torch.inference_mode():
    pred = predictor.predict(history, x_ts, y_ts, pred_len=H,
                             T=1.0, top_k=0, top_p=1.0,
                             sample_count=1, verbose=False)
assert pred.shape == (H, 6) and np.isfinite(pred.to_numpy()).all()
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
pred.to_csv(out, index_label='forecast_bar_open')
print('saved', out, 'shape', pred.shape)
print('Engineering forecast only; not an OOS performance estimate.')
```

Run from the repository root:

```bash
PYTHONPATH=. python scripts/smoke_es.py \
  --csv /mnt/DATA/kronos_es/engineering/es_development.csv \
  --out artifacts/engineering/smoke_forecast.csv
```

Expected output: 15 rows and six predicted channels, with future minute timestamps. Inspect generated OHLC consistency as well as finite output. Invalid candles are a model-output issue, not permission to silently repair the data. This single path tests the interface; it does not measure accuracy or calibrated uncertainty.

### 5.1 Obtain individual paths without changing upstream code

The stock function averages its sample dimension internally. Therefore `sample_count=256` returns an **average path**, not 256 paths.

For a slow but useful verification baseline, repeat `sample_count=1`, keep every result, and change the RNG seed each time:

```python
paths = []
for sample_id in range(16):  # engineering-only sample count
    torch.manual_seed(42000 + sample_id)
    paths.append(predictor.predict(
        history, x_ts, y_ts, pred_len=15, T=1.0, top_k=0, top_p=1.0,
        sample_count=1, verbose=False).to_numpy())
paths = np.stack(paths, axis=0)  # [sample=16, horizon=15, channel=6]
np.save('artifacts/engineering/smoke_paths.npy', paths)
```

Do not reset every path to the same seed. The production research adapter should retain the internal sample axis and batch paths for speed, with reproducibility verified for its declared batching policy.

## 6. Exercise fine-tuning — engineering smoke only

**First fine-tune the predictor while freezing the original tokenizer.** This isolates whether adapting the forecasting model helps. Adapting the tokenizer changes token meanings and reconstruction behavior; evaluate it as a later, paired tokenizer-plus-predictor arm.

Use only development data for this smoke run. This section exercises the current CSV trainer. Its ratio split, sequence normalization and validation implementation are unsuitable for the final OOS experiment; section 7 specifies the replacements.

### 6.1 Minimal stock YAML

Save as `finetune_csv/configs/es_1m_smoke.yaml`. Replace the three local root paths with your actual checkout/data/artifact roots. Use a CSV large enough that both its 90% and 10% portions contain more than 272 consecutive valid rows. A small hand-selected development slice is appropriate for engineering only.

```yaml
data:
  data_path: /mnt/DATA/kronos_es/engineering/es_development.csv
  lookback_window: 256
  predict_window: 15
  max_context: 512
  clip: 5.0
  train_ratio: 0.9
  val_ratio: 0.1
  test_ratio: 0.0

training:
  tokenizer_epochs: 2
  basemodel_epochs: 2
  batch_size: 4
  log_interval: 10
  num_workers: 0
  seed: 42
  tokenizer_learning_rate: 0.00002
  predictor_learning_rate: 0.00001
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1
  accumulation_steps: 1

model_paths:
  pretrained_tokenizer: /mnt/DATA/research/Kronos/pretrained/Kronos-Tokenizer-base
  pretrained_predictor: /mnt/DATA/research/Kronos/pretrained/Kronos-small
  # Despite this key's name, point it to the original frozen tokenizer.
  finetuned_tokenizer: /mnt/DATA/research/Kronos/pretrained/Kronos-Tokenizer-base
  exp_name: es_1m_engineering_smoke
  base_path: /mnt/DATA/kronos_es/artifacts/engineering
  base_save_path: ""
  tokenizer_save_name: tokenizer
  basemodel_save_name: basemodel

experiment:
  name: es_1m_engineering_smoke
  description: Development-only pipeline test; not OOS evidence
  use_comet: false
  train_tokenizer: false
  train_basemodel: true
  skip_existing: false
  pre_trained_tokenizer: true
  pre_trained_predictor: true

device:
  use_cuda: true
  device_id: 0

distributed:
  use_ddp: false
  backend: nccl
```

Run from the `finetune_csv` directory because the upstream scripts use relative import paths:

```bash
cd finetune_csv
python train_sequential.py --config configs/es_1m_smoke.yaml --skip-tokenizer
```

Expected predictor checkpoint:

```text
/mnt/DATA/kronos_es/artifacts/engineering/es_1m_engineering_smoke/basemodel/best_model/
```

`base_save_path` already contains the experiment name after config expansion; do not append the experiment name twice. Inspect actual model weights/config, training logs and exit status. `best_model` is inference state, not a full optimizer/RNG resume checkpoint.

For a post-training smoke forecast, load this local predictor directory with the **original** tokenizer and reuse section 5. Keep the untouched and fine-tuned output paths separate. Do not select a better-looking plot or describe the CSV validation loss as an OOS gain.

Use `train_sequential.py` for this smoke route. In the reviewed `finetune_base_model.py`, the standalone `main()` contains later `import json, os` statements, making `os` local throughout the function even though `os.makedirs` is called earlier. That direct entrypoint needs an import-scope fix; the sequential launcher calls the training function and bypasses this `main()` defect.

### 6.2 Optional tokenizer adaptation, after the first predictor-only experiment

For a separately identified arm:

1. Reload the original tokenizer; fine-tune it on TRAIN only.
2. Select its checkpoint within the allowed development selection procedure; never use OOS reconstruction error to select it.
3. Freeze that tokenizer and regenerate that arm's tokens.
4. Reload the original predictor and adapt it using the newly selected tokenizer.
5. Save and load the adapted tokenizer and predictor as a matched pair.

With a separate YAML containing `train_tokenizer: true` and `finetuned_tokenizer: ""`, the current engineering launcher is:

```bash
python train_sequential.py --config configs/es_1m_tokenizer_plus_predictor_smoke.yaml
```

This second YAML is a configuration to create, not a bundled file. Use a different experiment directory. After replacing the dataset and evaluation logic, the research adapter can implement the same two-phase lifecycle. Do not train a new tokenizer once on all years and reuse it in older folds.

## 7. Corrections required before research scoring

These findings refer to the pinned source, not to a trained-model performance result.

| Finding in current code | Why it matters | Required research behavior |
|---|---|---|
| `CustomKlineDataset` splits by row ratios | 90/10 is not the repository's calendar-based four-week holdout | Explicit TRAIN/V-select/V-cal date masks and target-resolution cutoffs |
| Dataset computes mean/std over the whole input-plus-target window | Changing future targets changes normalized historical inputs; mismatch with inference | Context-only normalization at each origin, applied unchanged to future target rows |
| Dataset emits overlapping windows across a flat table | Ignores the repository's tumbling-training rule and may bridge gaps/sessions/contracts | Non-overlapping training examples within valid contiguous segments; sliding inference only |
| `window = lookback + predict + 1` and loss on all shifted positions | `predict_window` is not a separately masked multi-step forecast objective | Explicit context/horizon indices and future-only loss mask |
| Example YAML uses 512+48+1 rows | Its nominal `max_context` field does not enforce a training-sequence limit | Assert actual sequence/input lengths; start with 256+15 rows |
| CSV trainer calls `.eval()` for predictor validation, while dependency cross-attention uses `is_causal=self.training` | Full-sequence validation can attend to later positions when evaluating earlier predictions | Free-running prefix-only validation; separately make any full-sequence validation attention causal |
| Sampling-based fine-token conditioning | Repeated validation loss can vary from RNG draws | Freeze validation RNG; use downstream free-running scores for selection |
| Frozen tokenizer state is not explicitly enforced in the CSV training loop | `no_grad()` does not itself set dropout/evaluation state | Explicit `tokenizer.eval()` and `requires_grad_(False)`; assert no tokenizer optimizer parameters |
| Predictor trainer does not implement gradient accumulation or mixed precision | Config keys alone cannot deliver those memory/runtime behaviors | Implement and verify them, or keep truthful FP32/batch-size reporting |
| `auto_regressive_inference` averages sample axis | Destroys the distribution needed for event probabilities, quantiles and barrier statistics | Return individual decoded paths; average only for a chosen point estimate |
| Save logic retains only the best model weights | Cannot resume an interrupted training step exactly | Save epoch/step, optimizer/scheduler, RNG and dataset cursor in a separate resume checkpoint |
| Existing `run_fold.py` still runs legacy Nexus components | Its existence does not mean Kronos is integrated | Dedicated Kronos forecast runner, sharing audited data/calendar helpers |

### 7.1 Context-only normalization: exact contract

For each training/forecast origin t, let `X` contain the L=256 known context rows and `Y` the H=15 future rows. For each channel d:

```text
mu[d]    = mean(X[:, d])
sd[d]    = population_std(X[:, d])        # ddof=0, matching predictor
Xnorm    = clip((X − mu) / (sd + 1e−5), −5, 5)
Ynorm    = clip((Y − mu) / (sd + 1e−5), −5, 5)
path_raw = path_normalized × (sd + 1e−5) + mu
```

Do not fit a second global scaler on the full CSV. Do not normalize targets with their own statistics. Inverse-transform each generated path before calculating downside, volatility or excursions. Record clipping rates by feature and split; frequent target clipping can indicate a poor representation for ES tails.

A constant-zero amount channel is valid under the numerical epsilon convention. Treat generated amount as auxiliary and apply a declared near-zero numerical tolerance; do not use a tiny decoder residual in an unavailable amount channel as a market-risk signal.

**Mutation test:** preserve all context bars, alter only future OHLC, and recompute the example. Context values, normalization statistics and context token IDs must be unchanged. The unmodified CSV class fails the context-value part of this test.

### 7.2 Training sample and loss alignment

For the corrected adapter, use exactly **L+H=271** rows per example:

```text
rows [0 ... 255]   = observed context through t
rows [256 ... 270] = future supervision through t+15 minutes
token inputs      = token rows [0 ... 269]
token targets     = token rows [1 ... 270]
loss positions    = [255 ... 269] only
```

Compute the two token-head cross-entropies on those 15 forecast positions, retaining the implementation's sampled coarse-token conditioning unless a separate ablation changes it. Teacher forcing of **earlier future bars as sequence inputs during training** is normal autoregressive supervision; using an unobserved later bar to construct a context input is not. At validation/OOS inference, generate the full horizon freely without observed future OHLC.

The tokenizer's reviewed self-attention is causal. Add a prefix-invariance test: encoding the context alone must produce the same context token IDs as encoding context plus target, with identical normalization/evaluation state. Avoid context-loss positions before t: their normalization would contain later rows relative to those earlier positions even though everything is known at the chosen origin t.

For training, build non-overlapping 271-row windows in each valid segment; freeze the segment anchor and stride. Shuffle the resulting training examples only. Never random-split windows into training and validation. Report examples, represented bars and dropped segment tails per fold. If you later adopt overlapping training windows, record that departure from current repository standards as a separate experiment.

Use chronological sliding origins for validation/OOS forecasts. Require every target to resolve inside its split. Historical context shared across a split boundary is allowed when already observable; training labels extending into validation or validation labels extending into TEST are not.

### 7.3 Causal validation attention

In `model/module.py`, `CrossAttention.forward` sets `is_causal_flag = self.training`. During a full-sequence evaluation pass, that removes the causal mask. Do not rely on its full-sequence validation cross-entropy as an honest forecasting score.

The primary checkpoint-selection path must generate from a past-only prefix, exactly as OOS inference does. If token validation loss is also reported, expose an explicit causal argument for aligned multi-position sequence evaluation. Generation with a one-position query over an entirely known historical prefix should still be able to attend to that whole prefix. Do not globally flip the model to training mode to obtain an attention mask; that would also change dropout behavior.

### 7.4 Individual sample API

Add an adapter method with the following contract; this method does not exist upstream:

```python
def predict_paths(history, future_timestamps, *, sample_count, seed, chunk_size):
    ...  # Interface to implement; not supplied by upstream.
# returns float32 [origin, sample, future_bar, OHLCVA_channel]
# plus forecast status, quality flags and provenance
```

Preserve `z.reshape(batch, samples, sequence, features)` before the upstream mean. Slice only the last H decoded bars, inverse-transform with the corresponding origin's context statistics, and persist individual paths. Update every downstream squeeze/broadcast/slice operation to retain the sample axis; deleting one mean line without changing its callers is insufficient.

Test single-sample equivalence with the upstream interface using the same RNG state. For multiple samples, either guarantee per-path RNG independent of chunk size or freeze chunk size and include it in artifact identity. If hardware requires a new chunk size and exact equivalence is not established, it is a new declared sampling configuration.

## 8. Weekly fine-tuning algorithm — adapter required

### 8.1 Initial optimization settings

These are proposed starting values, not tuned performance claims.

| Parameter | First registered setting |
|---|---|
| Predictor initialization | Pinned Kronos-small, reloaded independently every fold |
| Tokenizer | Original base tokenizer, frozen in evaluation mode |
| Trainable weights | All predictor parameters; no tokenizer updates |
| Optimizer | AdamW, learning rate 1e−5, betas (0.9, 0.95), weight decay 0.1 |
| Epoch budget | Maximum 10; choose using V-select Brier score |
| Early stopping | Patience 3 evaluations; tie-breaker: earliest epoch |
| GPU microbatch | 4 training examples initially |
| Effective batch | 32 via 8 accumulation microsteps after implementation |
| Precision | FP32 correctness baseline; optional measured AMP implementation with numerical parity checks |
| Gradient clipping | Global norm 1.0, after unscaling if AMP is used |
| Scheduler | 10% warmup then cosine over the registered maximum optimizer-step budget |
| Validation sampling | Same 256-path configuration as OOS, on the complete eligible V-select cohort |
| Data workers | Start 0, then measure; any change must preserve sampled examples and seeds |

With gradient accumulation, divide the microbatch loss by the effective accumulation count; step the optimizer/scheduler only when an accumulation group finishes. Handle the final partial group with correct weighting. Base the scheduler on actual optimizer steps, not dataloader batches. Do not claim that the current CSV `accumulation_steps` key implements this for the predictor.

Keep the number of candidates small. Do not simultaneously search model size, context, horizons, tokenizer mode, sampling temperature, LR, calibration and exit thresholds. The initial FT-versus-ZS experiment should explain whether ES adaptation helps at all.

### 8.2 Per-fold execution order

1. Resolve C, A and all date blocks; verify complete available history, artifact provenance and calendar eligibility. Write a planned manifest even for an excluded/unfit fold.
2. Build the fold-vintage ES inputs and permitted label windows. Fail on integrity faults. Freeze eligible-origin IDs before model generation.
3. Load the original tokenizer and original predictor. Verify file hashes. Create a fresh optimizer, scheduler and RNG state.
4. Fine-tune on TRAIN only, with corrected windowing/normalization/loss masks.
5. After each epoch, generate free-running V-select forecasts. Select the minimum Brier score; stop after the frozen patience rule. Record all tried epochs and selection scores.
6. Reload the selected model and freeze it. Generate V-cal paths and fit the prespecified calibration method. Do this independently for the untouched and fine-tuned arms with equal calibration budgets.
7. Freeze model, tokenizer, calibration parameters, input policy, origin list, seed policy and metrics. Hash this fold state before TEST scoring.
8. Walk chronologically through TEST. At every eligible origin use only completed observations available then. Prior observed TEST bars may enter later contexts; TEST labels may not alter weights or calibration.
9. Save predictions first, then join realized outcomes in a separate scoring step. Report failures and unavailable origins along with scores.
10. Close the fold manifest with hashes, timing, memory and status. Advance C by one calendar week. Reload the original weights for the next fold.

An interrupted run may resume **the same fold** from its verified checkpoint. It must not use the preceding week's fitted model as a starting point; that retains information older than the declared rolling window and changes the experiment. Resume requires matching data, code, vintage, source-weight hashes and configuration.

### 8.3 Calibration

Start with an optional two-parameter logistic map fitted on V-cal only:

```text
p_raw   = number of generated endpoint losses / M
p_input = clip(p_raw, 1/(2M), 1 − 1/(2M))
p_cal   = sigmoid(a + b × logit(p_input))
```

Freeze the fitting method/regularization before the first scored run. Report raw and calibrated probabilities. Use the same map class and fallback rules for both Kronos arms. When calibration is unavailable, the default fallback is `p_cal=p_raw` with a reason flag. Calibration of the −5-point endpoint event does not automatically calibrate stop-touch events or other horizons.

Selecting a model on V-select and assessing its chosen checkpoint there is selection, not independent performance evidence. The OOS week supplies the score; the multiweek aggregate supplies the uncertainty assessment.

## 9. Measure model performance on the real OOS data

### 9.1 Forecast targets and formulas

Let C_t be the latest observed completed close, C_(t+h) the observed endpoint, and C^(m)_(t+h) the endpoint of generated path m. All prices use the same fold basis; h is elapsed minutes within an uninterrupted valid horizon.

| Target | Observed value | Forecast derived from individual paths |
|---|---|---|
| Endpoint move | `D_h = C_(t+h) − C_t`, in ES points | Sample mean for RMSE; sample median for MAE; sample quantiles for intervals |
| Primary downside event | `y = 1[D_15 <= −5]` | `p_raw = mean_m(1[C^(m)_(t+15) − C_t <= −5])` |
| Current-price adverse excursion | `max(0, C_t − min_j L_(t+j))` | Distribution of the same function, computed separately on each path |
| Current-price favorable excursion | `max(0, max_j H_(t+j) − C_t)` | Distribution of the same function on each path |
| Fixed barrier touch | `1[min_j L_(t+j) <= S_t]` | Fraction of valid paths touching a barrier known at t |
| Realized return variance | `sum_(j=1..h) log(C_(t+j)/C_(t+j−1))²`, with first denominator C_t | Compute variance per generated path, then summarize the sample distribution |

The return-variance definition here deliberately includes the initial transition from C_t. Label it explicitly; do not mix it with a formula that excludes that transition. For a comparison with your existing GK model, reproduce the exact existing GK target, aggregation and units on both the observed and simulated paths. Do not compare MAEs from different definitions or annualizations.

An endpoint event and a barrier touch differ: a path may fall six points and then finish two points above C_t. For the −5-point endpoint event its label is zero; for a barrier at C_t−5 its touch label is one. Neither is automatically a stop-fill probability. One-minute OHLC cannot determine which of a stop and target was hit first when both were touched inside that minute.

**Do not calculate downside, excursions or variance from the average OHLC path.** For example, taking the minimum after averaging removes adverse movement that occurred at different times in different samples.

### 9.2 Minimum comparison table

| Arm | Fitted state | Required role |
|---|---|---|
| Persistence | None: endpoint prediction is C_t | Mandatory point-error reference |
| Empirical ES baseline | TRAIN-only endpoint-move distribution and event frequency | Mandatory probabilistic reference; optionally a prespecified clock-bucket version with pooled fallback |
| K-ZS | Original pinned tokenizer/predictor | Measures what pretraining provides before ES adaptation |
| K-FT | Original tokenizer + fold-fine-tuned predictor | Measures incremental adaptation value |
| K-TFT, later | Fold-adapted tokenizer + fold-adapted predictor | Tests representation adaptation as a separate candidate |
| TRUELIGHT/B1, next stage | Fresh matched-fold fits | Tests incremental information relative to existing work |

“K-ZS” refers to the unchanged neural checkpoint. If its probability is calibrated on V-cal, call the reported probability **calibrated zero-shot model output**, since the complete pipeline has a local learned component.

For the TRUELIGHT comparison, make a fixture proving that the one-minute close at the 15-minute endpoint equals the corresponding close of three aggregated five-minute bars, at the identical origin/availability convention. If the supporting model has a different cohort, score the intersection and show the full attrition table. Do not attribute a cohort change to model skill.

### 9.3 Metrics

**Primary for this fine-tuning runbook:** paired OOS improvement in Brier score for the 15-minute, −5-point endpoint event:

```text
BS(arm)      = mean_i (p_arm,i − y_i)²
Delta_BS     = BS(K-ZS) − BS(K-FT)        # positive means fine-tuning helps
Brier_skill  = 1 − BS(arm)/BS(empirical_ES_baseline)
```

The baseline predictions must have been estimated before each OOS week. Never construct the reference event rate from the same OOS outcomes being scored. Report a zero baseline denominator as undefined, not an infinite skill score.

Report these parallel diagnostics:

- **Point forecasts:** MAE in ES points, RMSE in points, signed bias, and Spearman correlation between predicted and realized **moves**, not raw price levels. State mean versus median forecast choice.
- **Probabilities:** raw and calibrated Brier score; Brier skill; log loss with a frozen floor (for example 1e−6); AUC only as a secondary ranking diagnostic; event prevalence and reliability bins.
- **Distributions:** pinball loss at quantiles 0.10/0.50/0.90; 80% interval coverage and average width; CRPS for the endpoint move.
- **Risk paths:** MAE/MFE quantile coverage and touch-event scores only on structurally valid generated paths under the declared handling policy.
- **Operations:** origin coverage, failed predictions, invalid candles, inference latency p50/p95, training wall time, peak VRAM and artifact size.

For quantile q_tau and realized move D, pinball loss is `(tau − 1[D < q_tau]) × (D − q_tau)`. For M generated endpoint moves d_m, empirical CRPS is:

```text
CRPS = (1/M) sum_m |d_m − D|
       − (1/(2 M²)) sum_m sum_n |d_m − d_n|
```

Compute it in sorted-sample form to avoid a large M-by-M tensor per origin. A sharp interval is useful only with adequate coverage; a 90th-percentile bound breached much more than 10% of the time is not a calibrated tail estimate.

More generated paths reduce sampling noise only. At probability 0.30 with M=256, the conditional Monte Carlo standard error is approximately `sqrt(0.3*0.7/256)=0.0286`. This does not measure model uncertainty or prove that the true event frequency is 30%.

### 9.4 Failed or invalid forecasts

Freeze the eligible origin grid from data/calendar rules before looking at model output. Maintain separate endpoint and full-OHLC-path validity:

- An endpoint forecast requires all M endpoint closes to be finite and positive. Do not discard individual bad samples and renormalize the remaining probabilities.
- Path excursions and touch metrics additionally require coherent OHLC across each full sampled horizon. A bad high/low disables those path metrics; it need not hide an otherwise valid close-only diagnostic.
- Record unavailable or negative auxiliary volume/amount separately. Do not silently project malformed candles into valid ones. Any projection/resampling is a separately evaluated forecast transformation.
- If an endpoint forecast is unavailable, use the frozen TRAIN-only empirical event-rate fallback for the primary all-origin comparison and label the fallback. Also report model-only paired scores and coverage. A model cannot improve its headline score merely by failing on difficult origins.
- Report both strict and numerical-tolerance counts using tolerances frozen before results. Preserve raw outputs for diagnosis.

### 9.5 Multiweek aggregation and interpretation

One OOS week verifies the pipeline; it rarely establishes an edge. Run all provenance-eligible, fully covered weeks in the frozen range. Report all expected weeks, including excluded, missing and failed ones, and distinguish partial latest weeks from complete weeks.

For statistical uncertainty, use paired differences on identical origins, aggregate to weeks, then resample **blocks of adjacent weeks** with the same resampled indices for every arm. A proposed reporting default is 2,000 bootstrap draws, four-week blocks and a fixed seed; freeze these choices in the experiment specification. A short history or very few events may be inconclusive regardless of the nominal number of minute bars.

Report pooled-origin and equal-week-weighted scores separately. Preserve negative weeks and weak regimes. Overlapping 15-minute forecasts on a five-minute grid are dependent, and training sets also overlap between weeks. Do not use an IID minute-row confidence interval or annualize a forecast score as if it were portfolio returns.

Record the full candidate/epoch/hyperparameter search budget. Before making an edge claim, set the minimum practically useful Brier improvement, coverage floor, calibration tolerance and minimum effective weeks/events. This runbook does not invent an economically meaningful numerical threshold from unobserved results. Until those gates are registered, the output is a descriptive forecast benchmark.

The merged idea's later primary claim remains **incremental value beyond TRUELIGHT/B1**, using matched combinations. Beating K-ZS alone does not demonstrate that claim. Keep the stages distinct.

## 10. Connect to the existing test framework

Reuse the existing framework's calendar, split algebra, vintage construction, artifact identity and reporting conventions. Do not invoke its legacy trading runner and assume it has run Kronos.

| Existing component | Reuse / adaptation |
|---|---|
| `data/market_data/01-Minutes` | Read-only one-minute source; inspect schema/timezone/actual coverage |
| `data/market_data/contract_summary.csv` | Roll-offset source with causal audit |
| `/mnt/DATA/market-data-store/reference/es_contract_windows.csv` | Current configured contract calendar; bind/version the actual local path |
| `nexus_matrix.calendar_es.fold_blocks` | Canonical checkpoint date algebra |
| `nexus_matrix.market_data` helpers | Additive vintage and equality/invariance checks |
| `.ai-agent/models/time-series-standards.md` | UTC availability, left-label semantics and tumbling training windows |
| Existing forecast/report tables | Match IDs and units; add Kronos fields without replacing existing model results |

Suggested local integration layout, **to be implemented**, under the Kronos fork:

```text
es_walkforward/config.py       typed, versioned experiment configuration
es_walkforward/data.py         ES source mapping and UTC availability
es_walkforward/folds.py        bridge to audited QDL fold/calendar/vintage helpers
es_walkforward/dataset.py      causal normalization, tumbling samples, loss masks
es_walkforward/train.py        predictor fitting and resume state
es_walkforward/predict.py      path-preserving local inference
es_walkforward/calibrate.py    V-cal-only probability mapping
es_walkforward/evaluate.py     real outcomes, metrics, paired reports
es_walkforward/cli.py          stage execution and all-week orchestration
tests/es_walkforward/          temporal and artifact-invariant checks
```

Avoid putting the new package under the name `model` in a shared Python path: Kronos already has a top-level `model` package. A dedicated virtual environment/subprocess interface with Parquet artifacts avoids import-name and dependency collisions with QDL. Reuse shared functions through a small explicit bridge, or copy a minimal pure-function implementation with its upstream commit and parity tests; do not maintain a silently divergent calendar.

### 10.1 Proposed experiment config — adapter required

This is the desired schema for `configs/es_1m_walkforward.yaml`. The current CSV `ConfigLoader` does **not** implement it.

```yaml
experiment_id: KRONOS-ES-1M-WF-v1
research_track: external_pretraining
data_root: /mnt/DATA/quant-discovery-lab/data/market_data/01-Minutes
qdl_root: /mnt/DATA/quant-discovery-lab
contract_windows: /mnt/DATA/market-data-store/reference/es_contract_windows.csv
artifact_root: /mnt/DATA/kronos_es/artifacts
artifact_lock: pretrained/artifacts.lock.json

data:
  source_timestamp_semantics: bar_open
  source_timezone: America/Chicago  # legacy store only; verify against actual file
  authoritative_timezone: UTC
  feature_clock_timezone: America/Chicago
  bar_minutes: 1
  adjustment: additive_fold_vintage
  amount_mode: explicit_zero
  context_bars: 256
  horizon_bars: 15
  training_window_stride: 271
  require_consecutive_minutes: true
  allow_synthetic_bars: false

folds:
  development_months: 24
  validation_days: 28
  selection_days: 14
  calibration_days: 14
  test_weeks: 1
  checkpoint_weekday: Saturday
  reset_to_original_weights: true
  exclude_roll_weeks: true

model:
  predictor_path: pretrained/Kronos-small
  tokenizer_path: pretrained/Kronos-Tokenizer-base
  train_tokenizer: false
  max_context: 512

training:
  learning_rate: 0.00001
  max_epochs: 10
  patience: 3
  microbatch_size: 4
  gradient_accumulation_steps: 8
  precision: fp32
  selection_metric: endpoint_brier
  loss_mask: future_only
  seed: 42

inference:
  decision_grid_minutes: 5
  primary_session: RTH
  first_decision_local: '08:35'
  last_decision_local: '14:45'
  sample_count: 256
  chunk_size: 8
  temperature: 1.0
  top_k: 0
  top_p: 1.0

evaluation:
  endpoint_horizon_minutes: 15
  endpoint_threshold_points: -5.0
  comparison: K-FT_vs_K-ZS
  calibration: logistic_vcal_only
  missing_forecast_policy: train_empirical_fallback
  bootstrap_unit: consecutive_week_blocks
  bootstrap_block_weeks: 4
  bootstrap_replicates: 2000
  claim_mode: descriptive_until_numeric_gates_registered
```

The implementation must reject unknown/misspelled keys and missing source provenance. Store the resolved absolute paths and all defaults in each manifest. Never permit a default source timezone to override conflicting file metadata.

### 10.2 Proposed CLI contract — not implemented in the reviewed repository

After the adapter and acceptance checks exist, its commands should support this exact operational sequence. These are **implementation requirements**, not presently runnable upstream commands:

```bash
# Bind data, source/model hashes and eligible dates; no training.
python -m es_walkforward.cli preflight --config configs/es_1m_walkforward.yaml

# Inspect one exact fold's dates, rows, vintage and coverage before training.
python -m es_walkforward.cli prepare \
  --config configs/es_1m_walkforward.yaml --checkpoint 2026-02-07

# Complete one historical development/pilot fold, including both arms.
python -m es_walkforward.cli run-fold \
  --config configs/es_1m_walkforward.yaml --checkpoint 2026-02-07 \
  --arms K-ZS K-FT --run-id es1m-pilot-001

# Full frozen date range, once provenance and data coverage support it.
python -m es_walkforward.cli run-range \
  --config configs/es_1m_walkforward.yaml \
  --first-checkpoint 2026-02-07 --last-checkpoint 2026-08-29 \
  --arms K-ZS K-FT --run-id es1m-wf-001

# Resume only artifacts with exact matching input/config/model identities.
python -m es_walkforward.cli resume \
  --config configs/es_1m_walkforward.yaml --run-id es1m-wf-001

# Aggregate frozen prediction tapes; no fitting or hyperparameter selection.
python -m es_walkforward.cli report \
  --config configs/es_1m_walkforward.yaml --run-id es1m-wf-001
```

The example range is not a promise of available data and is not an instruction to skip earlier eligible weeks. Preflight should enumerate the full eligible range from the verified provenance bounds and the data freeze. It should refuse a clean-OOS label for ineligible dates. All multiweek runs retain the complete census of expected folds.

For debugging, support explicit `--stage prepare|fit|calibrate|predict|score` and `--limit-origins` in an `engineering_only` mode. Any limited/subsampled run must carry that status and cannot be merged into the primary full-population result. No task is scheduled or executed by this runbook.

## 11. Required artifacts, resources and failure recovery

### 11.1 Per-fold output

Write each run to an immutable experiment/run/checkpoint/arm identity. At minimum:

| Artifact | Required contents |
|---|---|
| `fold_manifest.json` | C/A/date bounds, session masks, all hashes, model/tokenizer revisions, price vintage, provenance/exposure status, seed and fit/score status |
| `data_audit.json` | Source counts, UTC conversion, missing/synthetic bars, discarded window tails, calendar/contract exclusions, origin attrition |
| `origin_grid.parquet` | Fixed eligible decision IDs, input end/availability, target timestamps and expected cohort |
| `model/` | Selected local predictor weights and config; original tokenizer identity or adapted pair |
| `training_state.pt` | Trusted locally produced resume state: optimizer, scheduler, epoch, step, RNG and sampler position |
| `calibration.json` | Method/parameters, V-cal dates, counts, artifact input hashes, fallback status |
| `paths/` | Chunked float32 arrays `[origin,sample,horizon,6]`, axis definitions, UTC times and raw-output validity flags |
| `forecasts.parquet` | Point/quantile forecasts, raw/calibrated event probabilities, availability and status |
| `outcomes.parquet` | Observed labels and resolution times, generated separately from predictions |
| `metrics.json`, `weekly_metrics.csv` | Paired scores, baselines, coverage, reliability, runtime and resource use |
| `report.md` | Method, data budget, all weekly results, uncertainty and limitations |

Every forecast row must include: `experiment_id`, `run_id`, `checkpoint`, `arm`, `origin_id`, `contract`, `vintage_id`, UTC `origin_time`, `input_available_max`, `forecast_available_at`, `target_end_time`, context/horizon lengths, `sample_count`, seed identity, predictor/tokenizer hashes, `p_raw`, `p_cal`, fallback/quality status and prediction artifact hash.

No realized target, future high/low, future regime classification or future P&L field belongs in the serialized inference request. Join labels only after predictions are fixed. Keep model weights and data outside Git; commit small configs, source, manifests and concise reports according to repository policy.

### 11.2 Resource planning

On the initial 08:35–14:45 grid there are at most 75 origins/full day, or 375 in a normal five-day week. With 256 paths, 15 bars and six float32 channels, the uncompressed path values alone occupy approximately **33 MiB per arm per full week**, excluding IDs, logs and checkpoints. Training checkpoints will often dominate storage.

The same week requires 1,440,000 generated one-minute bars per arm. Two validation weeks at every training epoch can cost more than the final OOS generation. Measure:

- GPU time for one training epoch, with actual example count.
- Time for 10 development origins at 1, 8 and 256 paths.
- Peak allocated/reserved VRAM and physical free memory.
- Tokenizer encode time, autoregressive generation time, decoding and disk I/O separately.

Use those measurements to estimate the full run before launching it. Change sampling/validation budgets only before the evaluation specification is frozen. A resource-driven change is still a configuration change. Avoid claiming that a 15-minute forecast can drive a five-minute decision loop if its measured latency misses the next decision boundary.

For CUDA OOM, first reduce the **path chunk** or training microbatch, not the number of forecast samples or context length. Maintain the registered effective batch via implemented accumulation. If the changed batching alters reproducibility, create a new configuration identity and regenerate that arm consistently.

### 11.3 Troubleshooting

| Symptom | Likely cause / action |
|---|---|
| `No module named yaml` | Install PyYAML; it is not in root requirements |
| `No module named model` / `config_loader` | Use the documented working directory/PYTHONPATH; avoid conflicting top-level packages |
| Fine-tuned tokenizer path does not exist | Predictor-only route must explicitly point `finetuned_tokenizer` at the original local tokenizer |
| `UnboundLocalError` for `os` in standalone base trainer | Use sequential launcher; fix the function-local import before using that entrypoint |
| Empty/negative sample count | Split or segment is shorter than its required window; emit unfit status, do not fabricate rows |
| GPU memory changes after setting `accumulation_steps` | Stock predictor ignores that feature; implement real accumulation or reduce batch size |
| Very smooth path / low apparent volatility | Multiple samples were averaged before calculating risk |
| Perfectly repeated Monte Carlo paths | Same seed reset for each sample, dropout/state issue, or accidental output reuse |
| Implausibly strong old-year performance | Audit pretraining dates, future-inclusive normalization, full-sequence validation attention and price vintage |
| Forecast unavailable at open | Insufficient real premarket context, a calendar gap, or source coverage limitation |
| Different results after resume | Missing RNG/optimizer/sampler state, changed batch/chunk size, source revision or nondeterministic backend |
| Strong token loss but poor Brier/MAE | Token objective did not produce useful free-running ES forecasts; retain the negative finding |

## 12. Acceptance sequence and what to do first

### 12.1 Tests required before the first scored week

| Test | Pass condition |
|---|---|
| Fold date parity | Adapter bounds match the existing framework exactly, including calendar-month and DST cases |
| Availability fixture | A 09:59-open minute is usable at 10:00 plus latency; 10:00-open OHLC is absent at that origin |
| Future mutation | Changing bars after an origin leaves its inputs, normalization, context tokens and forecast unchanged under a fixed seed |
| Split purge | TRAIN target availability precedes V-select; selection targets precede V-cal; calibration targets precede A |
| Provenance | Both tokenizer and predictor eligibility are checked for selection/calibration/OOS origins |
| Price vintage | OOS raw equality; earlier features/predictions invariant to future roll additions |
| Session integrity | No hidden concatenation across exchange closure, gap, synthetic bar or contract boundary |
| Training windows | Tumbling segments and explicit future-only loss indices; no input beyond the supported context |
| Causal evaluation | Full-sequence diagnostics cannot see later positions; primary evaluation uses free-running prefixes |
| Frozen tokenizer | No gradients/optimizer updates; deterministic eval-mode encoding; tokenizer checksum unchanged |
| Path dimensions | M independently seeded paths retained; M=1 agrees with the upstream path within declared tolerance |
| Risk fixture | Stop-touch-and-recovery example distinguishes endpoint event from barrier touch |
| Bad output | Invalid samples are retained/flagged; fallback and metric denominators follow frozen rules |
| Weekly reset | Each fold's initial predictor hash equals the original pinned hash, not last week's adapted hash |
| Resume | Interrupted and uninterrupted development runs reproduce predictions within declared tolerance |
| Reproduction | Complete run can be rebuilt from immutable config, data and weight identities |

Use synthetic or explicitly available development fixtures for these checks. Preserve the actual evaluation data's exposure status. Existing researched historical weeks can be used for authorized exploratory evaluation; they do not become unseen merely because a new model is applied.

### 12.2 Order of work

1. **Environment and provenance:** pin source/weights, inspect local GPU and dataset schema, determine the eligible historical span.
2. **Inference smoke:** run section 5 on development data; verify actual timestamps, 15-row output and individual paths.
3. **Fine-tuning smoke:** run section 6 once to prove loading, backward pass, checkpoint saving and reload work locally.
4. **Implement the adapter:** fix the issues in section 7; connect the existing date/vintage helpers; complete acceptance checks.
5. **One development/pilot fold:** compare K-ZS, K-FT and simple baselines; measure time/memory. Mark that week exposed.
6. **Freeze the full experiment:** register origin/cohort rules, finite search budget, calibration, metrics, failure behavior and any numeric evidence gates.
7. **Run every eligible OOS week:** collect immutable forecast tapes and paired multiweek results, including negative and missing outcomes.
8. **Decide from evidence:** retain K-ZS if adaptation does not help; reject Kronos if it fails simple baselines or validity/calibration checks; test incremental TRUELIGHT/B1 information only if justified.

If a later exit study is justified, follow the current Nexus rev1 economic contract: identical-entry controls, feasible one-position replay, one LONG ES contract, capital Policies A/B, hard risk and calendar limits, $5 round-trip commission and separately modeled 0.25-point adverse market-fill slippage. Lower stop-loss frequency alone is not success. No broker connection or live-trading step belongs in this model runbook.

## 13. Source links and validation performed for this document

- [Kronos paper, HTML corresponding to PDF v1](https://arxiv.org/html/2508.02739v1)
- [Inference and tokenizer implementation](https://github.com/Kevin7806/Kronos/blob/67b630e67f6a18c9e9be918d9b4337c960db1e9a/model/kronos.py)
- [Attention and loss implementation](https://github.com/Kevin7806/Kronos/blob/67b630e67f6a18c9e9be918d9b4337c960db1e9a/model/module.py)
- [CSV fine-tuning guide](https://github.com/Kevin7806/Kronos/blob/67b630e67f6a18c9e9be918d9b4337c960db1e9a/finetune_csv/README.md)
- [CSV dataset and predictor training loop](https://github.com/Kevin7806/Kronos/blob/67b630e67f6a18c9e9be918d9b4337c960db1e9a/finetune_csv/finetune_base_model.py)
- [Sequential training launcher](https://github.com/Kevin7806/Kronos/blob/67b630e67f6a18c9e9be918d9b4337c960db1e9a/finetune_csv/train_sequential.py)
- [Configuration loader](https://github.com/Kevin7806/Kronos/blob/67b630e67f6a18c9e9be918d9b4337c960db1e9a/finetune_csv/config_loader.py)
- [Repository time-series standards](https://github.com/Kevin7806/quant-discovery-lab/blob/ed86575aa1eddcc8c712a2f3c8746a05d0a8235b/.ai-agent/models/time-series-standards.md)
- [Canonical weekly fold algebra](https://github.com/Kevin7806/quant-discovery-lab/blob/ed86575aa1eddcc8c712a2f3c8746a05d0a8235b/strategy/Nexus_Matrix/code/nexus_matrix/calendar_es.py)
- [Price-vintage helpers](https://github.com/Kevin7806/quant-discovery-lab/blob/ed86575aa1eddcc8c712a2f3c8746a05d0a8235b/strategy/Nexus_Matrix/code/nexus_matrix/market_data.py)
- [Existing legacy fold runner](https://github.com/Kevin7806/quant-discovery-lab/blob/ed86575aa1eddcc8c712a2f3c8746a05d0a8235b/strategy/Nexus_Matrix/code/run_fold.py)
- [Current Nexus rev1 requirements](https://github.com/Kevin7806/quant-discovery-lab/blob/ed86575aa1eddcc8c712a2f3c8746a05d0a8235b/strategy/Nexus_Matrix/NEXUS-MATRIX-economic-trading-test-requirements-rev1.md)

**Checks performed while preparing the runbook:** read the paper's method/experimental details and the pinned implementations/contracts; reproduced the CSV normalization problem using the actual dataset class extracted into a synthetic NumPy/Pandas fixture; verified the standalone import-scope defect through Python symbol analysis; checked the worked fold dates and proposed document snippets/configuration syntax. In the mutation fixture, changing only future targets changed historical normalized price values by up to 1.7181.

**Not executed here:** model-weight download, CUDA inference, fine-tuning, the proposed adapter or any ES OOS performance evaluation. Your local ES files and GPU were not accessed. The environment/configuration and adapter choices remain proposed until validated on your workstation. This document specifies the research workflow and required adapter work; it does not implement or execute a strategy.
