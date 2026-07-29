<div align="center">

<img src="./public/exomamba-logo.png" alt="ExoMamba" width="450">

# 

### Selective state-space models for exoplanet vetting in TESS light curves.

Linear-time sequence modeling over complete 18,000-cadence light curves, measured against a ladder of baselines: stratified random, catalog logistic regression, single-branch CNN and dual-branch AstroNet. Reported with bootstrap confidence intervals, DeLong tests and a negative ablation.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Mamba](https://img.shields.io/badge/Mamba-State_Space_Model-111827?style=for-the-badge)](https://github.com/state-spaces/mamba)
[![TESS](https://img.shields.io/badge/TESS-Light_Curves-0B3D91?style=for-the-badge)](https://science.nasa.gov/mission/tess/)
[![CI](https://img.shields.io/github/actions/workflow/status/JoseZum/ExoMamba/ci.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI)](https://github.com/JoseZum/ExoMamba/actions)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br>

**José Fabián Zumbado Ruiz · Jeremmy Aguilar Villanueva**<br>
School of Computing · Instituto Tecnológico de Costa Rica

<br>

[Results](#results-sealed-test-set) · [Background](#background-what-is-this-project-about) · [Installation](#installation) · [Pipeline](#reproducing-the-full-pipeline) · [Status](#project-status)

</div>

---

The accompanying paper is `paper/paper_ssm_tess_vetting.tex` — *Selective State-Space Models for TESS Exoplanet Vetting from Long Light Curves*. Developed for the Artificial Intelligence course at ITCR, Semester I 2026, advised by Kenneth Obando Rodríguez.

## Goal

Evaluate whether a **Mamba**-based architecture (Gu & Dao, 2023) can match or beat state-of-the-art 1D CNN classifiers (the AstroNet / ExoMiner family) on the binary task of separating **Confirmed Planets (CP)** from **False Positives (FP)** in TESS 2-minute-cadence light curves, operating directly on the raw `PDCSAP_FLUX` signal with no feature engineering.

This is a **controlled feasibility study**, not a production vetting catalog. The claim is deliberately narrow: under small data and a 4 GB GPU, Mamba extracts temporal signal that tabular features and a shallow CNN miss.

## Results (sealed test set)

| Model | Test AUC-ROC | 95% bootstrap CI |
|---|---:|---|
| Stratified random | 0.500 | — |
| Catalog LogReg | 0.605 | — |
| CNN single-branch | 0.604 | [0.529, 0.677] |
| Mamba locked (single seed) | 0.763 | — |
| **Mamba ensemble (5 seeds)** | **0.806** | **[0.747, 0.857]** |
| Mamba best seed (789) | 0.810 | [0.753, 0.862] |
| ExoMamba V1 (negative ablation) | 0.460 | [0.382, 0.541] |

The Mamba-vs-CNN gap is significant (DeLong *p* = 3.7e-8; paired-bootstrap ΔAUC CI [+0.13, +0.28]).

**Where the evidence lives.** Per-sample predictions, confidence intervals and DeLong tests are versioned under `paper/results/` — in particular `paper/results/statistics.json` and `paper/results/mamba_ensemble/`. Everything under `experiments/` is gitignored (checkpoints and logs are too large), so a fresh clone can recompute every number in the paper from the stored predictions, but cannot re-derive them from the raw checkpoints without retraining.

```bash
python scripts/compute_stats.py   # regenerates paper/results/statistics.json
```

---

## Background: what is this project about?

> For readers arriving without a background in astronomy or machine learning.

### What is an exoplanet, and how is it detected?

An **exoplanet** is a planet orbiting a star other than the Sun. We cannot photograph them directly — they are far too distant. One of the most widely used indirect methods is the **transit method**: when a planet passes in front of its star from our line of sight, it blocks a small fraction of the light. The star's brightness dips briefly and then returns to normal.

```
No transit:    ─────────────────────────────
With transit:  ───────────\____/───────────
```

<img src="public/transit_white.png" width="480" alt="Transit diagram"/>

If that dip is small, periodic and symmetric, it is evidence of an orbiting planet.

### What is a light curve, and why is it the model input?

A **light curve** is the time series of a star's brightness. TESS samples it every 2 minutes for roughly 27 days per sector, producing a sequence of about 18,000 points per star:

```
[1.0001, 0.9998, 1.0000, 0.9999, 0.9982, 0.9979, 0.9981, ...]
```

That sequence is exactly what the model receives as input. The transit signal is the dip in those values, barely perceptible against the noise.

### What are TESS and the TOI Catalog?

**TESS** (*Transiting Exoplanet Survey Satellite*, NASA, 2018) monitors the ~200,000 brightest dwarf stars in the sky at 2-minute cadence, plus full-frame images. Reviewing all of it by hand is impossible, which is why automated classifiers matter.

The **TOI Catalog** (*TESS Objects of Interest*) is the public table where NASA records every candidate TESS detects. Each star has a unique identifier (**TIC ID**) and a disposition:

| Disposition | Meaning | Use in this project |
|---|---|---|
| `CP` — Confirmed Planet | Planet confirmed by scientific review | **Positive class** (label = 1) |
| `FP` — False Positive | Signal ruled out: eclipsing binary, artifact, etc. | **Negative class** (label = 0) |
| `PC` — Planet Candidate | Not yet confirmed | Excluded from supervised training |
| `KP` — Known Planet | Confirmed by earlier missions | Excluded by experimental design |

After filtering for availability in `lightkurve`, the labelled dataset contains 1,576 TICs (603 CP / 973 FP).

Excluding `PC` is the single most consequential design decision here, and it makes the task easier than operational triage. `scripts/make_labels.py` implements the alternative formulations, and `docs/pc_fp_expansion.md` documents what moving to PC-vs-FP would cost and change.

TESS does not observe the whole sky at once: it splits it into **sectors**, each observed for about 27 days. The same star can appear in several sectors, producing multiple light curves for one TIC ID.

<img src="public/observation_sector.jpg" width="480" alt="TESS observation sectors"/>

### Per-star data leakage: the most common trap in this domain

Because one star can be observed across several sectors, a naive split can put sector 1 of a star in train and sector 13 of the same star in test. The model then learns that star's idiosyncrasies — its intrinsic variability and noise signature — and overfits the test set. The result is inflated metrics that say nothing about generalization. The split is therefore made **by TIC ID, never by sector**.

```
TIC 261136679 → train   (sectors 1, 2 and 13 all go to train)
TIC 123456789 → test    (all of its sectors go to test)
```

No star appears in more than one partition. This is enforced by `scripts/make_splits.py` at generation time and asserted on the versioned CSVs by `tests/test_splits.py`, so a later edit that broke it would fail CI.

### The sealed test set

The test split is evaluated **once per run**. This is enforced in code, not by convention:

- `scripts/evaluate.py` defaults to `--split val`; touching the test set requires asking for it explicitly.
- Each test evaluation appends a record (timestamp, git commit, sample count, AUC) to `test_seal_ledger.json`, which is versioned.
- A second evaluation of the same run directory aborts unless `--force-reeval-test` is passed, and that override is itself written to the ledger.

---

## Repository layout

```
ExoMamba/
├── configs/                # one YAML per experiment (one file = one reproducible run)
├── data/
│   ├── raw/                # .fits files downloaded from MAST     (gitignored)
│   ├── processed/          # tensors ready for training           (gitignored)
│   └── splits/             # train/val/test TIC IDs + manifests   (versioned)
├── src/exoplanet/          # source code as an installable package
│   ├── data/               # Dataset, augmentation
│   ├── models/             # cnn_baseline, mamba_baseline, astronet_multibranch, exomamba_v1
│   ├── training/           # loop, losses, schedulers, checkpoints, runner
│   ├── evaluation/         # plots, XAI
│   └── utils/              # seeds, logging, paths, git info
├── scripts/                # reproducible CLIs (one script per pipeline stage)
│   └── wsl2/               # shell helpers for the WSL2 environment
├── agent/                  # LLM vetting assistant that calls the model as a tool
│   ├── inference/          # FastAPI microservice serving the real Mamba (Docker + CUDA)
│   └── eval/               # scenario suite and metrics for the agent
├── notebooks/              # exploratory analysis
├── experiments/            # per-run outputs                      (gitignored)
├── tests/                  # pytest (73 tests)
├── docs/                   # internal documentation
├── public/                 # images used by this README
└── paper/                  # LaTeX paper + figures + tables + versioned results
```

---

## Installation

**Prerequisites:**

- Python **3.10 or 3.11** (tested on 3.11.9). On Windows, prefer the official [python.org](https://www.python.org/downloads/) build over the Microsoft Store version.
- Git Bash or PowerShell on Windows; bash on Linux/macOS.
- ~2.5 GB of free disk space for the environment (PyTorch with CUDA included).

> **OneDrive note:** if the repository lives inside a OneDrive-synced folder, move it to a local path (e.g. `C:\dev\ExoMamba\`) **before** creating the `.venv`. OneDrive will try to sync thousands of virtual-environment files and can corrupt PyTorch binaries.

### 1. Clone

```bash
git clone https://github.com/JoseZum/ExoMamba
cd ExoMamba
```

### 2. Create and activate the virtual environment

```bash
python -m venv .venv

# Activate — Git Bash on Windows:
source .venv/Scripts/activate
# Activate — PowerShell:
# .venv\Scripts\Activate.ps1
# Activate — Linux / macOS:
# source .venv/bin/activate
```

### 3. Install the package in editable mode

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

This installs the `exoplanet` package and the dependencies declared in `pyproject.toml`, including `torch` (CPU build by default), `lightkurve`, `astropy`, `jupyterlab`, `pytest` and `ruff`.

### 4. Reinstall PyTorch with CUDA (needed for GPU training)

The CPU build of `torch` will not use the GPU. **Check your driver's CUDA version first:**

```bash
nvidia-smi    # look for "CUDA Version: XX.Y" in the top-right corner
```

Then replace the CPU build with the matching CUDA wheel. With driver 581+ (CUDA 13.0), the CUDA 12.8 wheel works:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

For other CUDA versions see <https://pytorch.org/get-started/locally/>.

Verify:

```bash
python -c "import torch; print('CUDA OK' if torch.cuda.is_available() else 'CPU only', '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### 5. WSL2 setup for Mamba (only needed to train Mamba)

`mamba-ssm` compiles CUDA extensions with `nvcc` and ships no prebuilt wheels for native Windows. The Mamba model is therefore developed and trained under WSL2 with Ubuntu 24.04. Everything else — download, preprocessing, CNN baselines, evaluation — runs on native Windows.

```powershell
# In an elevated PowerShell (Windows):
wsl --install -d Ubuntu-24.04
```

```bash
# Inside Ubuntu WSL2, from wherever you cloned the repo:
cd /mnt/c/path/to/ExoMamba
chmod +x scripts/setup_wsl2.sh
./scripts/setup_wsl2.sh
```

`setup_wsl2.sh` is idempotent and handles everything: apt dependencies, nvcc, venv, torch+cuda, `pip install -e ".[dev,mamba]"`, and a final `verify_wsl2_env.py` check.

The pinned versions matter and are not arbitrary: `mamba-ssm` is held at `<2.3` because 2.3+ requires `triton>=3.5`, which only ships with `torch>=2.12`, while this environment runs torch 2.5.1+cu121 (triton 3.1). `causal-conv1d` and `mamba-ssm` are installed from git tags rather than PyPI, because the PyPI source distributions omit the `csrc/` directory needed to build the kernels. See `agent/inference/Dockerfile` for a fully specified, reproducible build of this environment.

### 6. Final check

```bash
pytest -q                                                      # 73 tests
python -c "import exoplanet; print(exoplanet.__version__)"     # → 0.1.0
```

Tests that depend on downloaded or preprocessed data skip themselves on a clean clone rather than failing.

---

## Reproducing the full pipeline

Commands marked **[WSL2]** require Linux plus `mamba-ssm`; the rest run on native Windows. All are executed from the repository root with the virtualenv active.

### 1. Data: download and preprocessing (once)

```bash
# TOI catalog → data/splits/toi_summary.csv
python scripts/get_data.py

# Label definition (CP vs FP). Inspect before writing:
python scripts/make_labels.py --task cp_fp --dry-run
python scripts/make_labels.py --task cp_fp

# Light curves from MAST (~3-4 h, ~7 GB; idempotent and resumable)
python scripts/download_lightcurves.py --max-sectors 3 --shuffle

# Tier 1 preprocessing: one global L=18000 tensor per TIC
python scripts/preprocess_global.py

# TIC-level splits (70/15/15)
python scripts/make_splits.py
```

Versioned outputs: `data/splits/{train,val,test}_tics.csv`.
Large outputs (gitignored): `data/raw/`, `data/processed/`.

### 2. Train the Tier 1 baselines

```bash
# Stratified random (~5 s, CPU)
python scripts/train.py --config configs/random_baseline.yaml

# Single-branch CNN (~30 min, CPU or GPU)
python scripts/train.py --config configs/cnn_baseline.yaml

# Mamba locked baseline  [WSL2, ~1 h]
python scripts/train.py --config configs/mamba_small.yaml

# Mamba multi-seed sweep  [WSL2, ~1 h × 5]
for seed in 42 123 456 789 2024; do
    python scripts/train.py --config configs/mamba_small.yaml \
        --seed $seed --name-suffix "_seed${seed}"
done

# Logistic regression on catalog features (~10 s, CPU)
python scripts/train_logreg.py
```

### 3. Evaluate against the sealed test set (once per run)

```bash
python scripts/evaluate.py --run experiments/<run_dir> --split test
```

Each evaluation writes `<run_dir>/eval_test/{metrics.json, predictions.csv, roc_curve.png, pr_curve.png, confusion_matrix.png, calibration.png}` and records the event in `test_seal_ledger.json`. Running it twice on the same run aborts by design — see [The sealed test set](#the-sealed-test-set).

Omit `--split test` to evaluate the validation split, which is unrestricted and is what you want during development.

### 4. Ensembles (probability averaging across seeds)

```bash
python scripts/ensemble_eval.py \
  --runs experiments/<seed42>,experiments/<seed123>,experiments/<seed456>,experiments/<seed789>,experiments/<seed2024> \
  --split test \
  --output-dir paper/results/mamba_ensemble
```

### 5. Inference statistics, figures and analysis

```bash
# Bootstrap CIs (2,000 stratified resamples) + DeLong tests
python scripts/compute_stats.py            # → paper/results/statistics.json

# Comparative ROC curve
python scripts/plot_tier1_comparison.py    # → paper/figures/roc_tier1.png

# [WSL2] Saliency + Integrated Gradients + Occlusion on 8 cases
# (top-2 per TP/TN/FN/FP quadrant)
python scripts/run_xai.py \
  --run experiments/<mamba_seed789> \
  --split test \
  --output paper/figures/xai/mamba_seed789

# Error analysis on the ensemble
python scripts/error_analysis.py \
  --predictions paper/results/mamba_ensemble/ensemble_predictions.csv \
  --catalog data/splits/toi_summary.csv \
  --output paper/results/error_analysis/mamba_ensemble
```

### 6. Build the paper

```bash
cd paper
pdflatex paper_ssm_tess_vetting.tex
pdflatex paper_ssm_tess_vetting.tex   # second pass resolves references
```

The bibliography is embedded in a `thebibliography` block, so the document compiles standalone with no BibTeX pass. `paper/references.bib` is the canonical source used to generate per-venue variants.

### 7. Tests and linting

```bash
pytest -q
ruff check .
```

Both run automatically on every push via GitHub Actions (`.github/workflows/ci.yml`).

---

## Reference environment

The exact environment used to produce the reported results.

| Parameter | General pipeline (Windows) | Mamba model (WSL2) |
|---|---|---|
| OS | Windows 11 Home 26200 | Ubuntu 24.04 (WSL2) |
| Python | 3.11.9 | 3.12.x |
| PyTorch | 2.11.0+cu128 | 2.5.1+cu121 |
| CUDA Toolkit | 12.8 (via wheel) | 12.1 (native nvcc) |
| GPU | NVIDIA RTX 3050 4 GB | NVIDIA RTX 3050 4 GB (via WSL2) |
| NVIDIA driver | 581.83 | 581.83 (host) |
| mamba-ssm | N/A | 2.2.6.post3 (pinned `<2.3`) |
| causal-conv1d | N/A | rebuilt against torch 2.5 |
| transformers | N/A | `<5` (pinned for compatibility) |
| Multi-seed values | n/a | {42, 123, 456, 789, 2024} |

### Reference hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 3050 (4 GB VRAM — the bottleneck) |
| CPU | Intel Core i5-12450H (8 cores, 12 threads) |
| RAM | 40 GB |

The VRAM constraint is what motivates mixed precision (FP16), `batch_size = 16` and gradient checkpointing for Mamba.

**On determinism:** `set_seed` fixes the Python, NumPy, PyTorch and CUDA seeds on every run. cuDNN's deterministic mode is available via `experiment.deterministic: true` in the config but was **not** enabled for the reported runs, so results are reproducible up to cuDNN autotuning noise rather than bit-for-bit. Run-to-run variation is reported through the multi-seed spread instead.

---

## Project status

**Modeling, training, XAI and evaluation — complete.**

- **Baselines:** stratified random, catalog LogReg, single-branch CNN, dual-branch AstroNet.
- **Main model:** single-view Mamba, locked run plus a 5-seed sweep and ensemble.
- **Protocol:** TIC-level splits (70/15/15), programmatically sealed test set, multi-seed reporting in place of k-fold.
- **Metrics:** AUC-ROC, AUC-PR, F1, recall, precision, Brier; ROC and PR curves; confusion matrix; calibration.
- **Inferential statistics:** stratified bootstrap CIs (2,000 resamples) and DeLong tests via the fast Sun & Xu algorithm.
- **Negative ablation:** a naive Mamba+CNN late fusion (ExoMamba V1) that collapses below chance, reported rather than hidden.
- **Error analysis:** top FN/FP, `y_prob` histograms by class, error rate against physical features.
- **XAI:** gradient saliency, integrated gradients and occlusion sensitivity on 8 cases.
- **Reproducibility:** versioned YAML configs, fixed seeds, `env_info.txt` + `git_info.txt` per run, 73 automated tests, CI.

**LLM vetting agent — built and running.** A conversational assistant that uses the trained Mamba as a tool through a FastAPI microservice (Docker, CUDA), with physical plausibility checks, contrast against the official NASA disposition, and session logs as auditable evidence. See `agent/README.md`. Its figure-generation and explanation tools currently return illustrative synthetic figures, clearly labelled as such inside the images themselves; the real attribution pipeline is `scripts/run_xai.py`.

**Known gaps**, documented rather than papered over:

- The task is CP-vs-FP, not the operationally relevant PC-vs-FP triage. See `docs/pc_fp_expansion.md`.
- No transfer learning from Kepler, and no external validation on an independent catalog.
- The agent's evaluation metrics were measured in deterministic mock mode; a run against a live LLM is still pending.

---

## Citation

If you use this code or its results, please cite the software (see `CITATION.cff`):

```bibtex
@software{zumbado_aguilar_exomamba_2026,
    title   = {ExoMamba: Selective State-Space Models for TESS Exoplanet Vetting},
    author  = {Zumbado Ruiz, Jos\'e Fabi\'an and Aguilar Villanueva, Jeremmy},
    year    = {2026},
    version = {0.1.0},
    url     = {https://github.com/JoseZum/ExoMamba}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
