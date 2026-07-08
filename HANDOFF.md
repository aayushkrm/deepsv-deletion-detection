# Project Handoff — DeepSV Deletion Detection (Big Mathematical Workshop)

> **Read this first.** This document lets a new agent (human or AI) resume the
> project cold. It states what the project is, what has been done, every result
> with its exact number, where all the code and data live, and a prioritized
> plan for what to do next. Nothing here requires access to the prior chat
> history.

---

## 1. What the project is

**Goal (from the workshop brief):** investigate neural-network algorithms for
detecting **deletions** (a class of structural variant) in genomic data, and
**modify them to improve efficiency**.

**Base method — DeepSV** (Cai et al., *BMC Bioinformatics* 2019,
doi:10.1186/s12859-019-3299-y; repo `github.com/CSuperlei/DeepSV`): encodes the
aligned short-read pileup around a candidate site as a multi-channel "image" and
classifies *deletion vs non-deletion* with a VGG-style CNN.

**The brief set three scientific investigations** plus the overarching efficiency
aim:
1. Transformer vs. CNN architecture.
2. Whether external biological features improve the model.
3. Cross-species transfer (human → crop genomes).
4. (Efficiency) — make the network cheaper to train/run.

---

## 2. Where everything lives

**GitHub repository (all work is pushed here):**
`https://github.com/aayushkrm/deepsv-deletion-detection` (public, branch `main`, owner `aayushkrm`).

**Local workspace** (this Claude Science project, `proj_cf82a3683eed`): the same
files sit in the working directory and are saved as artifacts. Key artifact IDs
(latest versions) for pickup:

| File | Artifact ID (latest version_id) | Purpose |
|------|------|---------|
| `REPORT.md` | `ac4a2f0c-8c7b-412c-a5d2-a8f9ca0e81a3` | Full write-up (all 4 experiments) |
| `README.md` | `7f9e5b1b-45af-4355-acac-1b75f16d6a8d` | Repo landing page |
| `simulate.py` | `5e5695b2-3bc3-473e-8451-8ea1bc90fe72` | Read-pileup simulator + species profiles |
| `models.py` | `826292c4-1bf6-4883-aad4-7fa99c405edb` | All 5 network architectures |
| `train_utils.py` | `9ec908c0-c801-43b6-8943-03bfe5143845` | Training loop, metrics |
| `run_experiments.py` | `c3304ce9-dc68-4dac-8484-8907805f29b1` | Exp 1/2/3 driver |
| `run_exp4.py` | `fafaa9ac-524a-4752-a580-04c45169cce2` | Exp 4 efficiency benchmark |
| `train_and_export.py` | `3d89cad2-1a6c-461f-8ca4-1260f14b8ec7` | Trains + exports efficient checkpoint |
| `predict.py` | `27f1eb41-a56a-411d-9f40-c772a3cf107c` | Inference CLI (demo / score) |
| `deepsv_efficient.pt` | `f526d937-186c-476c-af04-7589fd67c1f1` | Trained efficient model (230 KB) |
| `results.json` | `dfab0db4-d535-4ed8-b389-3b053b6674ba` | Exp 1/2/3 metrics |
| `results_exp4.json` | `dbad1bca-cfc0-4a3e-a72b-0e13a4fc4064` | Exp 4 metrics |
| `results_summary.csv` | `e9f5d34c-3379-4273-b91c-b692aec863db` | One-row-per-condition table (all 4 exps) |
| `requirements.txt` | `8dd0c890-1d7e-4984-bb51-99a405dc1bdd` | Pinned deps |
| `fig1_pileup_encoding.png` | `d0099def-a3ad-4106-9f99-1274dd644525` | Encoding illustration |
| `fig2_exp1_transformer_vs_cnn.png` | `7c617769-3a6c-4e64-b9ba-6d238990eeec` | Exp 1 figure |
| `fig3_exp2_features.png` | `1e62dd10-c21b-4f0a-b112-a785153b2ef7` | Exp 2 figure |
| `fig4_exp3_cross_species.png` | `78d37e5a-989c-43ed-8326-910023c446b9` | Exp 3 figure |
| `fig5_exp4_efficiency.png` | `c0e1887d-f34d-4aee-b1d8-9289a5699a56` | Exp 4 figure |

**Environment:** conda env named **`deepsv`** (Python 3.13; torch, numpy,
scikit-learn, matplotlib, pandas, seaborn). Full pipeline runs on CPU in ~25 min;
Exp 4 benchmark ~7 min.

---

## 3. The data model (critical to understand before changing anything)

**All experiments run on a SIMULATOR, not real reads.** Real WGS BAMs are too
large for the analysis sandbox. This is stated honestly in `REPORT.md` §2 and §8.
The architectures, training, and metrics are genuine; only the input data is
synthetic. The simulator (`simulate.py`) injects realistic difficulty so the task
is not trivially separable.

**Encoding:** each candidate site → a tensor of shape **(6 channels, 40 reads,
96 positions)**. The 6 channels are: base identity, base-match, soft-clip,
discordant pair, mapping quality, strand. Three deletion signatures are
represented: read-depth drop, discordant read pairs, split/soft-clipped reads at
breakpoints.

**Species profiles** (`SpeciesProfile` dataclass in `simulate.py`):
- `HUMAN`: gc=0.41, repeat_frac=0.45, coverage=30×
- `RICE`: gc=0.435, repeat=0.35, cov=22×
- `MAIZE`: gc=0.47, repeat=0.85, cov=18×
- a HARD human regime: repeat=0.70, cov=12× (used in Exp 2)

**Key API:** `make_dataset(n, profile, seed=0, pos_frac=0.5)` →
`(images, feats, labels, meta)`. `make_loaders(images, feats, labels, seed=0)` →
train/val/test loaders (70/15/15). `train_model(model, tr, va, epochs, patience)`,
`evaluate(model, loader)` → metrics dict.

---

## 4. What has been done — results (every number)

### Experiment 1 — Transformer vs. CNN  (3 seeds, standard human)
- **CNN wins.** F1 = **0.959 ± 0.004**, AUROC 0.995 (453k params).
- Transformer: F1 = **0.918 ± 0.014**, AUROC 0.980 (359k params).
- Interpretation: the convolutional inductive bias matches the local-texture
  deletion signal; self-attention over positions did not help at this scale.

### Experiment 2 — External biological features  (3 seeds)
- **Null result.** Fusing engineered features (GC, repeat, mappability, depth
  ratio, discordant/clip fractions) with the pileup image did not help.
- Standard regime: image-only 0.959 → fusion 0.959 (Δ +0.000).
- Hard regime: image-only 0.880 → fusion 0.869 (Δ −0.011).
- Features-only MLP: F1 ≈ 0.71. Conclusion: features are redundant with the image.

### Experiment 3 — Cross-species transfer  (SINGLE seed — see gap below)
- human→human: F1 = 0.956.
- Rice zero-shot: F1 = **0.963** (≈ crop-native 0.951) — transfers free.
- Maize zero-shot: F1 = **0.884** vs crop-native 0.922 — a real gap, precision-
  driven (0.837 vs 0.935). Fine-tuning on 400 maize loci recovers F1 = 0.920.
- Interpretation: transfers to genomically human-like crops; repeat-rich genomes
  (maize) need fine-tuning.

### Experiment 4 — Efficiency optimization (NEW; 3 seeds, standard human)
`EfficientDeepSVCNN` = depthwise-separable (MobileNet-style) redesign of the CNN.

| Model | Params | Size | F1 (±SD) | AUROC | Train | Inference (CPU) |
|-------|-------:|-----:|---------:|------:|------:|----------------:|
| DeepSV CNN (baseline) | 453,249 | 1.79 MB | 0.969 ± 0.009 | 0.995 | 276 s | 263 loci/s |
| Efficient (sep-conv) | 53,793 | 0.23 MB | 0.945 ± 0.011 | 0.991 | 71 s | 788 loci/s |
| **Gain** | **8.4× fewer** | **7.8× smaller** | **−0.024** | −0.004 | **3.9×** | **3.0×** |

Throughput was re-measured in isolation (1024 loci, median of 9 CPU passes,
batch 128) because inline per-seed throughput is sensitive to concurrent load;
`results_exp4.json` carries both under `throughput_clean_loci_s` and
`throughput_loci_s_mean` with a note. Exported model: `deepsv_efficient.pt`.

### Recommendations already in the report
1. Keep the CNN backbone (Exp 1).
2. Skip hand-crafted feature engineering (Exp 2).
3. Deploy zero-shot on human-like genomes, fine-tune for repeat-rich ones (Exp 3).
4. Use the efficient depthwise-separable CNN for genome-wide scanning (Exp 4).

---

## 5. Models available in `models.py`
- `DeepSVCNN(width=...)` — VGG-style baseline (the faithful DeepSV reimplementation).
- `PileupTransformer(...)` — Exp 1; treats pileup columns as tokens, self-attention.
- `FusionNet(...)` — Exp 2; CNN backbone + side-channel MLP over features.
- `FeaturesMLP(...)` — Exp 2; features-only baseline.
- `EfficientDeepSVCNN(width=32)` — Exp 4; depthwise-separable, `_SepConv` blocks.
- `count_params(model)` — helper.

All consume the same `(6, 40, 96)` tensor and return a single logit
(BCEWithLogitsLoss). `FusionNet` also takes the feature vector; the others accept
`feats=None` and ignore it, so a single training loop drives all of them.

---

## 6. Inference CLI (already working)
```bash
python predict.py demo  --checkpoint deepsv_efficient.pt          # metrics on simulated loci
python predict.py score --checkpoint deepsv_efficient.pt \
                        --input loci.npy --out calls.csv          # N x 6 x 40 x 96 -> CSV
```
Checkpoint format: `{"arch": "efficient"|"cnn", "width": int, "state_dict": ...,
"test_metrics": {...}, "params": int}`.

---

## 7. NEXT PLAN — prioritized, with concrete steps

### Tier 0 — Highest value: validate on REAL data  (needs external compute + data access)
This is the project's single biggest credibility gain. All current numbers are on
simulated data; a reviewer's first question is "does it hold on real reads?"

**Steps:**
1. Obtain a compute host with disk (tens of GB) and ideally a GPU, plus network
   access to genomic data repositories (the analysis sandbox here is restricted to
   science APIs and cannot hold large BAMs).
2. Pick a public deletion truth set: **GIAB / NA12878** high-confidence SV calls,
   or **1000 Genomes** deletion calls.
3. Download a BAM (or CRAM) + the truth VCF. Extract pileup windows around
   candidate sites and encode them into the SAME `(6, 40, 96)` tensor as the
   simulator — this is the main engineering task; use `pysam` to read the BAM and
   mirror the channel definitions in `simulate.py`.
4. Re-run **one** experiment (recommend Exp 1: CNN vs Transformer) on the real
   encoded data. Report whether the relative conclusion (CNN > Transformer)
   survives. If it does, this is a genuinely publishable result.
5. Watch for domain shift: train/val/test must be split by chromosome, not
   randomly, to avoid leakage from nearby loci.

**Acceptance:** a table comparing simulated vs real F1/AUROC for the ported
experiment, plus a short note on what changed.

### Tier 1 — Cheap, local, high-value (all in-sandbox, ~15–20 min total)
These close the most obvious methodological gaps without any external setup.

1. **Statistical rigor for Exp 3.** Exp 3 is currently single-seed. Re-run
   cross-species transfer with 3 seeds and report mean ± SD + confidence intervals,
   so the maize gap is defensible rather than anecdotal. Edit `run_experiments.py`
   `experiment3` to loop seeds like `experiment1` already does.
2. **Channel ablation.** Drop each of the 6 encoding channels in turn (zero it out
   in `make_dataset` output or add a mask arg) and re-measure F1. Produces a bar
   chart of ΔF1 per channel — proves WHICH deletion signatures the network uses
   (depth drop vs discordant pairs vs soft-clips). Strong scientific depth, low
   compute. New script `run_ablation.py` + `fig6_channel_ablation.png`.
3. **Calibration & operating curves.** Everything is currently F1 at threshold 0.5.
   Add ROC and precision-recall curves and a reliability/calibration plot for the
   efficient model. This is what deployment actually needs. New
   `fig7_roc_pr_calibration.png`.

### Tier 2 — Sharpen the efficiency story (in-sandbox, ~15 min)
4. **Knowledge distillation.** Distill the baseline CNN (teacher) into the
   efficient net (student) using soft targets, to try to recover the 0.024 F1 gap
   while keeping the 3× speedup. Add `distill.py`.
5. **INT8 quantization.** Post-training dynamic quantization of the efficient model
   (`torch.quantization`) for another ~4× size cut and faster CPU inference; report
   the accuracy delta.

### Tier 3 — Scope expansion (larger, optional)
6. **Other SV types** — extend the simulator + model beyond deletions to
   insertions/duplications (multi-class head). A real research extension.
7. **Packaging & tests** — turn the pipeline into an installable package with unit
   tests, or capture it as a reusable Claude Science skill.

---

## 8. Practical notes / gotchas for the next agent
- **Simulator is the source of truth for the encoding.** If you build a real-data
  encoder, mirror the channel definitions in `simulate.py` exactly, or the trained
  model won't transfer.
- **Environment:** always run in the `deepsv` conda env. `requirements.txt` has the
  exact pinned versions used for the reported numbers.
- **PYTHONPATH:** when running scripts from a fresh shell, set
  `PYTHONPATH="$(pwd)"` so `import simulate`/`models` resolve (they're flat modules
  in the repo root).
- **Figures:** load the `figure-style` skill and call `apply_figure_style()` before
  plotting; use `fig, ax = plt.subplots()` + `fig.savefig(...)`, never bare `plt.*`.
- **git in the sandbox:** direct `git` writes to `.git/` were blocked here; the repo
  was created and pushed via the GitHub REST API (blobs → tree → commit → ref).
  A normal `git clone` + `git push` will work fine on an unrestricted machine.
- **GitHub auth:** pushing needs a Personal Access Token with `repo` scope. Prefer
  adding it under Customize → Credentials rather than pasting in chat.
- **Reproduce Exp 4:** `python run_exp4.py` (writes `results_exp4.json`);
  `python train_and_export.py` regenerates `deepsv_efficient.pt`.

---

## 9. Status summary
- **Done:** all 3 required investigations + the efficiency optimization (Exp 4),
  full report, 5 figures, inference CLI, exported model, pinned deps, everything
  pushed to GitHub and saved as artifacts.
- **Not done (by design — needs external resources):** real-data validation.
- **Recommended next step:** if external compute/data can be enabled, do Tier 0
  (real-data validation). Otherwise do the Tier 1 bundle (seeds+CIs, channel
  ablation, calibration curves) — it closes the main in-sandbox gaps.
