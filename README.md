# DeepSV — Neural Networks for Detecting Deletions in Genomic Data

Big Mathematical Workshop project: investigating and extending **DeepSV**
(Cai *et al.*, *BMC Bioinformatics* 2019, doi:[10.1186/s12859-019-3299-y](https://doi.org/10.1186/s12859-019-3299-y);
base repo: [CSuperlei/DeepSV](https://github.com/CSuperlei/DeepSV)) for detecting
**deletions** — a class of structural variant — from aligned short-read pileups.

DeepSV encodes the aligned reads around a candidate site as a multi-channel
image and classifies *deletion* vs *non-deletion* with a VGG-style CNN. This
project reproduces that idea and runs three controlled investigations.

## Three investigations

| # | Question | Result |
|---|----------|--------|
| **1** | Transformer vs. CNN architecture | **CNN wins.** F1 = 0.959 ± 0.004 (AUROC 0.995) vs. Transformer F1 = 0.918 ± 0.014 (AUROC 0.980). The convolutional inductive bias matches the local-texture deletion signal. |
| **2** | Do external biological features help? | **No (null result).** Fusing engineered features with the pileup image gives Δ F1 ≈ 0.000 in the standard regime and −0.011 in a hard regime — the features are redundant with the image. |
| **3** | Cross-species transfer (human → crop) | **Transfers for human-like genomes, needs fine-tuning for repeat-rich ones.** Rice zero-shot F1 = 0.963 (≈ crop-native). Maize zero-shot F1 = 0.884 vs. 0.922 native; fine-tuning on 400 maize loci recovers F1 = 0.920. |
| **4** | **Efficiency optimization** (the workshop's core aim) | **8.4× smaller, 3.0× faster, −0.024 F1.** A depthwise-separable redesign (`EfficientDeepSVCNN`) reaches F1 = 0.945 vs. 0.969 baseline with **54k vs. 453k params** (226 KB vs. 1.8 MB), **3.0× inference throughput** (788 vs. 263 loci/s, CPU) and 3.9× faster training. |

**Recommendation:** keep the CNN, skip feature engineering, deploy zero-shot on
human-like genomes but fine-tune for repeat-rich genomes (e.g. maize) — and for
genome-wide scanning use the **efficiency-optimized depthwise-separable CNN**,
which trades a fraction of a percent of F1 for a 3× throughput gain.

## Quick start — run the trained model

```bash
pip install -r requirements.txt
# sanity check on simulated human loci:
python predict.py demo  --checkpoint deepsv_efficient.pt
# score your own pileup tensors (shape N x 6 x 40 x 96, saved as .npy):
python predict.py score --checkpoint deepsv_efficient.pt --input loci.npy --out calls.csv
```

The efficiency-optimized model ships in the repo as `deepsv_efficient.pt` (230 KB)
— no download or training required to try it.

## Encoding

Each candidate site → a **6-channel × 40-read × 96-position** tensor:
base identity, base-match, soft-clip, discordant pair, mapping quality, strand.
Three deletion signatures are represented: read-depth drop, discordant read
pairs, and split/soft-clipped reads at breakpoints.

## Repository layout

```
simulate.py          # read-pileup simulator; SpeciesProfile + HUMAN/RICE/MAIZE/CROPS; make_dataset
models.py            # DeepSVCNN, PileupTransformer, FusionNet, FeaturesMLP, EfficientDeepSVCNN
train_utils.py       # make_loaders, train_model, evaluate, set_seed
run_experiments.py   # experiment 1/2/3 -> results.json
run_exp4.py          # efficiency benchmark (baseline vs efficient) -> results_exp4.json
train_and_export.py  # train efficient model -> deepsv_efficient.pt
predict.py           # inference CLI: demo | score
deepsv_efficient.pt  # trained efficient model (230 KB, ready to run)
requirements.txt     # pinned package versions
results.json         # full metrics (incl. hard-regime + cross-species)
results_exp4.json    # efficiency-benchmark metrics
results_summary.csv  # one-row-per-condition summary table (all 4 experiments)
REPORT.md            # full write-up (methods, results, discussion, honesty note)
figures/             # fig1..fig5 PNGs
```

## Reproduce

```bash
pip install torch numpy scikit-learn matplotlib pandas seaborn
python run_experiments.py     # ~25 min on CPU; writes results.json
```

## Honesty note

Experiments run on a **read-pileup simulator with ground-truth labels** (real
WGS BAMs are too large for the analysis sandbox). The simulator injects
realistic difficulty — repeat artefacts, subtle/heterozygous deletions,
species-specific GC/repeat/coverage profiles — so the task is not trivially
separable. All **architectures, training, and metrics are genuine**; only the
input data is simulated. See `REPORT.md` §2 and §8.
