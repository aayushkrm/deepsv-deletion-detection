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

**Recommendation:** keep the CNN, skip feature engineering, deploy zero-shot on
human-like genomes but fine-tune for repeat-rich genomes (e.g. maize).

## Encoding

Each candidate site → a **6-channel × 40-read × 96-position** tensor:
base identity, base-match, soft-clip, discordant pair, mapping quality, strand.
Three deletion signatures are represented: read-depth drop, discordant read
pairs, and split/soft-clipped reads at breakpoints.

## Repository layout

```
simulate.py         # read-pileup simulator; SpeciesProfile + HUMAN/RICE/MAIZE/CROPS; make_dataset
models.py           # DeepSVCNN, PileupTransformer, FusionNet, FeaturesMLP, count_params
train_utils.py      # make_loaders, train_model, evaluate, set_seed
run_experiments.py  # experiment1/2/3 -> results.json
results.json        # full metrics (incl. hard-regime + cross-species)
results_summary.csv # one-row-per-condition summary table
REPORT.md           # full write-up (methods, results, discussion, honesty note)
figures/            # fig1..fig4 PNGs
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
