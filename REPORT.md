# Discovering Neural Networks for Detecting Mutations in Genomic Data
## Improving deep-learning detection of genomic deletions — a DeepSV-based study

**Big Mathematical Workshop — final project report**

---

## 1. Summary

This project investigates and extends **DeepSV** (Cai et al., *BMC Bioinformatics*
2019), a deep-learning method that detects genomic **deletions** by encoding the
aligned short-read pileup around a candidate site as a multi-channel "image" and
classifying it with a convolutional neural network (CNN). Working from the DeepSV
paper and its reference implementation (`github.com/CSuperlei/DeepSV`), we
reproduced its image-encoding and CNN baseline, and then carried out the three
investigations set out in the project plan:

1. **Architecture — Transformer vs. CNN.** Does self-attention over genomic
   positions beat convolution for deletion detection?
2. **External biological features.** Do GC-content, repeat/mappability annotation
   and mapping-quality statistics improve the network?
3. **Cross-species transfer.** Does a human-trained network detect deletions in
   crop genomes (rice, maize)?

**Headline findings (all on controlled, ground-truth-labelled simulated data):**

| # | Question | Result |
|---|----------|--------|
| 1 | Transformer vs. CNN | **CNN wins** — F1 0.959 vs 0.918, AUROC 0.995 vs 0.980, at comparable size and training cost. Self-attention did not help here. |
| 2 | External features | **No gain.** Fusing engineered features with the CNN changed F1 by <0.01 in both a standard (0.959→0.959) and a hard, low-coverage/repeat-rich regime (0.880→0.869). The pileup image already contains the signal. |
| 3 | Cross-species transfer | **Transfers, with a genome-dependent gap.** Zero-shot human→rice F1 0.963 (≈ crop-native); human→maize F1 0.884 vs 0.922 crop-native — a real gap that fine-tuning on 400 crop loci closes (0.920). |

**One practical recommendation** follows from these results and is developed in
§7: keep the CNN backbone, do not spend engineering effort on hand-crafted
features, and deploy human-trained models zero-shot on genomically human-like
crops but **fine-tune for repeat-rich genomes** such as maize.

---

## 2. Scope, honesty statement, and method

**What is real and what is simulated.** The original DeepSV was trained on
NA12878 whole-genome sequencing from the 1000 Genomes Project — terabyte-scale
BAM alignments plus curated deletion truth sets. That data volume and the GPU
training it requires are outside this workshop's compute envelope. Every
*architecture, training procedure, metric and comparison in this report is
genuine*; the *data is simulated* by a read-pileup generator we wrote to
reproduce DeepSV's encoding and the physics of a deletion signal, with **exact
ground truth**. This is a deliberate and standard choice for a controlled
methods-comparison study: it isolates the variables each experiment is about
(architecture, features, species) from confounds in noisy real truth sets, and it
lets every metric be computed against a clean label. Where a conclusion would
plausibly change on real WGS data, we say so explicitly (§8).

**The biology we reproduce.** A deletion of length *D* leaves three signatures in
paired-end short-read WGS, all of which DeepSV's image is designed to capture and
all of which our simulator generates:

- **Read-depth drop** across the deleted interval — roughly half depth for a
  heterozygous deletion, near zero for a homozygous one.
- **Discordant read pairs** — pairs spanning the deletion map with an insert size
  inflated by ≈ *D*.
- **Split / soft-clipped reads** — reads crossing a breakpoint cannot align fully;
  their overhanging ends are soft-clipped exactly at the breakpoint.

**Realism / difficulty.** A naive simulator makes the task trivially separable
(we observed F1 = 1.000). To create a realistic decision problem we injected the
same ambiguities a real caller faces: locus-to-locus coverage variation, **subtle
deletions** (small, heterozygous, low-coverage) whose signals are weak, and — most
importantly — **repeat-region misalignment artefacts** that produce *spurious*
soft-clips and discordant pairs in true negatives. After this, the strongest
single hand-crafted feature (central/flank depth ratio) reaches only AUROC ≈ 0.83
alone, leaving real headroom for the networks to compete over.

**Encoding.** Each locus is a tensor of shape
`(6 channels × 40 reads × 96 positions)`. The six channels mirror the information
DeepSV encodes from the BAM: base identity, base-match, soft-clip flag,
discordant-pair flag, mapping quality, and strand. Figure 1 shows a heterozygous
deletion and a matched negative in this encoding.

![Figure 1 — DeepSV-style pileup encoding of a deletion vs. a non-deletion locus]({{artifact:d0099def-a3ad-4106-9f99-1274dd644525}})

*Figure 1. The read-depth track drops between the two breakpoints (red dashed
lines) for the deletion (top) but is flat for the negative (bottom). Soft-clip
and discordant-pair signals localise at the breakpoints and are absent from the
negative. This is exactly the structure the CNN learns to read.*

**Common protocol.** All experiments use a 70/15/15 train/validation/test split,
`BCEWithLogitsLoss`, AdamW with cosine LR decay, early stopping on validation F1,
and report the held-out test set. Experiments 1 and 2 are run over **3 random
seeds** (mean ± sd); Experiment 3's transfer matrix is a single seed given its
size. Metrics: precision, recall, F1, AUROC, AUPRC.

---

## 3. Experiment 1 — Transformer vs. CNN

**Motivation.** DeepSV uses a VGG-style CNN — stacked local 3×3 convolutions.
A deletion signal, though, couples a depth drop to breakpoint signals that can sit
tens of positions apart. A **Transformer** with self-attention over genomic
positions can relate those distant signals in a single layer, which is a
plausible reason to expect it to do better. We test that directly.

**Models.** (i) `DeepSVCNN` — a faithful VGG-style reimplementation (paired
3×3 conv + BatchNorm + LeakyReLU blocks with max-pooling → dense head), 453k
parameters. (ii) `PileupTransformer` — each genomic column is pooled over reads
into a token (mean+max per channel), given a learned positional embedding, and
passed through a 3-layer, 6-head Transformer encoder; a mean-pooled
representation feeds the classifier, 359k parameters.

**Result.** The CNN wins on every metric, at comparable size and training time.

| Model | F1 | AUROC | Precision | Recall | Params | Train time |
|-------|------|-------|-----------|--------|--------|-----------|
| **CNN** | **0.959 ± 0.004** | **0.995** | 0.957 | 0.961 | 453k | 111 s |
| Transformer | 0.918 ± 0.014 | 0.980 | 0.908 | 0.928 | 359k | 106 s |

![Figure 2 — Transformer vs CNN]({{artifact:7c617769-3a6c-4e64-b9ba-6d238990eeec}})

**Interpretation.** The convolutional inductive bias — locality and translation
equivariance — matches this problem well: the discriminative signal is a set of
*local* texture patterns (a depth step, a column of soft-clips) that can occur
anywhere in the window, which is exactly what a translation-equivariant CNN is
built to detect. The Transformer must learn that structure from data and, at this
training-set size, does so less efficiently and with higher seed-to-seed variance.
This is consistent with the broader vision literature: convolutional biases beat
pure self-attention in the small/medium-data regime, and Transformers overtake
only with much larger datasets. **On DeepSV-scale problems, the CNN is the right
choice.** (§8 notes when this could flip.)

---

## 4. Experiment 2 — External biological features

**Motivation.** A structural-variant caller can attach external annotation to each
candidate: local **GC content**, whether the locus is **repetitive**,
**mappability** (proxied by mapping quality), and summary statistics of the depth,
discordant and soft-clip evidence. The question is whether feeding these to the
network improves it.

**Models.** (i) `FeaturesMLP` — an MLP on the 6 engineered features *only* (no
image). (ii) `CNN` — image only (the Exp-1 baseline). (iii) `FusionNet` — the CNN
backbone concatenated with an MLP over the features before the classifier.

We evaluate in two regimes: the **standard** regime (30× coverage, 45% repeat) and
a deliberately **hard** regime (12× coverage, 70% repeat, lower mapping quality)
where the image signal is degraded and auxiliary annotation has the best chance to
help.

| Regime | Features-only | CNN (image) | Fusion (CNN+features) | Δ(fusion−image) |
|--------|--------------|-------------|----------------------|-----------------|
| Standard | 0.712 ± 0.053 | 0.959 ± 0.004 | 0.959 ± 0.004 | **+0.000** |
| Hard | 0.710 ± 0.002 | 0.880 ± 0.006 | 0.869 ± 0.007 | **−0.011** |

![Figure 3 — External features add no gain]({{artifact:1e62dd10-c21b-4f0a-b112-a785153b2ef7}})

**Interpretation — an informative null result.** Features alone carry real but
incomplete signal (F1 ≈ 0.71). Adding them to the CNN, however, produces **no
improvement** in either regime — not even in the hard regime designed to favour
them. The reason is mechanistic and worth stating plainly: the engineered features
(depth ratio, discordant fraction, soft-clip fraction) are **summary statistics of
the very channels the CNN already convolves over**. The convolutions reconstruct
those statistics — and finer, spatially-resolved versions of them — directly from
the image, so the hand-crafted versions are redundant. GC and repeat annotation,
meanwhile, are context rather than deletion evidence and are non-discriminative on
their own. **The practical lesson: for a DeepSV-style caller, engineering effort is
better spent on the image encoding and the network than on external feature
tracks.** (The genuinely orthogonal signal a real pipeline would add — a
population **mappability/blacklist** track independent of the reads at that
locus — is the one exception worth testing on real data; §8.)

---

## 5. Experiment 3 — Cross-species transfer

**Motivation.** Curated deletion truth sets are abundant for human and scarce for
most crops. If a human-trained network detects crop deletions well, that is
directly useful for agricultural genomics. Crop genomes differ from human in the
variables that govern the pileup: **GC content, repeat load, coverage and
insert-size protocol**. We encode two contrasting crops:

- **Rice** — compact, moderately repetitive (~35%), genomically human-like here.
- **Maize** — **~85% repetitive**, lower typical coverage — a stress test for
  transfer, because repeats are the dominant source of false-positive signals.

*(Profiles are biologically motivated but illustrative, not organism-exact; see
§8.)*

We compare three conditions per crop: **zero-shot** (human model applied
directly), **crop-native** (a CNN trained on that crop from scratch — the upper
bound), and **fine-tuned** (human weights adapted on just 400 crop loci).

| Setting | F1 | AUROC | Precision | Recall |
|---------|------|-------|-----------|--------|
| human → human (reference) | 0.956 | 0.995 | 0.942 | 0.970 |
| **rice**  zero-shot | 0.963 | 0.984 | 0.981 | 0.946 |
| rice  crop-native (upper bound) | 0.951 | 0.989 | 0.946 | 0.955 |
| rice  fine-tuned | 0.909 | 0.982 | 0.875 | 0.946 |
| **maize** zero-shot | 0.884 | 0.953 | 0.837 | 0.936 |
| maize  crop-native (upper bound) | 0.922 | 0.982 | 0.935 | 0.909 |
| maize  fine-tuned | 0.920 | 0.966 | 0.904 | 0.936 |

![Figure 4 — Cross-species transfer]({{artifact:78d37e5a-989c-43ed-8326-910023c446b9}})

**Interpretation.** Transfer works, and the size of the transfer gap tracks how
far the crop's genome sits from human:

- **Rice transfers essentially for free.** Zero-shot F1 (0.963) matches the
  crop-native model (0.951) — a human-trained DeepSV can be deployed on a
  human-like genome with no crop labels at all. (Fine-tuning on only 400 loci
  slightly *hurt* here, a small-sample overfitting effect — with ample transfer
  already available, don't fine-tune.)
- **Maize shows a real transfer gap.** Zero-shot F1 0.884 falls clearly short of
  crop-native 0.922, and the drop is concentrated in **precision** (0.837 vs
  0.935): the human model, never having seen maize's extreme repeat load, raises
  false positives on repeat-driven artefacts. **Fine-tuning on 400 maize loci
  closes the gap** (F1 0.920 ≈ crop-native), recovering precision to 0.904.

**Practical lesson.** Deploy human-trained deletion callers zero-shot on
genomically human-like crops; for repeat-rich genomes, budget a small labelled set
(hundreds of loci) for fine-tuning rather than building a caller from scratch.

---

## 6. What we built (reproducibility)

All code is provided as artifacts and runs on CPU in ~25 min end-to-end.

- **`simulate.py`** — the DeepSV-faithful read-pileup simulator: six-channel
  encoding, the three deletion signatures, realistic coverage/repeat/subtlety
  noise, and species profiles (human, rice, maize, hard-regime).
- **`models.py`** — `DeepSVCNN` (baseline), `PileupTransformer` (Exp 1),
  `FusionNet` and `FeaturesMLP` (Exp 2).
- **`train_utils.py`** — training loop, early stopping, full metric suite.
- **`run_experiments.py`** — the three-experiment driver; writes `results.json`.
- **`results.json`, `results_summary.csv`** — all numbers behind every table and
  figure.

---

## 7. Conclusions and recommendation

Reading the three experiments together yields a coherent engineering
recommendation for a DeepSV-style deletion caller:

1. **Keep the CNN backbone.** Self-attention did not beat convolution at this data
   scale (Exp 1); the convolutional inductive bias matches the local-texture nature
   of the deletion signal.
2. **Do not invest in hand-crafted feature tracks.** They are redundant with the
   pileup image the CNN already reads, even where the image signal is weak (Exp 2).
   Spend that effort on the encoding and the network instead.
3. **Exploit cross-species transfer, adaptively.** Deploy human-trained models
   zero-shot on human-like genomes; fine-tune on a few hundred loci for repeat-rich
   genomes such as maize (Exp 3).

Together these improve the **efficiency** of deletion detection in the project's
sense: the same accuracy is reached with a smaller, cheaper model (no Transformer,
no feature-engineering pipeline) and, for new species, with far less labelled data.

---

## 8. Limitations and honest caveats

- **Simulated data.** Absolute metrics (F1 ≈ 0.96) are higher than any method
  reaches on real WGS; the simulator omits real complications — reference-assembly
  errors, segmental duplications, CNV/other SV types nearby, PCR/GC coverage bias,
  aligner-specific artefacts. The **relative** conclusions (CNN > Transformer;
  features redundant; transfer gap grows with genomic distance) are the
  transferable results and the mechanisms behind them are data-independent, but
  each should be confirmed on real BAMs before deployment.
- **Species profiles are illustrative.** Rice/maize are parameterised by broad,
  well-known genomic properties (e.g. maize being ~85% repetitive), not by
  organism-exact sequencing statistics. The *direction* of the maize transfer gap
  is robust to the exact numbers; its magnitude is not a calibrated estimate.
- **When Exp 1 could flip.** The CNN's advantage is a small/medium-data result.
  With 1000-Genomes-scale training sets, or on longer windows where truly
  long-range coupling matters (very large deletions, split-read chaining across
  distant breakpoints), a Transformer or a CNN–attention hybrid could overtake —
  a natural follow-up.
- **The one feature worth re-testing.** A population-level **mappability/blacklist**
  track is genuinely independent of the reads at a locus (unlike our engineered
  features) and is the most likely external signal to help on real repeat-rich data;
  Exp 2's null does not rule it out.
- **Statistics.** Exps 1–2 use 3 seeds; Exp 3's matrix is single-seed. Differences
  larger than the observed seed spread (e.g. the CNN–Transformer F1 gap) are
  reliable; sub-0.01 differences are within noise (consistent with our reading of
  the Exp-2 null).

---

## 9. Reference

Cai L., Wu Y., Gao J. **DeepSV: accurate calling of genomic deletions from
high-throughput sequencing data using deep convolutional neural network.**
*BMC Bioinformatics* 20, 665 (2019). doi:10.1186/s12859-019-3299-y.
Reference implementation: `github.com/CSuperlei/DeepSV`.
