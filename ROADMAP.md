# Roadmap to Publication — DeepSV Deletion Detection

> **Purpose.** This document is the plan to take the project from its current
> state (a complete, internally-consistent study on *simulated* data) to a
> **publication-ready manuscript** with results that survive peer review. It is
> organized as phases with explicit deliverables, acceptance criteria, effort
> estimates, and the resources each phase requires. Read `HANDOFF.md` first for
> the current status, file locations, and artifact IDs.

---

## 0. Honest starting assessment

**What is already strong**
- A coherent scientific narrative: four well-posed questions (architecture,
  features, transfer, efficiency), each answered with a controlled experiment.
- Genuine ML methodology: real architectures, real training, multi-seed variance
  for 3 of 4 experiments, a full metric suite, and clean figures.
- A concrete, useful efficiency result (8.4× smaller, 3.0× faster CNN).
- Reproducible: pinned deps, exported checkpoint, inference CLI, code on GitHub.

**The one blocker to publication**
- **Every result is on a simulator.** No journal or serious venue will accept a
  genomics-methods paper whose claims are never tested on real sequencing data.
  This is the gate. Everything in Phase 1 below exists to clear it.

**Secondary gaps a reviewer will flag**
- No comparison against *existing* deletion callers (Pindel, Delly, Manta, LUMPY,
  GRIDSS, or the original DeepSV) on a common benchmark.
- Experiment 3 (cross-species) is single-seed.
- Metrics are reported at a single threshold (0.5); no ROC/PR/calibration.
- No ablation showing *which* parts of the encoding carry the signal.
- No statistical significance testing between methods.

**Realistic outcome tiers**
- *As-is:* a solid workshop report / course project. Not publishable.
- *After Phase 1–2:* a workshop paper or short methods note (e.g. a bioinformatics
  workshop, or arXiv/bioRxiv preprint).
- *After Phase 1–4:* a full methods paper suitable for a peer-reviewed venue.

---

## Phase 1 — Real-data validation  (THE GATE; ~2–4 weeks, needs compute + data)

**Objective.** Reproduce at least the headline results on a real, public deletion
truth set. Without this the paper does not exist.

**Resources required**
- A compute host with ≥100 GB disk, ≥32 GB RAM, and ideally one GPU.
- Network access to genomic data repositories (the current analysis sandbox is
  restricted to science APIs and cannot hold large BAMs — this must be enabled or
  the work moved to a cluster).
- Tools: `pysam`, `samtools`, `bcftools`; a truth set.

**Recommended datasets (pick one primary + one for cross-dataset generalization)**
1. **GIAB / HG002 (Ashkenazi son)** — the current gold standard for benchmarking
   SV calls; a curated Tier-1 high-confidence deletion set with defined
   confident regions (use only calls inside them). Illumina + long-read supported.
2. **GIAB / NA12878 (HG001)** — widely used, lots of prior comparisons.
3. **1000 Genomes Project** phase-3 deletion call set — many samples, larger but
   noisier truth.

**Steps**
1. **Data acquisition.** Download one BAM/CRAM + its confident-region BED + truth
   VCF (deletions only, SVTYPE=DEL). Restrict to a size range the encoding can
   represent (e.g. 50 bp–10 kb) and document the filter.
2. **Candidate generation.** Real callers do not score every base; they score
   *candidate* sites. Generate candidates by (a) scanning for read-depth
   drops / discordant-pair clusters / soft-clip clusters (a lightweight
   signal-based proposer), or (b) taking the union of a fast existing caller's raw
   calls. Label a candidate positive if it reciprocally overlaps a truth deletion
   (≥50% reciprocal overlap is the standard).
3. **Real-read encoder.** Build `encode_bam.py`: for each candidate locus, read the
   pileup with `pysam` and produce the SAME `(6, 40, 96)` tensor the simulator
   emits — base identity, base-match, soft-clip, discordant, mapping quality,
   strand. **This must mirror `simulate.py`'s channel definitions exactly**, or the
   model cannot transfer. This is the main engineering task of the whole project.
4. **Chromosome-disjoint splits.** Train/val/test split **by chromosome**
   (e.g. train chr1–16, val chr17–18, test chr19–22), never random — random splits
   leak information between nearby loci and inflate metrics.
5. **Re-run experiments on real data.** At minimum Exp 1 (CNN vs Transformer) and
   Exp 4 (efficient vs baseline). Report real F1/precision/recall/AUROC and compare
   directly to the simulated numbers in a single table.
6. **Class imbalance.** Real candidate sets are highly imbalanced (far more
   non-deletions). Report precision-recall (not just F1), use class weighting or
   focal loss, and state the base rate.

**Acceptance criteria**
- A table: simulated vs real F1/AUROC for Exp 1 and Exp 4.
- The relative conclusions either **hold** (CNN > Transformer; efficient ≈ baseline
  at 3× speed) — in which case you have a paper — or they **don't**, which is
  itself a publishable, honest finding about the simulation-to-reality gap.
- Splits, filters, overlap criterion, and base rate all documented.

---

## Phase 2 — Benchmark against existing callers  (~1–2 weeks, needs compute)

**Objective.** Situate the method against the tools practitioners actually use.
Reviewers will not accept a new caller with no baseline comparison.

**Steps**
1. Run 2–4 established deletion callers on the same sample/regions: **Manta**,
   **Delly**, **LUMPY**, **Pindel**, or **GRIDSS** (Manta + Delly are the common
   minimum). If feasible, run the original **DeepSV** too.
2. Evaluate all callers with a standard harness — **Truvari** (the de-facto SV
   benchmarking tool) against the GIAB truth set, with the same confident regions
   and reciprocal-overlap criterion.
3. Report a common table: precision / recall / F1 per caller, plus runtime and
   resource use. Stratify by deletion size (50–100 bp, 100 bp–1 kb, 1–10 kb) and by
   genomic context (unique vs repeat-rich) — this is where a learned method can
   show an advantage.

**Acceptance criteria**
- A benchmark table with ≥2 established callers + your two models, evaluated by
  Truvari, stratified by size and context. This table is the core of the paper.

---

## Phase 3 — Statistical rigor & scientific depth  (~1 week, mostly in-sandbox)

These sharpen the existing experiments and can begin now on simulated data, then
be repeated on real data once Phase 1 lands.

1. **Multi-seed everything.** Bring Experiment 3 (cross-species) to ≥3 seeds with
   mean ± SD, matching Exp 1/2/4. Report 95% CIs.
2. **Significance testing.** For each head-to-head claim (CNN vs Transformer;
   efficient vs baseline; zero-shot vs fine-tuned maize), run a paired test across
   seeds/folds (e.g. paired bootstrap or a permutation test on per-locus
   predictions) and report a p-value or effect size, not just point estimates.
3. **Channel ablation.** Zero each of the 6 encoding channels in turn and
   re-measure — quantifies which deletion signatures (depth drop, discordant pairs,
   soft-clips) the network relies on. Deliver `fig6_channel_ablation.png`.
4. **Operating characteristics.** ROC, precision-recall, and calibration
   (reliability) curves for the deployed model; report the precision-recall
   trade-off across thresholds, not a single F1. Deliver
   `fig7_roc_pr_calibration.png`.
5. **Error analysis.** Characterize the false negatives/positives — by deletion
   size, zygosity (het vs hom), coverage, and repeat context. A figure showing
   *where* the model fails is exactly what reviewers ask for.

**Acceptance criteria**
- Every headline comparison carries a CI and a significance statement.
- Ablation + calibration + error-analysis figures exist and are discussed.

---

## Phase 4 — Method advances (make the contribution novel)  (~2–4 weeks)

Real-data validation makes the project *sound*; these make it *novel* enough to
publish as a methods contribution rather than a reproduction.

1. **Sharpen efficiency (the workshop's own aim).**
   - *Knowledge distillation:* distill the baseline CNN into the efficient net with
     soft targets to recover the 0.024 F1 gap while keeping the 3× speedup.
   - *INT8 quantization:* post-training dynamic quantization for another ~4× size
     cut; report the accuracy delta and the deployment footprint.
   - Report throughput on GPU as well as CPU, at genome scale (loci/second and
     wall-clock for a whole-genome candidate set).
2. **Improve the encoding.** Test higher read depth (e.g. 60 reads) and wider
   windows; add a base-quality channel; try learned positional encodings. Show an
   encoding ablation.
3. **Extend beyond deletions (optional, higher ceiling).** Generalize the head to
   multi-class SV typing (deletion / insertion / duplication) — a substantially
   larger contribution.
4. **Uncertainty & confidence scores.** Emit a calibrated probability suitable for
   VCF `QUAL`, so the caller integrates into standard pipelines.

**Acceptance criteria**
- At least one clear methodological advance over vanilla DeepSV with a measured
  improvement (efficiency, accuracy in a stratum, or scope).

---

## Phase 5 — Manuscript & release engineering  (~2 weeks)

**Manuscript structure (methods-paper template)**
1. *Abstract* — problem, method, headline real-data result, availability.
2. *Introduction* — SV/deletion detection, why deep learning, gap addressed.
3. *Related work* — DeepSV and other learned callers; classical callers.
4. *Methods* — encoding, architectures (CNN / Transformer / efficient), training,
   the four experiments, datasets, evaluation protocol (Truvari, splits, overlap).
5. *Results* — real-data validation, benchmark table, the four experiments with
   CIs and significance, ablations, error analysis.
6. *Discussion* — when to use which model, the efficiency trade-off, the
   simulation-to-reality gap, limitations.
7. *Availability* — code, trained models, exact commands to reproduce.

**Reproducibility & release**
- Dockerfile / conda-lock for a one-command environment.
- A `Makefile` or Snakemake/Nextflow workflow: raw data → encoded tensors →
  trained models → all tables and figures.
- Unit tests (`pytest`) for the encoder and the metric code; CI on GitHub Actions.
- Deposit trained weights and a small example dataset on Zenodo (gets a DOI).
- A `data/README` documenting exact accessions and download commands.

**Target venues (by ambition)**
- *Preprint:* bioRxiv (immediately, once Phase 1–3 land).
- *Journal:* *BMC Bioinformatics* (the base paper's venue — natural fit),
  *Bioinformatics* (Oxford), *GigaScience*, *BMC Genomics*.
- *Conference:* RECOMB, ISMB/ECCB, or a machine-learning-for-genomics workshop
  (e.g. at NeurIPS/ICML) for the efficiency angle.

**Acceptance criteria**
- A submittable manuscript + a repository that a stranger can clone and reproduce
  every figure with one command.

---

## Critical path & dependencies

```
Phase 1 (real data) ──┬──► Phase 2 (benchmark) ──┐
                      │                          ├──► Phase 5 (manuscript)
                      └──► Phase 3 (rigor) ───────┤
                                Phase 4 (advances)─┘
```
- **Phase 1 is the gate** — Phases 2 and 4 depend on it; Phase 3 can start now on
  simulated data and be re-run on real data afterward.
- If external compute/data **cannot** be enabled, the project caps at "workshop
  report + preprint of a simulation study" — still worth doing Phase 3 in-sandbox
  and being explicit in the write-up that validation is future work.

---

## Effort & resource summary

| Phase | Effort | Needs external compute/data? | Blocking? |
|-------|--------|------------------------------|-----------|
| 1 — Real-data validation | 2–4 wk | **Yes** (host + data access) | **Yes — the gate** |
| 2 — Benchmark vs callers | 1–2 wk | Yes | For a strong paper |
| 3 — Statistical rigor | ~1 wk | No (sim); re-run on real later | No |
| 4 — Method advances | 2–4 wk | Partly (GPU helps) | No |
| 5 — Manuscript & release | ~2 wk | No | Final step |

**Fastest path to a publishable result:** Phase 1 → Phase 3 → a bioRxiv preprint,
then Phase 2 + Phase 4 → a journal submission.

---

## Immediate next actions (what to do the moment you pick this up)
1. **Decide the compute story.** Can a host with disk + GPU + data-repository
   network access be enabled? That single decision determines whether the project
   can reach publication or stays a simulation study.
2. If yes → start **Phase 1, step 3** (the real-read encoder) — it is the long pole.
3. In parallel, and regardless → do **Phase 3** items 1–4 in-sandbox now
   (seeds+CIs for Exp 3, channel ablation, calibration curves); they need no
   external resources and directly strengthen the write-up.
4. Keep `REPORT.md`, `HANDOFF.md`, and this roadmap in sync as milestones land.
