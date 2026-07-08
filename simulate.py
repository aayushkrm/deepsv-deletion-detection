"""
DeepSV-faithful read-pileup simulator for deletion detection.

Background
----------
DeepSV (Cai et al., BMC Bioinformatics 2019, s12859-019-3299-y) detects genomic
deletions by encoding the aligned short-read pileup around a candidate site as a
multi-channel "image" and classifying it with a deep CNN. The three physical
signatures of a deletion in paired-end short-read WGS are:

  (1) READ-DEPTH DROP  -- coverage falls across the deleted interval
                          (heterozygous ~ half depth, homozygous ~ zero).
  (2) DISCORDANT PAIRS -- read pairs spanning the deletion map with an insert
                          size inflated by ~ the deletion length D.
  (3) SPLIT / SOFT-CLIP -- reads crossing a breakpoint cannot align fully; their
                          overhanging ends are soft-clipped at the breakpoint.

This module simulates paired-end reads over a reference window under a known
ground truth (deletion present/absent, position, length, zygosity), reproducing
those three signals, and encodes each locus as a DeepSV-style tensor. Because
ground truth is exact, every downstream metric (precision, recall, F1, AUC) is
computed against a clean label.

The simulator is deliberately transparent: no BAM / aligner dependency, fully
reproducible from a seed, and light enough to train real CNN / Transformer
networks on CPU.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Encoding constants
# ---------------------------------------------------------------------------
BASES = "ACGT"
WINDOW = 96           # positions (columns) in the pileup image
MAX_READS = 40        # rows (reads) in the pileup image
READ_LEN = 24         # bp per read (scaled to window)
# channel layout of the per-read pileup tensor
CH_BASE = 0           # normalized base identity where a read covers a position
CH_MATCH = 1          # 1 if base matches reference, 0 if mismatch
CH_SOFTCLIP = 2       # 1 where a read is soft-clipped (breakpoint signal)
CH_DISCORDANT = 3     # 1 if the read belongs to a discordant (large-insert) pair
CH_MAPQ = 4           # normalized mapping quality
CH_STRAND = 5         # read strand (0/1)
N_CHANNELS = 6


@dataclass
class SpeciesProfile:
    """Sequencing / genome characteristics that differ between organisms.

    Human vs. crop genomes differ in GC content, repeat/mappability structure,
    coverage and insert-size distribution -- exactly the variables that govern
    whether a human-trained network transfers.  This object parameterizes them.
    """
    name: str = "human"
    gc: float = 0.41                 # mean GC fraction of the reference
    repeat_frac: float = 0.45        # fraction of genome that is repetitive
    coverage: float = 30.0           # mean read depth
    insert_mu: float = 350.0         # mean fragment (insert) size, bp
    insert_sigma: float = 60.0       # sd of fragment size
    mapq_mean: float = 55.0          # mean mapping quality in unique regions
    mapq_repeat: float = 12.0        # mean mapping quality inside repeats
    seq_error: float = 0.01          # per-base sequencing error rate


@dataclass
class Locus:
    """A single simulated candidate locus and its ground truth."""
    image: np.ndarray                # (N_CHANNELS, MAX_READS, WINDOW)
    label: int                       # 1 = deletion, 0 = no deletion
    features: np.ndarray             # engineered biological features (Experiment 2)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Species profiles. Crop genomes differ from human in GC content, repeat load,
# coverage and insert-size protocols -- the variables governing whether a
# human-trained deletion caller transfers. Values reflect broad, well-known
# genomic differences (e.g. maize ~85% repetitive, rice higher GC, lower typical
# WGS coverage in large plant-breeding panels); they are illustrative, not
# organism-exact.
# ---------------------------------------------------------------------------
HUMAN = SpeciesProfile(name="human", gc=0.41, repeat_frac=0.45, coverage=30.0,
                       insert_mu=350.0, insert_sigma=60.0,
                       mapq_mean=55.0, mapq_repeat=12.0, seq_error=0.01)
RICE = SpeciesProfile(name="rice", gc=0.435, repeat_frac=0.35, coverage=22.0,
                      insert_mu=300.0, insert_sigma=70.0,
                      mapq_mean=50.0, mapq_repeat=11.0, seq_error=0.012)
MAIZE = SpeciesProfile(name="maize", gc=0.47, repeat_frac=0.85, coverage=18.0,
                       insert_mu=380.0, insert_sigma=90.0,
                       mapq_mean=42.0, mapq_repeat=8.0, seq_error=0.013)
CROPS = {"rice": RICE, "maize": MAIZE}


def _make_reference(rng, gc):
    """Sample a reference sequence of length WINDOW with a given GC fraction."""
    p_gc = gc / 2.0
    p_at = (1.0 - gc) / 2.0
    probs = [p_at, p_gc, p_gc, p_at]  # A C G T
    idx = rng.choice(4, size=WINDOW, p=probs)
    return idx


def simulate_locus(rng, prof: SpeciesProfile, is_deletion: bool):
    """Simulate one locus. Returns a Locus with a DeepSV-style pileup image,
    ground-truth label, and engineered biological features."""

    # ---- reference & local genomic context ---------------------------------
    # A locus is repetitive with probability repeat_frac; repeats depress mapq
    # and, in the negatives, are the main source of false-positive-like noise.
    is_repeat = rng.random() < prof.repeat_frac
    ref = _make_reference(rng, prof.gc)
    gc_local = float(np.mean((ref == 1) | (ref == 2)))
    base_mapq = prof.mapq_repeat if is_repeat else prof.mapq_mean

    # ---- deletion ground truth ---------------------------------------------
    # Realism: deletions vary in size and zygosity, and many are SUBTLE
    # (small, heterozygous, in low/uneven coverage) so their depth drop and
    # breakpoint signals partly overlap the non-deletion distribution.
    if is_deletion:
        del_len = int(rng.integers(12, 60))                 # in window units
        del_start = int(rng.integers(WINDOW // 5, WINDOW // 2))
        del_end = min(del_start + del_len, WINDOW - 4)
        zygosity = rng.choice([0.5, 1.0], p=[0.65, 0.35])   # het / hom
        # fraction of breakpoint-spanning reads that actually yield a
        # detectable soft-clip / discordant tag (mapper sensitivity)
        bp_sensitivity = float(rng.uniform(0.25, 0.8))
    else:
        del_start = del_end = -1
        zygosity = 0.0
        bp_sensitivity = 0.0

    image = np.zeros((N_CHANNELS, MAX_READS, WINDOW), dtype=np.float32)

    # ---- coverage model ----------------------------------------------------
    # Coverage varies locus-to-locus; repeats are noisier. This spread means a
    # random low-coverage dip in a negative can mimic a weak depth drop.
    depth_scale = prof.coverage / 30.0
    local_cov = rng.normal(0.8, 0.18 if not is_repeat else 0.28)
    local_cov = float(np.clip(local_cov, 0.35, 1.0))
    n_reads = min(MAX_READS, int(rng.poisson(MAX_READS * local_cov * depth_scale)))

    # In repeat regions, MISALIGNMENT produces spurious soft-clips and
    # discordant pairs even without any deletion -- the dominant source of
    # false positives that a caller must learn to discount.
    artifact_rate = 0.0
    if is_repeat:
        artifact_rate = float(rng.uniform(0.0, 0.35))

    strand_of_row = rng.integers(0, 2, size=MAX_READS)
    discordant_count = 0
    clipped_count = 0

    for r in range(n_reads):
        # fragment / insert size for this read pair
        insert = rng.normal(prof.insert_mu, prof.insert_sigma)
        # allow reads to originate just off either edge so coverage is uniform
        # across the window (no read-length edge artefact in the depth track)
        raw_start = int(rng.integers(-READ_LEN + 1, WINDOW))
        start = max(0, raw_start)
        end = min(WINDOW, raw_start + READ_LEN)
        if end - start < 5:
            continue
        covers_del = is_deletion and not (end <= del_start or start >= del_end)

        # In the deleted interval, coverage is suppressed by zygosity: for a
        # homozygous deletion the deleted allele contributes no reads.
        if covers_del and rng.random() < zygosity:
            # this read would fall on the deleted allele -> drop it (depth drop)
            continue

        strand = strand_of_row[r]
        mapq = max(0.0, rng.normal(base_mapq, 8.0))

        clip_mask = np.zeros(WINDOW, dtype=bool)
        is_discordant = False

        # --- true deletion breakpoint signals (subject to mapper sensitivity)
        if is_deletion and start < del_start and end > del_start and rng.random() < 0.5 * bp_sensitivity:
            is_discordant = True
            discordant_count += 1
        if is_deletion and (start < del_start < end) and rng.random() < bp_sensitivity:
            clip_mask[del_start:end] = True
            clipped_count += 1
        if is_deletion and (start < del_end < end) and rng.random() < bp_sensitivity:
            clip_mask[start:del_end] = True

        # --- spurious artefact signals in repeats (false-positive noise)
        if artifact_rate > 0 and rng.random() < artifact_rate:
            ap = int(rng.integers(start, end))
            if rng.random() < 0.5:
                clip_mask[ap:end] = True
                clipped_count += 1
            else:
                is_discordant = True
                discordant_count += 1

        for pos in range(start, end):
            # base identity, with sequencing error / mismatch
            true_base = ref[pos]
            if rng.random() < prof.seq_error:
                obs_base = rng.integers(0, 4)
            else:
                obs_base = true_base
            match = 1.0 if obs_base == true_base else 0.0

            image[CH_BASE, r, pos] = (obs_base + 1) / 4.0
            image[CH_MATCH, r, pos] = match
            image[CH_SOFTCLIP, r, pos] = 1.0 if clip_mask[pos] else 0.0
            image[CH_DISCORDANT, r, pos] = 1.0 if is_discordant else 0.0
            image[CH_MAPQ, r, pos] = mapq / 60.0
            image[CH_STRAND, r, pos] = float(strand)

    # ---- engineered biological features (Experiment 2) ---------------------
    # These summarize external annotation a caller could attach to a candidate.
    coverage_track = image[CH_BASE].astype(bool).sum(axis=0)  # reads per column
    central = coverage_track[WINDOW // 4: 3 * WINDOW // 4]
    flank = np.concatenate([coverage_track[:WINDOW // 4], coverage_track[3 * WINDOW // 4:]])
    depth_ratio = (central.mean() + 1e-6) / (flank.mean() + 1e-6)   # depth-drop signal
    features = np.array([
        gc_local,                                   # GC content
        float(is_repeat),                           # repeat annotation
        base_mapq / 60.0,                           # mappability proxy (mapq)
        depth_ratio,                                # central/flank depth ratio
        discordant_count / MAX_READS,               # discordant-pair fraction
        clipped_count / MAX_READS,                  # soft-clip fraction
    ], dtype=np.float32)

    label = 1 if is_deletion else 0
    return Locus(image=image, label=label, features=features,
                 meta=dict(is_repeat=is_repeat, zygosity=zygosity,
                           del_start=del_start, del_end=del_end,
                           species=prof.name))


def make_dataset(n, prof: SpeciesProfile, seed=0, pos_frac=0.5):
    """Generate a labelled dataset of n loci for a given species profile."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n, N_CHANNELS, MAX_READS, WINDOW), dtype=np.float32)
    feats = np.zeros((n, 6), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    meta = []
    for i in range(n):
        is_del = rng.random() < pos_frac
        loc = simulate_locus(rng, prof, is_del)
        images[i] = loc.image
        feats[i] = loc.features
        labels[i] = loc.label
        meta.append(loc.meta)
    return images, feats, labels, meta
