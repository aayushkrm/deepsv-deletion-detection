"""
Experiment 4 -- Efficiency optimization.

The Big Mathematical Workshop brief asks specifically to *improve the
efficiency* of the deletion-detection network. This experiment introduces
EfficientDeepSVCNN, a depthwise-separable-convolution redesign of the DeepSV
CNN, and benchmarks it head-to-head against the baseline over 3 seeds on the
standard human profile, reporting not just accuracy but the efficiency metrics
that matter for genome-wide scanning: parameter count, on-disk model size,
training time, and inference throughput (loci/second).
"""
from __future__ import annotations
import time, json, os
import numpy as np
import torch
from simulate import make_dataset, HUMAN
from train_utils import make_loaders, train_model, evaluate, set_seed
from models import DeepSVCNN, EfficientDeepSVCNN, count_params

SEEDS = [0, 1, 2]
N = 1400
EPOCHS = 15
PATIENCE = 5


def infer_throughput(model, images, batch=128, reps=5):
    model.eval()
    X = torch.from_numpy(images)
    times = []
    with torch.no_grad():
        _ = model(X[:batch])                      # warmup
        for _ in range(reps):
            t = time.perf_counter()
            for i in range(0, len(X), batch):
                _ = model(X[i:i + batch])
            times.append(time.perf_counter() - t)
    return len(X) / float(np.median(times))


def disk_size_kb(model):
    torch.save(model.state_dict(), "_tmp.pt")
    kb = os.path.getsize("_tmp.pt") / 1024.0
    os.remove("_tmp.pt")
    return kb


def agg(runs, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[k + "_mean"] = float(np.mean(vals))
        out[k + "_std"] = float(np.std(vals))
    out["runs"] = runs
    return out


def run():
    models = {"CNN_baseline": lambda: DeepSVCNN(),
              "Efficient_sepconv": lambda: EfficientDeepSVCNN(width=32)}
    results = {}
    for name, ctor in models.items():
        runs = []
        for seed in SEEDS:
            set_seed(seed)
            imgs, feats, labs, _ = make_dataset(N, HUMAN, seed=seed)
            tr, va, te = make_loaders(imgs, feats, labs, seed=seed)
            set_seed(seed)
            m = ctor()
            params = count_params(m)
            t0 = time.perf_counter()
            m, hist = train_model(m, tr, va, epochs=EPOCHS, patience=PATIENCE)
            train_s = time.perf_counter() - t0
            met, _, _ = evaluate(m, te)
            thr = infer_throughput(m, imgs[:512])
            size_kb = disk_size_kb(m)
            runs.append(dict(seed=seed, params=params, train_time_s=train_s,
                             throughput_loci_s=thr, size_kb=size_kb,
                             epochs_run=len(hist), **{k: float(v) for k, v in met.items()}))
            print(f"[{name}] seed{seed} F1={met['f1']:.3f} AUC={met['auc']:.3f} "
                  f"params={params} train={train_s:.0f}s thr={thr:.0f} loci/s size={size_kb:.0f}KB",
                  flush=True)
        results[name] = agg(runs, ["f1", "auc", "auprc", "precision", "recall",
                                   "accuracy", "params", "train_time_s",
                                   "throughput_loci_s", "size_kb", "epochs_run"])
    with open("results_exp4.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    r = run()
    b, e = r["CNN_baseline"], r["Efficient_sepconv"]
    print("\n=== Efficiency summary (mean over seeds) ===")
    print(f"baseline : F1={b['f1_mean']:.3f} params={b['params_mean']:.0f} "
          f"train={b['train_time_s_mean']:.0f}s thr={b['throughput_loci_s_mean']:.0f} loci/s "
          f"size={b['size_kb_mean']:.0f}KB")
    print(f"efficient: F1={e['f1_mean']:.3f} params={e['params_mean']:.0f} "
          f"train={e['train_time_s_mean']:.0f}s thr={e['throughput_loci_s_mean']:.0f} loci/s "
          f"size={e['size_kb_mean']:.0f}KB")
    print(f"gains: {b['params_mean']/e['params_mean']:.1f}x fewer params, "
          f"{e['throughput_loci_s_mean']/b['throughput_loci_s_mean']:.1f}x throughput, "
          f"{b['train_time_s_mean']/e['train_time_s_mean']:.1f}x faster train, "
          f"dF1={e['f1_mean']-b['f1_mean']:+.3f}")
