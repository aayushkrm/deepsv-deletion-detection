"""
Master experiment runner for the three project investigations:

  EXP1  Transformer vs. CNN architecture for deletion detection.
  EXP2  Does adding external biological features improve the network?
  EXP3  Cross-species transfer: human-trained network on crop deletions.

Results are written incrementally to results.json.
"""
from __future__ import annotations
import json, time, copy
import numpy as np
import torch
from simulate import make_dataset, HUMAN, CROPS
from models import DeepSVCNN, PileupTransformer, FusionNet, FeaturesMLP, count_params
from train_utils import make_loaders, train_model, evaluate, set_seed

torch.set_num_threads(4)
SEEDS = [0, 1, 2]
N = 1400
EPOCHS = 10
PATIENCE = 4
RESULTS = {"config": dict(seeds=SEEDS, n=N, epochs=EPOCHS,
                          human=HUMAN.__dict__,
                          crops={k: v.__dict__ for k, v in CROPS.items()})}


def dump():
    with open("results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)


def build(prof, seed):
    imgs, feats, labels, meta = make_dataset(N, prof, seed=100 + seed)
    return make_loaders(imgs, feats, labels, batch=64, seed=seed)


def run(model_fn, prof, seed, epochs=EPOCHS, lr=1e-3):
    set_seed(seed)
    tr, va, te = build(prof, seed)
    m = model_fn()
    t0 = time.time()
    m, hist = train_model(m, tr, va, epochs=epochs, lr=lr, patience=PATIENCE)
    dt = time.time() - t0
    res, y, p = evaluate(m, te)
    res["train_time_s"] = round(dt, 1)
    res["params"] = count_params(m)
    res["epochs_run"] = len(hist)
    return m, res, (tr, va, te)


def agg(runs):
    """Aggregate a list of metric dicts -> mean/std per metric."""
    keys = ["precision", "recall", "f1", "auc", "auprc", "accuracy",
            "train_time_s", "params", "epochs_run", "tp", "fp", "fn", "tn"]
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        out[k + "_mean"] = float(np.mean(vals))
        out[k + "_std"] = float(np.std(vals))
    out["runs"] = runs
    return out


# ===========================================================================
# EXPERIMENT 1 : Transformer vs CNN
# ===========================================================================
def experiment1():
    print("=== EXP1: Transformer vs CNN ===", flush=True)
    exp = {}
    for name, fn in [("CNN", DeepSVCNN), ("Transformer", PileupTransformer)]:
        runs = []
        for s in SEEDS:
            _, res, _ = run(fn, HUMAN, s)
            runs.append(res)
            print(f"  {name} seed{s}: F1={res['f1']:.3f} AUC={res['auc']:.3f} "
                  f"({res['train_time_s']}s, {res['params']:,}p)", flush=True)
        exp[name] = agg(runs)
    RESULTS["exp1_transformer_vs_cnn"] = exp
    dump()
    return exp


# ===========================================================================
# EXPERIMENT 2 : External biological features
# ===========================================================================
def experiment2(exp1=None):
    print("=== EXP2: external biological features ===", flush=True)
    exp = {}
    variants = [("FeaturesMLP", FeaturesMLP), ("CNN_image_only", DeepSVCNN),
                ("Fusion_CNN+features", FusionNet)]
    for name, fn in variants:
        # reuse EXP1 CNN results to avoid recomputation
        if name == "CNN_image_only" and exp1 is not None:
            exp[name] = exp1["CNN"]
            print(f"  {name}: reused from EXP1", flush=True)
            continue
        runs = []
        for s in SEEDS:
            _, res, _ = run(fn, HUMAN, s)
            runs.append(res)
            print(f"  {name} seed{s}: F1={res['f1']:.3f} AUC={res['auc']:.3f} "
                  f"P={res['precision']:.3f} R={res['recall']:.3f}", flush=True)
        exp[name] = agg(runs)
    RESULTS["exp2_external_features"] = exp
    dump()
    return exp


# ===========================================================================
# EXPERIMENT 3 : Cross-species transfer
# ===========================================================================
def experiment3():
    print("=== EXP3: cross-species transfer ===", flush=True)
    exp = {}
    seed = 0
    # 1) train the baseline CNN on HUMAN
    set_seed(seed)
    tr_h, va_h, te_h = build(HUMAN, seed)
    m_h = DeepSVCNN(); m_h, _ = train_model(m_h, tr_h, va_h, epochs=EPOCHS, patience=PATIENCE)
    res_hh, _, _ = evaluate(m_h, te_h)
    exp["human_on_human"] = res_hh
    print(f"  human->human: F1={res_hh['f1']:.3f} AUC={res_hh['auc']:.3f}", flush=True)

    for cname, prof in CROPS.items():
        set_seed(seed)
        imgs, feats, labels, meta = make_dataset(N, prof, seed=200)
        tr_c, va_c, te_c = make_loaders(imgs, feats, labels, batch=64, seed=seed)

        # zero-shot: human model evaluated directly on crop test set
        res_zs, _, _ = evaluate(m_h, te_c)
        # crop-native upper bound: CNN trained on crop from scratch
        set_seed(seed)
        m_c = DeepSVCNN(); m_c, _ = train_model(m_c, tr_c, va_c, epochs=EPOCHS, patience=PATIENCE)
        res_native, _, _ = evaluate(m_c, te_c)
        # fine-tune: start from human weights, adapt on a SMALL crop sample
        set_seed(seed)
        img_s, feat_s, lab_s, _ = make_dataset(400, prof, seed=201)
        tr_s, va_s, _ = make_loaders(img_s, feat_s, lab_s, batch=64, seed=seed,
                                     splits=(0.7, 0.3, 0.0))
        m_ft = copy.deepcopy(m_h)
        m_ft, _ = train_model(m_ft, tr_s, va_s, epochs=6, lr=3e-4, patience=3)
        res_ft, _, _ = evaluate(m_ft, te_c)

        exp[cname] = dict(zero_shot=res_zs, crop_native=res_native, finetuned=res_ft)
        print(f"  {cname}: zero-shot F1={res_zs['f1']:.3f} AUC={res_zs['auc']:.3f} | "
              f"native F1={res_native['f1']:.3f} | finetune F1={res_ft['f1']:.3f}", flush=True)
    RESULTS["exp3_cross_species"] = exp
    dump()
    return exp


if __name__ == "__main__":
    t0 = time.time()
    e1 = experiment1()
    experiment2(e1)
    experiment3()
    RESULTS["total_runtime_s"] = round(time.time() - t0, 1)
    dump()
    print(f"ALL DONE in {RESULTS['total_runtime_s']}s", flush=True)
