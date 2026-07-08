#!/usr/bin/env python
"""
predict.py -- run a trained deletion detector on pileup tensors.

Two modes:
  * demo   : simulate a small labelled set and report metrics + a few calls.
  * score  : load an .npy array of shape (N, 6, 40, 96) and emit per-locus
             deletion probabilities as CSV.

Usage
-----
  python predict.py demo  --checkpoint deepsv_efficient.pt
  python predict.py score --checkpoint deepsv_efficient.pt \
                          --input loci.npy --out calls.csv --threshold 0.5

The checkpoint is a dict {"arch": "efficient"|"cnn", "width": int,
"state_dict": ...} as written by export_checkpoint() / train_and_export.py.
"""
from __future__ import annotations
import argparse, csv, sys
import numpy as np
import torch
from models import DeepSVCNN, EfficientDeepSVCNN
from simulate import N_CHANNELS, MAX_READS, WINDOW


def build_model(ckpt):
    arch = ckpt.get("arch", "efficient")
    width = ckpt.get("width", 32)
    m = EfficientDeepSVCNN(width=width) if arch == "efficient" else DeepSVCNN(width=width)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def predict_proba(model, images, batch=128):
    X = torch.from_numpy(np.asarray(images, dtype=np.float32))
    out = []
    for i in range(0, len(X), batch):
        out.append(torch.sigmoid(model(X[i:i + batch])).numpy())
    return np.concatenate(out) if out else np.array([])


def cmd_score(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt)
    imgs = np.load(args.input)
    assert imgs.shape[1:] == (N_CHANNELS, MAX_READS, WINDOW), \
        f"expected (N,{N_CHANNELS},{MAX_READS},{WINDOW}), got {imgs.shape}"
    p = predict_proba(model, imgs)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["locus_index", "deletion_prob", "call"])
        for i, prob in enumerate(p):
            w.writerow([i, f"{prob:.4f}", int(prob >= args.threshold)])
    print(f"scored {len(p)} loci -> {args.out} "
          f"({int((p >= args.threshold).sum())} called deletion)")


def cmd_demo(args):
    from simulate import make_dataset, HUMAN
    from sklearn.metrics import f1_score, roc_auc_score
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt)
    imgs, feats, labs, meta = make_dataset(args.n, HUMAN, seed=args.seed)
    p = predict_proba(model, imgs)
    pred = (p >= args.threshold).astype(int)
    print(f"demo on {args.n} simulated human loci (arch={ckpt.get('arch')}):")
    print(f"  F1={f1_score(labs, pred):.3f}  AUROC={roc_auc_score(labs, p):.3f}")
    print("  first 8 loci (true / prob / call):")
    for i in range(min(8, args.n)):
        print(f"    locus {i}: true={labs[i]} prob={p[i]:.3f} call={pred[i]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.set_defaults(fn=cmd_demo)
    d.add_argument("--checkpoint", default="deepsv_efficient.pt")
    d.add_argument("--n", type=int, default=400)
    d.add_argument("--seed", type=int, default=123)
    d.add_argument("--threshold", type=float, default=0.5)
    s = sub.add_parser("score"); s.set_defaults(fn=cmd_score)
    s.add_argument("--checkpoint", default="deepsv_efficient.pt")
    s.add_argument("--input", required=True)
    s.add_argument("--out", default="calls.csv")
    s.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
