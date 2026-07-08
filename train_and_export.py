#!/usr/bin/env python
"""Train the efficiency-optimized detector on the standard human profile and
export a portable checkpoint (deepsv_efficient.pt) for use with predict.py."""
from __future__ import annotations
import torch
from simulate import make_dataset, HUMAN
from train_utils import make_loaders, train_model, evaluate, set_seed
from models import EfficientDeepSVCNN, count_params

def main(width=32, n=1400, epochs=15, seed=0, out="deepsv_efficient.pt"):
    set_seed(seed)
    imgs, feats, labs, _ = make_dataset(n, HUMAN, seed=seed)
    tr, va, te = make_loaders(imgs, feats, labs, seed=seed)
    set_seed(seed)
    m = EfficientDeepSVCNN(width=width)
    m, _ = train_model(m, tr, va, epochs=epochs, patience=5)
    met, _, _ = evaluate(m, te)
    torch.save({"arch": "efficient", "width": width,
                "state_dict": m.state_dict(),
                "test_metrics": {k: float(v) for k, v in met.items()},
                "params": count_params(m)}, out)
    print(f"saved {out}: F1={met['f1']:.3f} AUROC={met['auc']:.3f} params={count_params(m)}")

if __name__ == "__main__":
    main()
