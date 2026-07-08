"""
Training / evaluation utilities shared by all experiments.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def make_loaders(images, feats, labels, batch=64, seed=0, splits=(0.7, 0.15, 0.15)):
    n = len(labels)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(splits[0] * n)
    n_va = int(splits[1] * n)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    def ds(ix):
        return TensorDataset(torch.from_numpy(images[ix]),
                             torch.from_numpy(feats[ix]),
                             torch.from_numpy(labels[ix]).float())
    return (DataLoader(ds(tr), batch_size=batch, shuffle=True),
            DataLoader(ds(va), batch_size=batch),
            DataLoader(ds(te), batch_size=batch))


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    ys, ps = [], []
    for xb, fb, yb in loader:
        logit = model(xb.to(device), fb.to(device))
        ps.append(torch.sigmoid(logit).cpu().numpy())
        ys.append(yb.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = (p >= 0.5).astype(int)
    # guard against a degenerate single-class batch
    out = dict(
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        auc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        auprc=average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        accuracy=float((pred == y).mean()),
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return out, y, p


def train_model(model, train_loader, val_loader, epochs=15, lr=1e-3,
                device="cpu", weight_decay=1e-4, patience=5, verbose=False):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()
    best_f1, best_state, wait = -1.0, None, 0
    history = []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, fb, yb in train_loader:
            opt.zero_grad()
            logit = model(xb.to(device), fb.to(device))
            loss = lossf(logit, yb.to(device))
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        sched.step()
        val, _, _ = evaluate(model, val_loader, device)
        history.append(dict(epoch=ep, train_loss=tot / len(train_loader.dataset), **val))
        if verbose:
            print(f"ep{ep:02d} loss={history[-1]['train_loss']:.3f} "
                  f"valF1={val['f1']:.3f} valAUC={val['auc']:.3f}")
        if val["f1"] > best_f1:
            best_f1, best_state, wait = val["f1"], {k: v.detach().cpu().clone()
                                                    for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
