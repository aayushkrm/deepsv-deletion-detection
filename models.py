"""
Neural network architectures for deletion detection.

Three models, all consuming the same (N_CHANNELS, MAX_READS, WINDOW) pileup:

  DeepSVCNN     -- faithful VGG-style CNN reimplementation of DeepSV
                   (repeated 3x3 Conv+BN+LeakyReLU blocks + maxpool -> dense),
                   the baseline the project starts from.

  PileupTransformer -- Experiment 1. Treats each pileup COLUMN (a genomic
                   position, pooled over reads) as a token and applies a
                   Transformer encoder with self-attention over positions, so
                   the model can relate a depth drop to distant breakpoint /
                   discordant signals directly rather than through a stack of
                   local convolutions.

  FusionNet     -- Experiment 2. The CNN backbone plus a side-channel MLP over
                   engineered biological features (GC, repeat, mappability,
                   depth ratio, discordant/clip fractions), fused before the
                   classifier.

All return a single logit; trained with BCEWithLogitsLoss.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from simulate import N_CHANNELS, MAX_READS, WINDOW


# ---------------------------------------------------------------------------
# Baseline: DeepSV-style CNN
# ---------------------------------------------------------------------------
def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.1, inplace=True),
    )


class DeepSVCNN(nn.Module):
    """VGG-style CNN following the DeepSV architecture (scaled to our image
    size and CPU budget: 4 conv stages of paired 3x3 convs + maxpool)."""
    def __init__(self, width=32, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(N_CHANNELS, width),
            _conv_block(width, width),
            nn.MaxPool2d(2),                      # 64x128 -> 32x64
            _conv_block(width, width * 2),
            _conv_block(width * 2, width * 2),
            nn.MaxPool2d(2),                      # -> 16x32
            _conv_block(width * 2, width * 4),
            _conv_block(width * 4, width * 4),
            nn.MaxPool2d(2),                      # -> 8x16
            _conv_block(width * 4, width * 4),
            nn.Dropout2d(dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x, feats=None):
        return self.head(self.features(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# Experiment 1: Transformer over pileup columns
# ---------------------------------------------------------------------------
class PileupTransformer(nn.Module):
    """Self-attention over genomic positions.

    Each of the WINDOW columns is summarized into a token by pooling the
    per-read channel activations (mean + max over the read axis), projected to
    d_model, given a learned positional embedding, and passed through a
    Transformer encoder. A CLS-style mean pool feeds the classifier.
    """
    def __init__(self, d_model=96, nhead=6, layers=3, dropout=0.2):
        super().__init__()
        # per-column input = mean and max over reads, per channel -> 2*N_CHANNELS
        self.token_proj = nn.Sequential(
            nn.Linear(2 * N_CHANNELS, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )
        self.pos = nn.Parameter(torch.zeros(1, WINDOW, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, x, feats=None):
        # x: (B, C, R, W) -> tokens (B, W, 2C)
        mean_r = x.mean(dim=2)                    # (B, C, W)
        max_r = x.amax(dim=2)                     # (B, C, W)
        tok = torch.cat([mean_r, max_r], dim=1)   # (B, 2C, W)
        tok = tok.transpose(1, 2)                 # (B, W, 2C)
        h = self.token_proj(tok) + self.pos
        h = self.encoder(h)
        h = self.norm(h).mean(dim=1)              # mean-pool positions
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Experiment 2: CNN + engineered biological features
# ---------------------------------------------------------------------------
class FusionNet(nn.Module):
    """DeepSV CNN backbone fused with an MLP over external biological features."""
    def __init__(self, width=32, n_feat=6, dropout=0.3):
        super().__init__()
        base = DeepSVCNN(width=width, dropout=dropout)
        self.features = base.features
        self.cnn_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(width * 4, 128), nn.ReLU(inplace=True),
        )
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 32), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 + 32, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x, feats):
        c = self.cnn_head(self.features(x))
        f = self.feat_mlp(feats)
        return self.classifier(torch.cat([c, f], dim=1)).squeeze(-1)


class FeaturesMLP(nn.Module):
    """Baseline using ONLY the engineered biological features (no pileup image).
    Establishes how much of the signal is captured by hand-crafted annotation."""
    def __init__(self, n_feat=6, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, feats):
        return self.net(feats).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Efficiency optimization (Exp4): depthwise-separable CNN
# ---------------------------------------------------------------------------
class _SepConv(nn.Module):
    """Depthwise-separable convolution: a depthwise 3x3 (one filter per input
    channel) followed by a pointwise 1x1 mixing convolution. Replaces a dense
    3x3 conv at ~1/9 of its multiply-accumulate cost and parameter count while
    preserving receptive field."""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class EfficientDeepSVCNN(nn.Module):
    """Efficiency-optimized deletion detector.

    Same input tensor and classification target as DeepSVCNN, but the dense
    3x3 conv stack is replaced by depthwise-separable blocks and downsampling
    is folded into strided convolutions (no separate maxpool). A single
    (rather than paired) block per stage and a lighter head cut the parameter
    count roughly an order of magnitude and speed up both training and,
    importantly for genome-wide scanning, inference throughput.
    """
    def __init__(self, width=24, dropout=0.2):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(N_CHANNELS, w, 3, padding=1, bias=False),
            nn.BatchNorm2d(w), nn.LeakyReLU(0.1, inplace=True),
        )
        self.features = nn.Sequential(
            _SepConv(w, w * 2, stride=2),      # 64x96 -> 32x48
            _SepConv(w * 2, w * 2),
            _SepConv(w * 2, w * 4, stride=2),  # -> 16x24
            _SepConv(w * 4, w * 4),
            _SepConv(w * 4, w * 4, stride=2),  # -> 8x12
            nn.Dropout2d(dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(w * 4, 1),
        )

    def forward(self, x, feats=None):
        return self.head(self.features(self.stem(x))).squeeze(-1)
