#!/usr/bin/env python3
"""
DCGAN and conditional GAN on MNIST — the GAN part of project 9, in one file.

Four stages, selected by a sub-command:

    prepare   build the FID ruler shared by the whole group (run once)
    dcgan     train DCGAN            -> requirement 1 (architecture, loss curves)
                                     -> requirement 4 (instability, mode collapse)
    cgan      train a conditional GAN -> requirement 3 (generate a requested digit)
    compare   AE / VAE / GAN table    -> requirement 2 (FID + grids at the same epoch)
    all       prepare (if needed) -> dcgan -> cgan -> compare

    python 5_gan.py all --out-dir runs --epochs 10

ARCHITECTURE (Radford, Metz & Chintala 2016). The original GAN of Goodfellow et al.
2014 used fully connected nets; DCGAN replaces them with convolutions:

    Generator      z (100) -> Linear -> 128x7x7 -> ConvT -> 64x14x14 -> ConvT -> 1x28x28, tanh
    Discriminator  1x28x28 -> Conv -> 64x14x14 -> Conv -> 128x7x7 -> Linear -> one logit

DCGAN guidelines followed here: strided convolutions instead of pooling, BatchNorm
in both networks, no fully connected hidden layers, ReLU + tanh in G, LeakyReLU(0.2)
in D, weights from N(0, 0.02), Adam with beta1 = 0.5.

TRAINING. Each batch is two steps:

    D step   real -> label 1, fake -> label 0. The fake batch is detached, otherwise
             the same graph is back-propagated twice and D's update drags G along.
    G step   the same fakes are scored again, but G asks D to answer "real". This is
             the non-saturating loss -log D(G(z)) rather than log(1 - D(G(z))): early
             on D is almost always right, D(G(z)) ~ 0, and the saturating form has a
             vanishing gradient exactly there.

D returns a raw logit; the sigmoid lives inside BCEWithLogitsLoss.

Because the two networks optimise against each other there is no single objective
that should fall, so a decreasing loss proves nothing. Tracked per epoch instead:
D(x) and D(G(z)) (0.5 / 0.5 is the healthy balance), FID, and the class entropy of
the samples, which is the mode-collapse alarm (ln 10 = 2.303 when all ten digits
appear equally often).

Images live in [-1,1] here to match tanh; the AE/VAE scripts use [0,1] because
their decoders end in a sigmoid. Anything exchanged between the two parts is
rescaled by to_scale(), which raises rather than silently corrupting the FID.
"""

import argparse
import random
import re
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy import linalg
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

matplotlib.use("Agg")            # figures are written to files, never displayed
import matplotlib.pyplot as plt  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class CFG:
    """Defaults, overridden per run by configure() from the command line."""

    seed = 42
    epochs = 10          # same value for AE / VAE / GAN, otherwise FID means nothing
    batch_size = 128
    latent_dim = 100
    lr = 2e-4
    beta1 = 0.5          # DCGAN paper value
    g_feat = 128
    d_feat = 64
    fid_n = 5000         # number of images used to compute FID
    num_workers = 0

    data_dir = Path("data")      # MNIST download location
    shared_dir = Path("shared")  # feature_cnn.pt, fid_ref.npz and the team .npz files
    run_dir = Path("runs/gan_mnist_seed42")


def set_seed(seed=None):
    """Make a run reproducible (python, numpy, torch, cudnn)."""
    seed = CFG.seed if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure(args):
    """Copy parsed arguments into CFG, create the run directory, seed everything."""
    for name in ("seed", "batch_size", "latent_dim", "num_workers",
                 "data_dir", "shared_dir", "epochs", "lr", "fid_n"):
        if getattr(args, name, None) is not None:
            setattr(CFG, name, getattr(args, name))
    CFG.run_dir = args.out_dir / f"gan_mnist_seed{CFG.seed}"
    CFG.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed()
    print(f"Device: {DEVICE} | run dir: {CFG.run_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_mnist():
    """Return images [N,1,28,28] scaled to [-1,1] and labels [N].

    A CSV copy of MNIST is used as a fallback when the download is unavailable.
    """
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])
    try:
        ds = torchvision.datasets.MNIST(root=CFG.data_dir, train=True,
                                        download=True, transform=tfm)
        xs, ys = zip(*[(xb, yb) for xb, yb in DataLoader(ds, batch_size=2048)])
        return torch.cat(xs), torch.cat(ys)
    except Exception as e:
        print("torchvision download failed:", type(e).__name__, "- falling back to CSV")

    csvs = (sorted(Path(CFG.data_dir).rglob("*mnist*train*.csv"))
            or sorted(Path(".").rglob("*mnist*train*.csv")))
    if not csvs:
        raise FileNotFoundError(
            f"MNIST is neither downloadable nor present as a CSV under {CFG.data_dir}.")
    df = pd.read_csv(csvs[0])
    y = df.iloc[:, 0].values.astype(np.int64)
    x = df.iloc[:, 1:].values.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
    print("Loaded from", csvs[0].name)
    return torch.from_numpy((x - 0.5) / 0.5), torch.from_numpy(y)


def make_loader(x, y):
    """Training loader with a seeded shuffle generator, so runs are reproducible."""
    g = torch.Generator()
    g.manual_seed(CFG.seed)
    return DataLoader(TensorDataset(x, y), batch_size=CFG.batch_size, shuffle=True,
                      drop_last=True, generator=g, num_workers=CFG.num_workers)


def reference_set(x, y):
    """Return the CFG.fid_n real images that FID is measured against.

    All three models must score against the exact same reference images, otherwise
    their FID values cannot be put in one table. Reused from shared/anh_that_*.npz
    when present, created and saved for the group otherwise.
    """
    hits = sorted(Path(CFG.shared_dir).glob("anh_that_*.npz"))
    if hits:
        d = np.load(hits[0])
        x_ref = torch.from_numpy(d["imgs"].astype(np.float32))
        y_ref = torch.from_numpy(d["labels"].astype(np.int64))
        CFG.fid_n = len(x_ref)
        print(f"Shared reference set: {hits[0].name} -> {tuple(x_ref.shape)}")
    else:
        idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(CFG.seed))
        x_ref, y_ref = x[idx[:CFG.fid_n]], y[idx[:CFG.fid_n]]
        path = Path(CFG.shared_dir) / f"anh_that_{CFG.fid_n}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, imgs=x_ref.numpy().astype(np.float16),
                            labels=y_ref.numpy().astype(np.int16),
                            seed=CFG.seed, n=CFG.fid_n)
        print(f"No shared reference found -> created {path}. Give this file to the group.")

    print("Label counts:", torch.bincount(y_ref, minlength=10).tolist())
    return x_ref, y_ref


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORKS
# ═══════════════════════════════════════════════════════════════════════════════

def weights_init(m):
    """DCGAN initialisation: weights ~ N(0, 0.02)."""
    name = m.__class__.__name__
    if "Conv" in name or "Linear" in name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """z (100) -> 128x7x7 -> 64x14x14 -> 1x28x28, tanh output in [-1,1]."""

    def __init__(self, latent_dim=100, gf=128):
        super().__init__()
        self.gf = gf
        self.fc = nn.Linear(latent_dim, gf * 7 * 7)
        self.bn0 = nn.BatchNorm1d(gf * 7 * 7)
        self.up1 = nn.ConvTranspose2d(gf, gf // 2, 4, 2, 1)     # 7 -> 14
        self.bn1 = nn.BatchNorm2d(gf // 2)
        self.up2 = nn.ConvTranspose2d(gf // 2, 1, 4, 2, 1)      # 14 -> 28

    def forward(self, z):
        h = F.relu(self.bn0(self.fc(z))).view(-1, self.gf, 7, 7)
        h = F.relu(self.bn1(self.up1(h)))
        return torch.tanh(self.up2(h))


class Discriminator(nn.Module):
    """1x28x28 -> 64x14x14 -> 128x7x7 -> single raw logit."""

    def __init__(self, df=64):
        super().__init__()
        self.c1 = nn.Conv2d(1, df, 4, 2, 1)                     # 28 -> 14
        self.c2 = nn.Conv2d(df, df * 2, 4, 2, 1)                # 14 -> 7
        self.bn2 = nn.BatchNorm2d(df * 2)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(df * 2 * 7 * 7, 1)

    def forward(self, x):
        h = F.leaky_relu(self.c1(x), 0.2)
        h = F.leaky_relu(self.bn2(self.c2(h)), 0.2)
        return self.fc(self.drop(h.flatten(1)))


class CondGenerator(nn.Module):
    """Same as Generator, but the label is embedded and concatenated to z."""

    def __init__(self, latent_dim=100, n_classes=10, emb_dim=50, gf=128):
        super().__init__()
        self.gf = gf
        self.emb = nn.Embedding(n_classes, emb_dim)
        self.fc = nn.Linear(latent_dim + emb_dim, gf * 7 * 7)
        self.bn0 = nn.BatchNorm1d(gf * 7 * 7)
        self.up1 = nn.ConvTranspose2d(gf, gf // 2, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(gf // 2)
        self.up2 = nn.ConvTranspose2d(gf // 2, 1, 4, 2, 1)

    def forward(self, z, y):
        h = torch.cat([z, self.emb(y)], dim=1)
        h = F.relu(self.bn0(self.fc(h))).view(-1, self.gf, 7, 7)
        h = F.relu(self.bn1(self.up1(h)))
        return torch.tanh(self.up2(h))


class CondDiscriminator(nn.Module):
    """Same as Discriminator, but the label is broadcast into 10 one-hot channels
    and stacked onto the image, so D can reject 'real image, wrong label'.

    Conditioning D is the part that does the work: if only G saw the label, nothing
    would punish it for drawing a 3 when a 7 was requested.
    """

    def __init__(self, n_classes=10, df=64):
        super().__init__()
        self.n_classes = n_classes
        self.c1 = nn.Conv2d(1 + n_classes, df, 4, 2, 1)
        self.c2 = nn.Conv2d(df, df * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(df * 2)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(df * 2 * 7 * 7, 1)

    def forward(self, x, y):
        b, _, h_, w_ = x.shape
        ymap = F.one_hot(y, self.n_classes).float().view(b, -1, 1, 1).expand(-1, -1, h_, w_)
        h = F.leaky_relu(self.c1(torch.cat([x, ymap], dim=1)), 0.2)
        h = F.leaky_relu(self.bn2(self.c2(h)), 0.2)
        return self.fc(self.drop(h.flatten(1)))


# ═══════════════════════════════════════════════════════════════════════════════
# FID AND MODE-COLLAPSE METRICS
#
# The original FID uses InceptionV3, trained on colour ImageNet photos, whose
# features are close to meaningless on 28x28 grayscale digits. A small CNN trained
# on MNIST is used instead, so the numbers here are "MNIST-FID": comparable with
# each other, not with FID values published for other datasets.
#
# A score only means something relative to two fixed choices — the feature network
# and the reference images. If each member of the group picks their own, the three
# numbers are measured with three different rulers and nothing raises an error.
# Hence prepare_fid(), which freezes both and writes them to shared/.
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureCNN(nn.Module):
    """Digit classifier whose penultimate 128-d layer is the FID feature space."""

    def __init__(self, feat_dim=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.feat = nn.Linear(64, feat_dim)
        self.head = nn.Linear(feat_dim, 10)

    def forward(self, x, return_feat=False):
        h = self.feat(self.body(x))
        return h if return_feat else self.head(F.relu(h))


def train_feature_cnn(x, y, epochs=2, seed=1234):
    set_seed(seed)
    net = FeatureCNN().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=True, generator=g)
    for ep in range(epochs):
        correct = total = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = net(xb)
            F.cross_entropy(out, yb).backward()
            opt.step()
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        print(f"  FeatureCNN epoch {ep + 1}: acc = {correct / total:.4f}")
    return net.eval()


def to_scale(x, scale):
    """Convert images to the shared [-1,1] scale and fail loudly on a wrong claim.

    `scale` is the output range of the model that produced the images:
    sigmoid -> "[0,1]", tanh -> "[-1,1]". A silent mismatch would corrupt the FID
    without raising anything, which is why the bounds are checked here.
    """
    lo, hi = float(x.min()), float(x.max())
    if scale == "[0,1]":
        if lo < -0.05:
            raise ValueError(f"declared '[0,1]' but min = {lo:.3f} < 0 (looks like [-1,1])")
        return x * 2.0 - 1.0
    if scale == "[-1,1]":
        if lo > -0.05:
            raise ValueError(f"declared '[-1,1]' but min = {lo:.3f} >= 0 (looks like [0,1])")
        if hi > 1.05 or lo < -1.05:
            raise ValueError(f"images outside [-1,1]: [{lo:.3f}, {hi:.3f}]")
        return x
    raise ValueError(f"scale must be '[0,1]' or '[-1,1]', got {scale!r}")


def frechet_distance(mu1, s1, mu2, s2, eps=1e-6):
    """FID = |mu1 - mu2|^2 + Tr(s1 + s2 - 2*sqrt(s1 @ s2))."""
    diff = mu1 - mu2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov = linalg.sqrtm((s1 + eps * np.eye(len(s1))).dot(s2 + eps * np.eye(len(s2))))
    if isinstance(cov, tuple):
        cov = cov[0]
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(cov))


class FIDScorer:
    """Holds the shared feature network and the statistics of the real images.

    Pass either the reference images, or a (mu, sigma) pair exported earlier —
    the stored statistics are what the group's published numbers were measured
    against, so they win when both are available.
    """

    def __init__(self, featnet, ref_imgs=None, mu=None, sigma=None):
        self.net = featnet.to(DEVICE).eval()
        if mu is not None and sigma is not None:
            self.mu = np.asarray(mu, dtype=np.float64)
            self.sigma = np.asarray(sigma, dtype=np.float64)
        elif ref_imgs is not None:
            self.mu, self.sigma = self._stats(ref_imgs)
        else:
            raise ValueError("FIDScorer needs either ref_imgs or (mu, sigma).")

    @torch.no_grad()
    def features(self, imgs, bs=512):
        return np.concatenate(
            [self.net(imgs[i:i + bs].to(DEVICE), return_feat=True).cpu().numpy()
             for i in range(0, len(imgs), bs)], axis=0)

    def _stats(self, imgs):
        f = self.features(imgs)
        return f.mean(axis=0), np.cov(f, rowvar=False)

    def score(self, imgs, scale="[-1,1]"):
        mu, sigma = self._stats(to_scale(imgs, scale))
        return frechet_distance(self.mu, self.sigma, mu, sigma)

    @torch.no_grad()
    def predict(self, imgs, scale="[-1,1]"):
        """Predicted digit per image, used to check conditional generation."""
        imgs = to_scale(imgs, scale)
        return torch.cat([self.net(imgs[i:i + 512].to(DEVICE)).argmax(1).cpu()
                          for i in range(0, len(imgs), 512)])

    def mode_stats(self, imgs, scale="[-1,1]"):
        """Mode-collapse check: class distribution of the generated images.

        Entropy is maximal at ln(10) = 2.303 when all ten digits appear equally
        often; a low value means G only produces a few of them.
        """
        preds = self.predict(imgs, scale)
        hist = torch.bincount(preds, minlength=10).float() / len(preds)
        nz = hist[hist > 0]
        return dict(hist=hist.numpy(),
                    entropy=max(0.0, float(-(nz * nz.log()).sum())),
                    n_modes=int((hist >= 0.01).sum()),
                    pixel_std=float(to_scale(imgs, scale).std(dim=0).mean()))


@torch.no_grad()
def sample_images(G, n, labels=None, seed=None, bs=500):
    """Generate n images with a fixed seed so grids and FID are reproducible."""
    G.eval()
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    outs = []
    for i in range(0, n, bs):
        k = min(bs, n - i)
        z = torch.randn(k, CFG.latent_dim, generator=gen).to(DEVICE)
        outs.append(G(z, labels[i:i + k].to(DEVICE)).cpu() if labels is not None else G(z).cpu())
    G.train()
    return torch.cat(outs)


def balanced_labels(n):
    """0,1,...,9,0,1,... — equal numbers of each digit, for conditional sampling."""
    return torch.arange(10).repeat(n // 10 + 1)[:n]


def save_shared(featnet, x_ref, y_ref):
    """Export the feature network and the reference images for the other members."""
    path = Path(CFG.shared_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(featnet.state_dict(), path / "feature_cnn.pt")
    scorer = FIDScorer(featnet, x_ref)
    np.savez_compressed(path / "fid_ref.npz",
                        mu=scorer.mu, sigma=scorer.sigma, n_ref=len(x_ref),
                        real_imgs=x_ref.numpy().astype(np.float16),
                        real_labels=y_ref.numpy().astype(np.int16))
    print(f"Saved {path}/feature_cnn.pt and {path}/fid_ref.npz - share both with the group.")


def load_shared():
    """Load what save_shared() wrote: (scorer, real images, real labels).

    Re-training a feature network locally gives FID values measured with a
    different ruler, which cannot be compared with the numbers in the report.
    """
    path = Path(CFG.shared_dir)
    net_file, ref_file = path / "feature_cnn.pt", path / "fid_ref.npz"
    if not net_file.exists() or not ref_file.exists():
        raise FileNotFoundError(
            f"Missing {net_file} or {ref_file}. Run `python 5_gan.py prepare` first, "
            f"or copy the two files the group is already using into {path}/.")
    net = FeatureCNN()
    net.load_state_dict(torch.load(net_file, map_location=DEVICE))

    d = np.load(ref_file)
    if "real_imgs" in d.files:
        x_ref = torch.from_numpy(d["real_imgs"].astype(np.float32))
        y_ref = torch.from_numpy(d["real_labels"].astype(np.int64))
    else:
        # Older exports stored only mu and sigma; the images then live beside them.
        hits = sorted(path.glob("anh_that_*.npz"))
        if not hits:
            raise FileNotFoundError(
                f"{ref_file} holds only mu/sigma and no anh_that_*.npz sits next to it. "
                f"The reference images are needed for the grids - copy the group's "
                f"anh_that_*.npz into {path}/.")
        a = np.load(hits[0])
        x_ref = torch.from_numpy(a["imgs"].astype(np.float32))
        y_ref = torch.from_numpy(a["labels"].astype(np.int64))
        print(f"fid_ref.npz has no images; reference grid taken from {hits[0].name}")

    CFG.fid_n = len(x_ref)
    scorer = FIDScorer(net, mu=d["mu"], sigma=d["sigma"])

    # The stored statistics and the reference images must describe the same set:
    # scoring those images against their own statistics has to give ~0.
    drift = scorer.score(x_ref)
    if abs(drift) > 1.0:
        print("!" * 78)
        print(f"CẢNH BÁO: FID(ảnh tham chiếu, thống kê đã lưu) = {drift:.3f}, đáng lẽ ~0.")
        print("fid_ref.npz và tập ảnh thật không khớp nhau -> mọi FID sau đây đều sai.")
        print("Xin lại đúng cặp file dùng chung của nhóm.")
        print("!" * 78)
    else:
        print(f"Shared FID ruler loaded | FID(ref, ref) = {drift:.4f} | n = {CFG.fid_n}")
    return scorer, x_ref, y_ref


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES  (titles kept in Vietnamese to match the report)
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig_name, h_pad=None):
    plt.tight_layout(h_pad=h_pad) if h_pad else plt.tight_layout()
    path = CFG.run_dir / fig_name
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print("  figure ->", path)


def _as_image(imgs, nrow=8):
    return make_grid(imgs, nrow=nrow, normalize=True, value_range=(-1, 1)).permute(1, 2, 0)


def fig_training_curves(h, tag):
    """Hình 1 — G/D losses, the two probabilities D outputs, FID per epoch."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(h.epoch, h.loss_D, label="loss_D")
    ax[0].plot(h.epoch, h.loss_G, label="loss_G")
    ax[0].set_title("Đường loss của D và G")

    ax[1].plot(h.epoch, h.D_x, label="D(x) — ảnh thật")
    ax[1].plot(h.epoch, h.D_Gz, label="D(G(z)) — ảnh giả")
    ax[1].axhline(0.5, ls="--", c="k", lw=1, label="0.5 = cân bằng lý tưởng")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Xác suất D gán cho ảnh")

    ax[2].plot(h.epoch, h.FID, c="tab:red")
    ax[2].set_title("FID (càng thấp càng tốt)")

    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=.3)
    ax[0].legend()
    ax[1].legend()
    plt.suptitle(f"Hình 1 — Quá trình huấn luyện {tag}")
    _save(f"{tag.lower()}_curves.png")


def fig_progress(tag, epochs):
    """Hình 2 — samples from the same fixed z at several epochs."""
    epochs = [e for e in epochs if (CFG.run_dir / f"{tag}_ep{e:03d}.png").exists()]
    fig, axes = plt.subplots(1, len(epochs), figsize=(3.2 * len(epochs), 3.6))
    for ax, e in zip(np.atleast_1d(axes), epochs):
        ax.imshow(plt.imread(CFG.run_dir / f"{tag}_ep{e:03d}.png"), cmap="gray")
        ax.set_title(f"epoch {e}")
        ax.axis("off")
    plt.suptitle(f"Hình 2 — {tag}: cùng z, ảnh rõ dần qua các epoch")
    _save(f"{tag}_progress.png")


def fig_instability(h):
    """Hình 3 — epoch-to-epoch loss jumps and the diversity of the samples."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h.epoch[1:], h.loss_G.diff()[1:], label="Δ loss_G")
    ax[0].plot(h.epoch[1:], h.loss_D.diff()[1:], label="Δ loss_D")
    ax[0].axhline(0, c="k", lw=1)
    ax[0].set_title("Thay đổi loss giữa hai epoch liên tiếp")

    ax[1].plot(h.epoch, h.entropy, c="tab:green", label="entropy lớp")
    ax[1].axhline(np.log(10), ls="--", c="k", lw=1, label="tối đa = ln(10)")
    ax[1].axhline(1.8, ls=":", c="r", lw=1, label="ngưỡng mode collapse")
    ax[1].set_title("Độ đa dạng của ảnh sinh ra")

    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
        a.grid(alpha=.3)
    plt.suptitle("Hình 3 — Biểu hiện bất ổn định")
    _save("instability.png")


def fig_mode_histogram(ms_real, ms_gen):
    """Hình 4 — class distribution of real vs generated images."""
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.4
    ax.bar(np.arange(10) - w / 2, ms_real["hist"], w, label="MNIST thật")
    ax.bar(np.arange(10) + w / 2, ms_gen["hist"], w, label="DCGAN sinh ra")
    ax.axhline(0.1, ls="--", c="k", lw=1, label="phân bố đều = 0.1")
    ax.set_xticks(range(10))
    ax.set_xlabel("chữ số")
    ax.set_ylabel("tỉ lệ")
    ax.set_title(f"Hình 4 — Phân bố lớp | entropy thật {ms_real['entropy']:.3f} "
                 f"vs GAN {ms_gen['entropy']:.3f}")
    ax.legend()
    _save("mode_hist.png")


def fig_label_grid(imgs, title, fig_name):
    """Hình 5 — one requested digit per row, 10 different noise vectors per row."""
    plt.figure(figsize=(8, 8))
    plt.imshow(_as_image(imgs, nrow=10), cmap="gray")
    plt.title(title)
    plt.axis("off")
    _save(fig_name)


def fig_compare_grids(panels, title, fig_name):
    """Hình 6 / 6b — one 8x8 grid of samples per model, side by side."""
    ncol = min(5, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(panels):]:
        ax.axis("off")
    for ax, (name, imgs) in zip(axes, panels):
        ax.imshow(_as_image(imgs[:64]), cmap="gray")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    plt.suptitle(title, fontsize=13)
    _save(fig_name, h_pad=5.0)   # two-line panel titles need room between rows


def fig_compare_bars(table):
    """Hình 7 — FID and class entropy of every model in the comparison table."""
    colors = plt.cm.tab10(np.arange(len(table)) % 10)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(table["Mô hình"], table["FID"], color=colors)
    ax[0].set_title("FID (thấp = tốt)")
    ax[0].set_ylim(0, None)
    ax[1].bar(table["Mô hình"], table["Entropy lớp"], color=colors)
    ax[1].axhline(np.log(10), ls="--", c="k", lw=1, label="tối đa = ln(10)")
    ax[1].set_title("Entropy lớp (cao = đa dạng)")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=.3, axis="y")
        a.tick_params(axis="x", rotation=30)
    plt.suptitle("Hình 7 — So sánh định lượng")
    _save("so_sanh_cot.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_gan(loader, scorer, conditional=False, tag=None):
    """Train DCGAN, or the conditional variant when `conditional` is True.

    The two loops differ only in that the label is passed to both networks and
    that the fake labels are drawn at random; keeping them in one function is what
    guarantees the comparison between them is not confounded by a stray detail.
    """
    tag = tag or ("cgan" if conditional else "dcgan")
    set_seed()
    loader.generator.manual_seed(CFG.seed)

    if conditional:
        G = CondGenerator(CFG.latent_dim, gf=CFG.g_feat).to(DEVICE)
        D = CondDiscriminator(df=CFG.d_feat).to(DEVICE)
    else:
        G = Generator(CFG.latent_dim, CFG.g_feat).to(DEVICE)
        D = Discriminator(CFG.d_feat).to(DEVICE)
    G.apply(weights_init)
    D.apply(weights_init)
    optG = torch.optim.Adam(G.parameters(), lr=CFG.lr, betas=(CFG.beta1, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=CFG.lr, betas=(CFG.beta1, 0.999))
    bce = nn.BCEWithLogitsLoss()

    gfix = torch.Generator()
    gfix.manual_seed(2024)
    n_fix = 100 if conditional else 64
    fixed_z = torch.randn(n_fix, CFG.latent_dim, generator=gfix).to(DEVICE)  # same z every epoch
    fixed_y = torch.arange(10, device=DEVICE).repeat_interleave(10) if conditional else None
    y_eval = balanced_labels(CFG.fid_n) if conditional else None

    hist, t0 = [], time.time()
    for ep in range(1, CFG.epochs + 1):
        G.train()
        D.train()
        s = dict(lD=0., lG=0., dx=0., dgz=0., gn=0., nb=0)

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            b = len(xb)
            ones = torch.ones(b, 1, device=DEVICE)
            zeros = torch.zeros(b, 1, device=DEVICE)
            z = torch.randn(b, CFG.latent_dim, device=DEVICE)
            y_fake = torch.randint(0, 10, (b,), device=DEVICE) if conditional else None

            # --- D step: real -> 1, fake -> 0. detach() keeps the update out of G.
            optD.zero_grad(set_to_none=True)
            out_real = D(xb, yb) if conditional else D(xb)
            fake = G(z, y_fake) if conditional else G(z)
            out_fake = D(fake.detach(), y_fake) if conditional else D(fake.detach())
            lossD = bce(out_real, ones) + bce(out_fake, zeros)
            lossD.backward()
            optD.step()

            # --- G step: same fakes, but G asks D to answer "real" (non-saturating).
            optG.zero_grad(set_to_none=True)
            lossG = bce(D(fake, y_fake) if conditional else D(fake), ones)
            lossG.backward()
            with torch.no_grad():
                gn = torch.sqrt(sum((p.grad ** 2).sum()
                                    for p in G.parameters() if p.grad is not None))
            optG.step()

            s["lD"] += lossD.item()
            s["lG"] += lossG.item()
            s["dx"] += torch.sigmoid(out_real).mean().item()
            s["dgz"] += torch.sigmoid(out_fake).mean().item()
            s["gn"] += gn.item()
            s["nb"] += 1

        nb = s["nb"]
        imgs = sample_images(G, CFG.fid_n, labels=y_eval, seed=777)
        ms = scorer.mode_stats(imgs)
        hist.append(dict(epoch=ep, loss_D=s["lD"] / nb, loss_G=s["lG"] / nb,
                         D_x=s["dx"] / nb, D_Gz=s["dgz"] / nb, grad_G=s["gn"] / nb,
                         FID=scorer.score(imgs), entropy=ms["entropy"],
                         n_modes=ms["n_modes"], pixel_std=ms["pixel_std"],
                         sec=round(time.time() - t0, 1)))

        G.eval()
        with torch.no_grad():
            out = G(fixed_z, fixed_y) if conditional else G(fixed_z)
            save_image(out.cpu(), CFG.run_dir / f"{tag}_ep{ep:03d}.png",
                       nrow=10 if conditional else 8, normalize=True, value_range=(-1, 1))
        G.train()

        r = hist[-1]
        print(f"[{tag}] ep {ep:3d}/{CFG.epochs} | loss_D={r['loss_D']:.3f} "
              f"loss_G={r['loss_G']:.3f} | D(x)={r['D_x']:.3f} D(G(z))={r['D_Gz']:.3f} "
              f"| FID={r['FID']:.2f} entropy={r['entropy']:.3f} | {r['sec']:.0f}s")

    return G, D, pd.DataFrame(hist)


def diagnose(h):
    """Requirement 4: read the instability off the last 10 epochs."""
    tail = h.tail(min(10, len(h)))
    gap = float(tail.D_x.mean() - tail.D_Gz.mean())
    osc_d, osc_g = float(tail.loss_D.std()), float(tail.loss_G.std())
    ent, nm = float(h.entropy.iloc[-1]), int(h.n_modes.iloc[-1])

    print("PHÂN TÍCH BẤT ỔN ĐỊNH (10 epoch cuối)")
    print("-" * 60)
    print(f"  D(x) trung bình         = {tail.D_x.mean():.3f}")
    print(f"  D(G(z)) trung bình      = {tail.D_Gz.mean():.3f}")
    print(f"  Chênh lệch D(x)-D(G(z)) = {gap:.3f}   "
          f"{'-> D thắng áp đảo' if gap > 0.8 else '-> cân bằng chấp nhận được'}")
    print(f"  std(loss_D) = {osc_d:.3f} | std(loss_G) = {osc_g:.3f}   "
          f"{'-> dao động mạnh' if max(osc_d, osc_g) > 0.3 else '-> dao động vừa phải'}")
    print(f"  Chuẩn gradient G        = {tail.grad_G.mean():.3f}")
    print(f"  Entropy lớp cuối        = {ent:.3f} / {np.log(10):.3f} | "
          f"số mode sống = {nm}/10")
    collapsed = ent < 1.8 or nm < 8
    print("  -> CÓ dấu hiệu mode collapse" if collapsed else "  -> KHÔNG thấy mode collapse")
    print(f"  FID: đầu {h.FID.iloc[0]:.2f} -> cuối {h.FID.iloc[-1]:.2f} | "
          f"tốt nhất {h.FID.min():.2f} (epoch {int(h.epoch[h.FID.idxmin()])})")
    print("-" * 60)
    if h.FID.iloc[-1] > h.FID.min() * 1.15:
        print("  Lưu ý: FID cuối TỆ HƠN FID tốt nhất -> chất lượng lên xuống thất thường,")
        print("         đúng biểu hiện của bất ổn định. Nên lưu checkpoint tại epoch tốt nhất.")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON WITH THE AE / VAE IMAGES SENT BY THE OTHER MEMBERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_team_sets(scorer, team_dir):
    """Read every AE/VAE .npz the team produced.

    Accepts both naming schemes ('anh_sinh_<name>.npz' and
    '<ae|vae>_<kind>_latent<L>.npz'), rescales [0,1] outputs to [-1,1], marks each
    set as a true sample or a reconstruction, and flags sets that look broken.
    """
    sets, epochs, kinds = {}, {}, {}
    root = Path(team_dir)
    files = sorted(root.rglob("*latent*.npz")) + sorted(root.rglob("anh_sinh_*.npz"))
    print(f"Found {len(files)} team file(s) under {root}/")

    for p in files:
        d = np.load(p)
        key_img = "imgs" if "imgs" in d.files else d.files[0]
        a = torch.from_numpy(d[key_img].astype(np.float32))
        if a.ndim == 2:
            a = a.view(-1, 1, 28, 28)
        if float(a.min()) >= -0.05:          # sigmoid output -> rescale to [-1,1]
            a = a * 2.0 - 1.0

        stem = p.stem.replace("anh_sinh_", "")
        model = re.search(r"(ae|vae|gan|cgan)", stem, re.I)
        latent = re.search(r"latent(\d+)", stem, re.I)
        # " (nhóm)" keeps team sets from overwriting the models trained here
        name = ((model.group(1).upper() if model else stem)
                + (f" lat={latent.group(1)}" if latent else "") + " (nhóm)")
        if name in sets:
            continue

        sets[name] = a
        epochs[name] = int(d["epochs"]) if "epochs" in d.files else None
        kinds[name] = "tái tạo" if re.search(r"recon", stem, re.I) else "sinh mới"

        ms = scorer.mode_stats(a)
        flag = ""
        if ms["pixel_std"] < 0.12:
            flag = "  <<< BROKEN: every image is almost identical"
        elif ms["n_modes"] < 6:
            flag = f"  <<< SUSPICIOUS: only {ms['n_modes']}/10 digits appear"
        print(f"  {name:16s} {kinds[name]:9s} {tuple(a.shape)} | entropy={ms['entropy']:.2f} "
              f"| modes={ms['n_modes']}/10 | pixel_std={ms['pixel_std']:.3f}{flag}")

    if any(v == "tái tạo" for v in kinds.values()):
        print("\n" + "!" * 76)
        print("CẢNH BÁO: có bộ ảnh là TÁI TẠO (đưa ảnh thật vào rồi lấy ra), không phải SINH MỚI.")
        print("FID của ảnh tái tạo luôn thấp giả tạo vì mô hình đã được xem trước đáp án.")
        print("Không xếp chung cột với GAN; phải xin bản sinh từ z ~ N(0, I).")
        print("!" * 76)

    return sets, epochs, kinds


def build_table(sets, epochs, kinds, scorer):
    unknown = [k for k in sets if k in epochs and epochs.get(k) is None]
    if unknown:
        print(f"Lưu ý: {len(unknown)} bộ ảnh của nhóm không ghi số epoch trong file .npz "
              f"-> cột 'Số epoch' của chúng đang lấy theo --epochs ({CFG.epochs}). "
              f"Hỏi lại đồng đội để chắc chắn.")
    rows = []
    for name, imgs in sets.items():
        ms = scorer.mode_stats(imgs)
        rows.append({"Mô hình": name,
                     "Loại ảnh": kinds.get(name, "sinh mới"),
                     "Số epoch": epochs.get(name) or CFG.epochs,
                     "FID": round(scorer.score(imgs), 2),
                     "Entropy lớp": round(ms["entropy"], 3),
                     "Số mode (>=1%)": ms["n_modes"],
                     "Pixel std": round(ms["pixel_std"], 4)})
    table = pd.DataFrame(rows)

    all_epochs = sorted(set(table["Số epoch"]))
    if len(all_epochs) > 1:
        print("!" * 78)
        print(f"CẢNH BÁO: số epoch không đồng nhất giữa các mô hình: {all_epochs}.")
        print("Đề bài yêu cầu so sánh Ở CÙNG SỐ EPOCH -> phải chạy lại cho khớp.")
        print("!" * 78)
    return table


def load_generators():
    """Restore the two generators trained by the `dcgan` and `cgan` stages."""
    for name in ("dcgan.pt", "cgan.pt"):
        if not (CFG.run_dir / name).exists():
            raise FileNotFoundError(
                f"{CFG.run_dir / name} not found - run `python 5_gan.py dcgan` and "
                f"`python 5_gan.py cgan` first (same --out-dir and --seed).")
    G = Generator(CFG.latent_dim, CFG.g_feat).to(DEVICE)
    G.load_state_dict(torch.load(CFG.run_dir / "dcgan.pt", map_location=DEVICE)["G"])
    G_cond = CondGenerator(CFG.latent_dim, gf=CFG.g_feat).to(DEVICE)
    G_cond.load_state_dict(torch.load(CFG.run_dir / "cgan.pt", map_location=DEVICE)["G"])
    return G, G_cond


# ═══════════════════════════════════════════════════════════════════════════════
# STAGES
# ═══════════════════════════════════════════════════════════════════════════════

def stage_prepare(args):
    """Freeze the feature network and the reference images for the whole group."""
    x, y = load_mnist()
    print("images:", tuple(x.shape), "| range: [%.2f, %.2f]" % (x.min(), x.max()))

    x_ref, y_ref = reference_set(x, y)
    featnet = train_feature_cnn(x, y, epochs=args.feat_epochs)
    scorer = FIDScorer(featnet, x_ref)

    # Sanity check: two disjoint samples of real images must score close to 0.
    idx = torch.randperm(len(x))[:CFG.fid_n]
    print(f"FID(real, real) = {scorer.score(x[idx]):.4f}  (should be near 0)")
    ms = scorer.mode_stats(x_ref)
    print(f"Real MNIST: entropy = {ms['entropy']:.4f} (max {np.log(10):.4f}), "
          f"modes = {ms['n_modes']}/10")

    save_shared(featnet, x_ref, y_ref)


def stage_dcgan(args):
    """Requirement 1 (architecture, loss curves) and 4 (instability)."""
    scorer, x_ref, _ = load_shared()
    loader = make_loader(*load_mnist())
    print("batches/epoch:", len(loader))

    G, D, h = train_gan(loader, scorer, conditional=False)

    fig_training_curves(h, "DCGAN")
    marks = sorted({1, max(1, CFG.epochs // 6), max(1, CFG.epochs // 3),
                    max(1, CFG.epochs // 2), CFG.epochs})
    fig_progress("dcgan", marks)
    diagnose(h)
    fig_instability(h)

    ms_gen = scorer.mode_stats(sample_images(G, CFG.fid_n, seed=888))
    ms_real = scorer.mode_stats(x_ref)
    fig_mode_histogram(ms_real, ms_gen)
    print(f"Số mode sống (>=1%): {ms_gen['n_modes']}/10 | "
          f"pixel_std = {ms_gen['pixel_std']:.4f} (ảnh thật {ms_real['pixel_std']:.4f})")

    h.to_csv(CFG.run_dir / "lich_su_dcgan.csv", index=False)
    torch.save({"G": G.state_dict(), "D": D.state_dict()}, CFG.run_dir / "dcgan.pt")
    print("Saved", CFG.run_dir / "dcgan.pt")


def stage_cgan(args):
    """Requirement 3: generate the digit that was asked for."""
    scorer, _, _ = load_shared()
    loader = make_loader(*load_mnist())

    G, D, h = train_gan(loader, scorer, conditional=True)

    rows = torch.tensor(args.digits).repeat_interleave(10)
    fig_label_grid(sample_images(G, len(rows), labels=rows, seed=2024),
                   "Hình 5 — cGAN: mỗi hàng là một chữ số được chỉ định",
                   "cgan_by_label.png")
    fig_training_curves(h, "cGAN")

    # Share of samples the classifier reads back as the digit that was requested;
    # random guessing would give about 10%.
    y_req = balanced_labels(2000)
    acc = float((scorer.predict(sample_images(G, 2000, labels=y_req, seed=999)) == y_req)
                .float().mean())
    print(f"Độ khớp nhãn: {acc * 100:.1f}% ảnh sinh ra đúng chữ số được yêu cầu "
          f"(đoán mò chỉ ~10%)")

    h.to_csv(CFG.run_dir / "lich_su_cgan.csv", index=False)
    torch.save({"G": G.state_dict(), "D": D.state_dict()}, CFG.run_dir / "cgan.pt")
    print("Saved", CFG.run_dir / "cgan.pt")


def stage_compare(args):
    """Requirement 2: FID table and sample grids for AE / VAE / GAN."""
    scorer, x_ref, _ = load_shared()
    G, G_cond = load_generators()

    sets = {"DCGAN": sample_images(G, CFG.fid_n, seed=777),
            "cGAN": sample_images(G_cond, CFG.fid_n,
                                  labels=balanced_labels(CFG.fid_n), seed=777)}
    kinds = {k: "sinh mới" for k in sets}

    team_sets, team_epochs, team_kinds = load_team_sets(scorer, args.team_dir or CFG.shared_dir)
    sets.update(team_sets)
    kinds.update(team_kinds)

    table = build_table(sets, team_epochs, kinds, scorer)
    table.to_csv(CFG.run_dir / "bang_so_sanh.csv", index=False)
    print("\nBảng 1 — So sánh AE / VAE / GAN (FID càng thấp càng tốt)")
    print(table.to_string(index=False))

    def panels(names):
        out = [("MNIST thật", x_ref[:64])]
        for k in names:
            fid = table.loc[table["Mô hình"] == k, "FID"]
            out.append((f"{k}\nFID {fid.item():.2f}" if len(fid) else k, sets[k]))
        return out

    samples = [k for k in sets if kinds.get(k, "sinh mới") == "sinh mới"]
    recons = [k for k in sets if kinds.get(k) == "tái tạo"]

    ep_label = "/".join(str(e) for e in sorted(set(table["Số epoch"])))
    fig_compare_grids(panels(samples),
                      f"Hình 6 — Ảnh SINH MỚI ở cùng {ep_label} epoch (yêu cầu 2)",
                      "so_sanh_luoi_anh.png")
    if recons:
        recons.sort(key=lambda s: (s.split()[0],
                                   int(re.search(r"lat=(\d+)", s).group(1))
                                   if "lat=" in s else 0))
        fig_compare_grids(panels(recons),
                          "Hình 6b — Ảnh TÁI TẠO theo chiều latent (khảo sát không gian ẩn)",
                          "khao_sat_latent.png")

    fig_compare_bars(table)


def stage_all(args):
    if not (Path(CFG.shared_dir) / "feature_cnn.pt").exists():
        stage_prepare(args)
    stage_dcgan(args)
    stage_cgan(args)
    stage_compare(args)


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND LINE
# ═══════════════════════════════════════════════════════════════════════════════

STAGES = {"prepare": stage_prepare, "dcgan": stage_dcgan,
          "cgan": stage_cgan, "compare": stage_compare, "all": stage_all}


def main():
    parser = argparse.ArgumentParser(
        description="DCGAN / conditional GAN on MNIST, with the shared FID evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("stage", choices=list(STAGES),
                        help="prepare = FID setup, then dcgan, cgan, compare; "
                             "all = every stage in order")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"),
                        help="Parent directory for run outputs.")
    parser.add_argument("--data-dir", type=Path, default=CFG.data_dir,
                        help="Where MNIST is downloaded to.")
    parser.add_argument("--shared-dir", type=Path, default=CFG.shared_dir,
                        help="Feature network, FID reference set and the team .npz files.")
    parser.add_argument("--team-dir", type=Path, default=None,
                        help="Where the AE/VAE .npz files are (default: --shared-dir).")
    parser.add_argument("--epochs", type=int, default=CFG.epochs,
                        help="Must match the AE/VAE runs for the comparison to be valid.")
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--latent-dim", type=int, default=CFG.latent_dim)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--num-workers", type=int, default=CFG.num_workers)
    parser.add_argument("--fid-n", type=int, default=CFG.fid_n,
                        help="Reference images; ignored once shared/fid_ref.npz exists.")
    parser.add_argument("--feat-epochs", type=int, default=2,
                        help="Epochs to train the FID feature extractor (stage prepare).")
    parser.add_argument("--digits", type=int, nargs="+", default=list(range(10)),
                        help="Digits to put in the cGAN label grid, one row each.")
    args = parser.parse_args()

    configure(args)
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
