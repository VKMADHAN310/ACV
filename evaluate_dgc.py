#!/usr/bin/env python3
"""
Quick benchmarking of Dynamic‑Group CNN backbones on
Flowers‑102, Tiny‑ImageNet‑200, and a 3‑class Medical set.

Example:
  python evaluate_dgc.py --arch dyresnet18 --dataset flowers
"""

import argparse, time, csv, os, pathlib
from functools import partial

import torch, torchvision as tv
from torch import nn, optim
from torch.utils.data import DataLoader
from thop import profile                               # pip install thop
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1.  MODEL BUILDERS  (imported from your models package)
# ------------------------------------------------------------
from models import dyresnet18, dymobilenet_v2


# ------------------------------------------------------------
# 2.  DATASET HELPERS
# ------------------------------------------------------------
def tfm(size):
    """Common transform block"""
    return tv.transforms.Compose([
        tv.transforms.Resize((size, size)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5]*3),
    ])


def flowers_dataloaders(root, bs=64):
    """
    Folder layout (already on disk):

    /mnt/c/.../flower_data/
        train/
            0/ …jpg
            1/ …jpg
            ...
        valid/
            0/ …jpg
            ...
        test/   (un‑labelled – ignored here)
    """
    train_dir = pathlib.Path(root) / "train"
    val_dir   = pathlib.Path(root) / "valid"

    train_set = tv.datasets.ImageFolder(train_dir, transform=tfm(224))
    val_set   = tv.datasets.ImageFolder(val_dir,   transform=tfm(224))

    return (DataLoader(train_set, batch_size=bs, shuffle=True,  num_workers=4),
            DataLoader(val_set,   batch_size=bs, shuffle=False, num_workers=4),
            len(train_set.classes))                         # num_classes


class TinyImageNet(tv.datasets.VisionDataset):
    """Handles both the train/ and val/ folders of Tiny‑ImageNet‑200."""
    def __init__(self, root, split="train", transform=None):
        assert split in {"train", "val"}
        super().__init__(root, transform=transform, target_transform=None)

        self.loader = tv.datasets.folder.default_loader
        self.extensions = ("JPEG",)

        if split == "train":
            # train/<wnid>/images/*.JPEG
            root = pathlib.Path(root) / "train"
            classes = sorted(p.name for p in root.iterdir() if p.is_dir())
            class_to_idx = {c: i for i, c in enumerate(classes)}
            samples = []
            for cls in classes:
                for img in (root / cls / "images").glob("*.JPEG"):
                    samples.append((img, class_to_idx[cls]))
        else:
            # val/images/*.JPEG  +  val/val_annotations.txt
            root = pathlib.Path(root) / "val"
            ann = {}
            with open(root / "val_annotations.txt") as f:
                for line in f:
                    fname, wnid, *_ = line.split("\t")
                    ann[fname] = wnid
            classes = sorted({wnid for wnid in ann.values()})
            class_to_idx = {c: i for i, c in enumerate(classes)}
            samples = [(root / "images" / fname, class_to_idx[ann[fname]])
                       for fname in ann]

        self.classes, self.class_to_idx = classes, class_to_idx
        self.samples, self.targets = samples, [s[1] for s in samples]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)


def tiny_dataloaders(root, bs=64):
    """Returns (train_loader, val_loader, num_classes)."""
    train_set = TinyImageNet(root, "train", transform=tfm(64))
    val_set   = TinyImageNet(root, "val",   transform=tfm(64))
    return (DataLoader(train_set, batch_size=bs, shuffle=True,  num_workers=4),
            DataLoader(val_set,   batch_size=bs, shuffle=False, num_workers=4),
            200)                                                  # 200 classes


def medical_dataloaders(root, bs=64):
    """
    Root has three subfolders: CXR, BreastMRI, AbdomenCT
    """
    train_set = tv.datasets.ImageFolder(root, transform=tfm(128))
    # simple random split 80/20
    n = len(train_set)
    val_len = n // 5
    train_len = n - val_len
    train_set, val_set = torch.utils.data.random_split(
        train_set, [train_len, val_len],
        generator=torch.Generator().manual_seed(42))
    train_set.dataset.transform = tfm(128)
    val_set.dataset.transform   = tfm(128)
    return (DataLoader(train_set, batch_size=bs, shuffle=True,  num_workers=4),
            DataLoader(val_set,   batch_size=bs, shuffle=False, num_workers=4),
            3)                                                # 3 classes


DATASETS = {
    "flowers": partial(flowers_dataloaders,
                       root="/mnt/c/Users/vkmad/Downloads/flower_data/flower_data"),
    "tiny":    partial(tiny_dataloaders,
                       root="/mnt/c/Users/vkmad/Downloads/tiny-imagenet-200/tiny-imagenet-200"),
    "medical": partial(medical_dataloaders,
                       root="/mnt/c/Users/vkmad/Downloads/medical"),
}


# ------------------------------------------------------------
# 3.  TRAIN / EVAL LOOPS
# ------------------------------------------------------------
def build_model(arch, num_classes):
    if arch == "dyresnet18":
        return dyresnet18(num_classes)
    if arch == "dymobilenet_v2":
        return dymobilenet_v2(num_classes)
    raise ValueError(f"Unknown arch {arch}")


def train_one_epoch(model, loader, loss_fn, opt, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    t0 = time.time()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    inf_time = (time.time() - t0) / total
    return correct / total, inf_time             # acc, sec / img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["dyresnet18", "dymobilenet_v2"],
                        required=True)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--bs", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # build loaders & model
    train_loader, val_loader, n_cls = DATASETS[args.dataset](bs=args.bs)
    model = build_model(args.arch, n_cls).to(device)

    # MACs / params
    dummy = torch.randn(1, 3, next(iter(train_loader))[0].shape[-1],
                        next(iter(train_loader))[0].shape[-1]).to(device)
    macs, _params = profile(model, inputs=(dummy,), verbose=False)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print(f"==> Training {args.arch} on {args.dataset} for {args.epochs} epochs")
    for ep in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, loss_fn, opt, device)
        acc, _ = evaluate(model, val_loader, device)
        print(f"  Epoch {ep:2d}: val Top‑1 = {acc*100:6.2f}%")

    acc, inf_time = evaluate(model, val_loader, device)
    print("\n----- final metrics -----")
    print(f"Top‑1 accuracy : {acc*100:6.2f}%")
    print(f"MACs / image   : {macs/1e6:6.1f} M")
    print(f"Inference time : {inf_time*1e3:6.2f} ms  (batch‑1 on {device})")

    # bar‑chart
    plt.figure()
    plt.bar(["Acc (%)", "MACs (M)", "Time (ms)"],
            [acc*100, macs/1e6, inf_time*1e3])
    plt.title(f"{args.arch} on {args.dataset}")
    plt.savefig("results.png")
    print("Chart saved to results.png")


if __name__ == "__main__":
    main()
