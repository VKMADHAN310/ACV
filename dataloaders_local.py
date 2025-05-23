import os, json, csv, torch
from torchvision import transforms as T, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ───────────────────────────────
# Common image transform pipeline
# ───────────────────────────────
def _xform(size):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5]*3),
    ])

# ───────────────────────────────
# 1. Flowers‑102  (train / val / test)
# directory layout: flower_data/{train|valid|test}/<class_name>/*
# ───────────────────────────────
def flowers_loaders(root, img_size=224, batch=64, workers=4):
    tfm = _xform(img_size)
    train_ds = datasets.ImageFolder(os.path.join(root, "train"),  transform=tfm)
    val_ds   = datasets.ImageFolder(os.path.join(root, "valid"),  transform=tfm)
    test_ds  = datasets.ImageFolder(os.path.join(root, "test"),   transform=tfm)
    return _wrap(train_ds, val_ds, test_ds, batch, workers)

# ───────────────────────────────
# 2. Three‑class Medical set
# directory layout: medical/{CXR|BreastMRI|AbdomenCT}/*
# (no explicit splits – we’ll make an 80 / 20 random split)
# ───────────────────────────────
def medical_loaders(root, img_size=64, batch=64, workers=4, val_frac=0.2):
    tfm = _xform(img_size)
    full = datasets.ImageFolder(root, transform=tfm)
    # random split
    val_sz = int(len(full)*val_frac)
    train_sz = len(full) - val_sz
    train_ds, val_ds = torch.utils.data.random_split(full, [train_sz, val_sz])
    return _wrap(train_ds, val_ds, None, batch, workers)

# ───────────────────────────────
# 3. Tiny‑ImageNet‑200
# train:     train/<wnid>/images/*.JPEG
#           (+ we ignore *_boxes.txt for classification)
# val:       val/images/*.JPEG  +  val/val_annotations.txt
# test:      test/images/*.JPEG  (no labels, so we skip)
# ───────────────────────────────
class TinyVal(Dataset):
    """Validation‑split of Tiny‑ImageNet with labels from val_annotations.txt"""
    def __init__(self, root, transform):
        self.root = root
        self.tfm  = transform
        # read mapping
        ann = os.path.join(root, "val_annotations.txt")
        with open(ann) as f:
            reader = csv.reader(f, delimiter='\t')
            rows = [r for r in reader]
        self.im_paths = [os.path.join(root, "images", r[0]) for r in rows]
        wnids = [r[1] for r in rows]
        # map wnid → integer label using the order of train sub‑folders
        wnid_to_idx = {wnid: idx for idx, wnid in
                       enumerate(sorted(os.listdir(os.path.join(root, "..", "train"))))}
        self.targets = [wnid_to_idx[w] for w in wnids]

    def __len__(self): return len(self.im_paths)

    def __getitem__(self, i):
        img = Image.open(self.im_paths[i]).convert("RGB")
        return self.tfm(img), self.targets[i]

def tiny_loaders(root, img_size=64, batch=128, workers=4):
    tfm = _xform(img_size)
    train_dir = os.path.join(root, "train")
    train_ds  = datasets.ImageFolder(train_dir, transform=tfm)
    val_ds    = TinyVal(os.path.join(root, "val"), transform=tfm)
    return _wrap(train_ds, val_ds, None, batch, workers)

# ───────────────────────────────
# Helper to return neatly‑bundled loaders
# ───────────────────────────────
def _wrap(train_ds, val_ds, test_ds, batch, workers):
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    test_ld  = (DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
                if test_ds else None)
    num_classes = len(train_ds.dataset.classes) if isinstance(train_ds, torch.utils.data.Subset) else len(train_ds.classes)
    return train_ld, val_ld, test_ld, num_classes
