"""
Data loading, transforms, and staged-data verification shared across
notebooks: Dataset class, transform/normalisation pipeline, DataLoader
construction, and the folder-scanning helpers used to verify staged data
and build image listings for datasets with no pre-existing canonical CSV.

Extracted from 03_training.ipynb and Notebook 04's Section 2, so
Notebook 05 (XAI/GradCAM) can reuse the exact same data pipeline instead
of maintaining a third copy.

03_training.ipynb itself is left untouched -- same reasoning as
src/models.py and src/utils.py: it already ran successfully against its
own inline copies, and there's no reason to re-run anything that already
produced the 18 final checkpoints.

NOTE: this module only includes the augmentation-free 'eval' transform
path. 03_training.ipynb's build_train_augmentations() (the 'train' branch
of build_transform) is deliberately not reproduced here, since neither
Notebook 04 (evaluation) nor Notebook 05 (XAI/GradCAM) perform any
training. build_transform(mode='train') raises explicitly rather than
failing with a confusing NameError on a function that was never brought
over -- see its docstring below.
"""

import random

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import SEED


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_TO_IDX = {'benign': 0, 'malignant': 1, 'normal': 2}

# Confirmed final counts, from Notebook 02.
EXPECTED_COUNTS = {
    ('BUSI/BUSI_annotated', 'benign'): 296, ('BUSI/BUSI_annotated', 'malignant'): 161, ('BUSI/BUSI_annotated', 'normal'): 73,
    ('BUSI/BUSI_inpainted', 'benign'): 296, ('BUSI/BUSI_inpainted', 'malignant'): 161, ('BUSI/BUSI_inpainted', 'normal'): 73,
    ('BUSI/BUSI_masked', 'benign'): 296, ('BUSI/BUSI_masked', 'malignant'): 161, ('BUSI/BUSI_masked', 'normal'): 73,
    ('BUS-UCLM', 'benign'): 154, ('BUS-UCLM', 'malignant'): 71, ('BUS-UCLM', 'normal'): 404,
    ('BrEaST', 'benign'): 154, ('BrEaST', 'malignant'): 98, ('BrEaST', 'normal'): 4,
}

BATCH_SIZE = 16
NUM_WORKERS = 4


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class BreastUltrasoundDataset(Dataset):
  """
  General-purpose dataset for any single BUSI arm, or BUS-UCLM/BrEaST.

  df        : dataframe with 'image_name' and 'class' columns (a full dataset's
              listing, or a fold-filtered slice of one)
  base_dir  : directory containing per-class subfolders
  transform : torchvision transform pipeline; differs between train mode (augmentation
              + normalisation) and eval mode (normalisation only)
  """
  def __init__(self, df, base_dir, transform):
    self.df = df.reset_index(drop=True)
    self.base_dir = Path(base_dir)
    self.transform = transform

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    img_path = self.base_dir / row['class'] / row['image_name']

    image = Image.open(img_path).convert('RGB')
    image = self.transform(image)

    label = CLASS_TO_IDX[row['class']]
    return image, label


# ---------------------------------------------------------------------------
# Folder scanning / staged-data verification
# ---------------------------------------------------------------------------

def scan_class_folders(base_dir, classes):
  """
  Scans a directory containing one subfolder per class, returning a
  dataframe of (image_name, class) pairs.

  Generalises 03_training.ipynb's scan_busi_arm(arm) -- which only worked
  for BUSI's dataset/arm/class three-level structure via an implicit
  dependency on notebook-level globals (LOCAL_DATA_DIR, BUSI_CLASSES) --
  to any base_dir with a class subfolder directly beneath it, with no
  hidden dependencies: BUSI's per-arm folders (LOCAL_DATA_DIR/BUSI/{arm}),
  BUS-UCLM, and BrEaST (LOCAL_DATA_DIR/{dataset}) are all just different
  base_dir values passed in explicitly by the caller.
  """
  records = []
  for cls in classes:
    class_dir = base_dir / cls
    for img_path in sorted(class_dir.glob('*.png')):
      records.append({'image_name': img_path.name, 'class': cls})
  return pd.DataFrame(records)


def scan_local_counts(base_dir):
  """
  Walks the extracted local data structure and counts image files per
  (dataset/arm, class) folder. Returns a dict keyed identically to
  EXPECTED_COUNTS, for direct comparison.
  """
  counts = {}
  for (subpath, cls) in EXPECTED_COUNTS.keys():
    class_dir = base_dir / subpath / cls
    if not class_dir.exists():
      counts[(subpath, cls)] = None
    else:
      counts[(subpath, cls)] = len(list(class_dir.glob('*.png')))
  return counts


# ---------------------------------------------------------------------------
# Transforms & normalisation
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(mode, mean, std):
  """
  mode: 'train' (augmentation + normalisation) or 'eval' (normalise only).
  mean/std: IMAGENET_MEAN/STD, or compute_dataset_stats() output for scratch; caller
            decides which, based on training mode and, for scratch, the specific
            arm/fold subset being trained on.

  Only the 'eval' path is implemented in this module -- see module
  docstring for why.
  """
  if mode == 'train':
    raise NotImplementedError(
        "build_transform(mode='train') requires build_train_augmentations(), "
        "which was deliberately not extracted into this module (out of scope "
        "for Notebook 04/05, which are eval-only). If training-mode transforms "
        "are ever needed again, pull build_train_augmentations() over from "
        "03_training.ipynb's Section 3.2 first."
    )
  steps = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
  return transforms.Compose(steps)


def compute_dataset_stats(df, base_dir, batch_size=32):
  """
  Computes per-channel mean/std over a set of images, for scratch-mode normalisation.

  In 03_training.ipynb, called fresh for each (arm, fold) training subset during
  sweep/CV, and once per arm over the FULL busi_cv_85_pool (all folds, unfiltered)
  for the final retrain. To recover the exact normalisation the final scratch-mode
  checkpoints were actually trained with, callers must reproduce that final-retrain
  call precisely: df=busi_cv_85_pool with NO fold filtering.
  """
  stats_transform = transforms.Compose([transforms.ToTensor()])
  stats_ds = BreastUltrasoundDataset(df=df, base_dir=base_dir, transform=stats_transform)
  stats_loader = DataLoader(stats_ds, batch_size=batch_size, shuffle=False, num_workers=2)

  channel_sum = torch.zeros(3)
  channel_sum_sq = torch.zeros(3)
  n_pixels = 0

  for images, _ in stats_loader:
    b, c, h, w = images.shape
    channel_sum += images.sum(dim=[0, 2, 3])
    channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])
    n_pixels += b * h * w

  mean = (channel_sum / n_pixels).tolist()
  std = torch.sqrt(channel_sum_sq / n_pixels - torch.tensor(mean) ** 2).tolist()

  return mean, std


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------

def seed_worker(worker_id):
  """
  Without it, each DataLoader worker process inherits an unseeded RNG state:
  augmentation draws would differ run-to-run even with the global seed set, since
  that seed never propagates into worker subprocesses on its own. Inert for
  eval-mode usage (no augmentation randomness to seed), kept for exact parity
  with 03_training.ipynb.
  """
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


def build_dataloader(dataset, train):
  return DataLoader(
      dataset,
      batch_size=BATCH_SIZE,
      shuffle=train,
      num_workers=NUM_WORKERS,
      pin_memory=True,
      worker_init_fn=seed_worker,
      generator=torch.Generator().manual_seed(SEED),
  )
