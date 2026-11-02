"""
Small reusable utilities shared across notebooks: global reproducibility
seeding and WandB run-identity construction.

Extracted from 03_training.ipynb so Notebook 04 (and later Notebook 05)
import this instead of maintaining their own copies.

03_training.ipynb itself is left untouched -- same reasoning as
src/models.py: it already ran successfully against its own inline copy,
and there's no reason to re-run anything that already produced the 18
final checkpoints.
"""

import random
import numpy as np
import torch


SEED = 42


def seed_everything(seed=SEED):
  """
  Seeds every RNG source PyTorch code can touch. Called once at notebook
  start for global reproducibility, and again inside each training run /
  resampling procedure so results don't depend on call order or prior
  runs.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  torch.backends.cudnn.deterministic = False
  torch.backends.cudnn.benchmark = True


def build_run_identity(architecture, mode, arm, job_type, fold=None, eval_data=None):
  """
  Builds a consistent (group, job_type, tags, name) tuple for any WandB
  run in this project, so every sweep trial, CV fold, retrain, and eval
  run is filterable and human-readable in the WandB UI.

  architecture : 'resnet50' | 'efficientnet_b0' | 'swin_tiny'
  mode         : 'TL' | 'scratch'
  arm          : 'annotated' | 'inpainted' | 'masked'
  job_type     : 'sweep' | 'cv_fold' | 'final_retrain' | 'eval'
  fold         : int, only for job_type='cv_fold'
  eval_data    : 'busi_test' | 'bus_uclm' | 'breast', only for job_type='eval'
  """
  group = f'{architecture}_{mode}_{arm}'
  tags = [f'arch:{architecture}', f'mode:{mode}', f'arm:{arm}']

  name = group
  if job_type == 'cv_fold':
    if fold is None:
      raise ValueError("job_type='cv_fold' requires a fold number.")
    tags.append(f'fold:{fold}')
    name += f'_fold{fold}'
  elif job_type == 'eval':
    if eval_data is None:
      raise ValueError("job_type='eval' requires eval_data.")
    tags.append(f'eval_data:{eval_data}')
    name += f'_eval_{eval_data}'
  else:
    name += f'_{job_type}'

  return {'group': group, 'job_type': job_type, 'tags': tags, 'name': name}
