"""
Model architecture construction and layer-freezing utilities.

Extracted from 03_training.ipynb (Model Architecture section) so that
Notebook 04 (evaluation) and Notebook 05 (XAI/GradCAM) can rebuild the
exact architectures used to produce every checkpoint in final_models.json,
without duplicating this code notebook-to-notebook.

Every checkpoint saved via `save_best_checkpoint` (03_training.ipynb,
Section 5.5) is weights-only (`model.state_dict()`). To load one, a model
must first be constructed here with the exact same architecture, dropout,
and freeze configuration it was trained with -- get the architecture
wrong and `load_state_dict` either throws a shape mismatch or, worse,
loads silently into mismatched layers.

03_training.ipynb itself is left untouched -- it already ran successfully
against its own inline copy of this code, and there's no reason to risk
re-running anything that already produced the 18 final checkpoints.
"""

import torch.nn as nn
import torchvision.models as models


NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Architecture builders
# ---------------------------------------------------------------------------

def build_resnet50(pretrained, dropout):
  weights = models.ResNet50_Weights.DEFAULT if pretrained else None
  model = models.resnet50(weights=weights)
  model.fc = nn.Sequential(
      nn.Dropout(p=dropout),
      nn.Linear(model.fc.in_features, NUM_CLASSES),
  )
  return model


def build_efficientnet_b0(pretrained, dropout):
  weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
  model = models.efficientnet_b0(weights=weights)
  model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
  model.classifier[0].p = dropout
  return model


def build_swin_t(pretrained, dropout):
  """
  Unlike build_resnet50/build_efficientnet_b0, `dropout` here is NOT
  applied at the classifier head. torchvision's SwinTransformer applies
  it as general dropout inside the MLP layers of every transformer block
  throughout the backbone; model.head below is a bare nn.Linear with no
  dropout wrapper of its own.

  This is a deliberate, informed per-architecture choice -- each
  architecture's dropout follows its own family's established
  regularisation convention (head-level for the CNNs, body-wide for the
  transformer) rather than being forced into an identical placement
  across all three. Documented here explicitly so the same numeric
  dropout value from the hyperparameter sweep (e.g. 0.159 for
  swin_tiny/TL vs. 0.5 for resnet50/TL) is never read as doing the same
  job across architectures -- see Supplementary S4 for the full reasoning.
  """
  weights = models.Swin_T_Weights.DEFAULT if pretrained else None
  model = models.swin_t(weights=weights, dropout=dropout)
  model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
  return model


ARCHITECTURE_BUILDERS = {
    'resnet50': build_resnet50,
    'efficientnet_b0': build_efficientnet_b0,
    'swin_tiny': build_swin_t,
}


# ---------------------------------------------------------------------------
# Layer freezing (transfer learning only)
# ---------------------------------------------------------------------------

def freeze_resnet50(model, category):
  for param in model.parameters():
    param.requires_grad = False

  for param in model.fc.parameters():
    param.requires_grad = True

  if category == 'last_1_block':
    for param in model.layer4.parameters():
      param.requires_grad = True
  elif category == 'last_2_blocks':
    for param in model.layer3.parameters():
      param.requires_grad = True
    for param in model.layer4.parameters():
      param.requires_grad = True
  elif category != 'head_only':
    raise ValueError(f'Unknown category: {category}')

  return model


def freeze_efficientnet_b0(model, category):
  for param in model.parameters():
    param.requires_grad = False

  for param in model.classifier.parameters():
    param.requires_grad = True

  if category == 'last_1_block':
    for param in model.features[7].parameters():
      param.requires_grad = True
    for param in model.features[8].parameters():
      param.requires_grad = True
  elif category == 'last_2_blocks':
    for param in model.features[6].parameters():
      param.requires_grad = True
    for param in model.features[7].parameters():
      param.requires_grad = True
    for param in model.features[8].parameters():
      param.requires_grad = True
  elif category != 'head_only':
    raise ValueError(f'Unknown category: {category}')

  return model


def freeze_swin_t(model, category):
  for param in model.parameters():
    param.requires_grad = False

  for param in model.head.parameters():
    param.requires_grad = True

  if category == 'last_1_block':
    for param in model.features[7].parameters():
      param.requires_grad = True
  elif category == 'last_2_blocks':
    for param in model.features[5].parameters():
      param.requires_grad = True
    for param in model.features[6].parameters():
      param.requires_grad = True
    for param in model.features[7].parameters():
      param.requires_grad = True
  elif category != 'head_only':
    raise ValueError(f'Unknown category: {category}')

  return model


FREEZE_FUNCTIONS = {
    'resnet50': freeze_resnet50,
    'efficientnet_b0': freeze_efficientnet_b0,
    'swin_tiny': freeze_swin_t,
}


# ---------------------------------------------------------------------------
# Unified entry points
# ---------------------------------------------------------------------------

def build_model(architecture, mode, dropout, unfreeze_category=None):
  """
  architecture        : 'resnet50' | 'efficientnet_b0' | 'swin_tiny'
  mode                : 'TL' | 'scratch'
  unfreeze_category   : 'head_only' | 'last_1_block' | 'last_2_blocks'
                          required if mode='TL', ignored (must be None)
                          if mode='scratch'.
  """
  if mode == 'TL':
    if unfreeze_category is None:
      raise ValueError("mode='TL' requires an unfreeze_category.")
    model = ARCHITECTURE_BUILDERS[architecture](pretrained=True, dropout=dropout)
    model = FREEZE_FUNCTIONS[architecture](model, unfreeze_category)
  elif mode == 'scratch':
    if unfreeze_category is not None:
      raise ValueError("mode='scratch' does not use unfreeze_category.")
    model = ARCHITECTURE_BUILDERS[architecture](pretrained=False, dropout=dropout)
  else:
    raise ValueError(f'Unknown mode: {mode}')

  return model


def get_output_linear(model):
  """
  Returns the actual Linear output layer, regardless of whether it's
  wrapped in a Sequential (ResNet's fc, EfficientNet's classifier) or
  standalone (Swin's head).
  """
  candidate = model.fc if hasattr(model, 'fc') else (
      model.classifier if hasattr(model, 'classifier') else model.head
  )
  if isinstance(candidate, nn.Sequential):
    linear_layers = [m for m in candidate if isinstance(m, nn.Linear)]
    return linear_layers[-1]
  return candidate
