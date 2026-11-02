"""
patient_bootstrap.py -- patient-level (cluster) bootstrap for BUS-UCLM.

Design rule: these functions are the ORIGINAL Notebook 04 bootstrap with exactly
one change -- the resampling unit. The original resamples images within each
class; this resamples whole patients (clusters) within each stratum.

"Strata" is an ordered list; each stratum is a list of clusters; each cluster is
a numpy array of row indices (the images of one patient).
  - Original behaviour:  strata = classes (benign, malignant, normal), every
    image its own cluster  -> image_level_strata(). Must reproduce Notebook 04
    exactly (the identity test).
  - Patient level:       strata = (has_malignant, no_malignant), every patient a
    cluster                -> patient_level_strata().

Metric functions, percentile CI, p-value rule and seed are reused unchanged
from original_functions.py.
"""
import numpy as np

from original_functions import SEED


# ---------------- building strata ----------------
def image_level_strata(y_true):
  """Original Notebook 04 resampling, expressed as strata of single-image clusters:
  one stratum per class (0, 1, 2 in that order), ascending row index, empty
  classes skipped -- the same order and skipping as stratified_resample_indices()."""
  y_true = np.asarray(y_true)
  strata = []
  for cls in range(3):
    cls_idx = np.where(y_true == cls)[0]
    if len(cls_idx) == 0:
      continue
    strata.append([np.array([i]) for i in cls_idx])
  return strata


STRATUM_ORDER = ['has_malignant', 'no_malignant']   # fixed order, recorded in Methods


def patient_level_strata(patient_ids, stratum_labels):
  """Patient-level strata. patient_ids and stratum_labels are per-row arrays
  aligned with the comparison's rows (same order as y_true). Strata in
  STRATUM_ORDER; patients sorted by ID inside each stratum; a patient's rows
  in ascending row order."""
  patient_ids = np.asarray(patient_ids)
  stratum_labels = np.asarray(stratum_labels)
  strata = []
  for stratum in STRATUM_ORDER:
    in_stratum = stratum_labels == stratum
    patients = np.unique(patient_ids[in_stratum])            # sorted
    clusters = [np.where(patient_ids == p)[0] for p in patients]
    for p, rows in zip(patients, clusters):                 # a patient must not span strata
      if not np.all(stratum_labels[rows] == stratum):
        raise ValueError(f'Patient {p} has rows in more than one stratum.')
    strata.append(clusters)
  n_rows = sum(len(c) for s in strata for c in s)
  if n_rows != len(patient_ids):
    raise ValueError(f'Strata cover {n_rows} rows, expected {len(patient_ids)}.')
  return strata


# ================= CORE: one cluster resample =================
def cluster_resample_indices(strata, rng):
  """One stratified cluster resample: in each stratum, draw as many clusters
  as it has, with replacement, and take ALL rows of each drawn cluster.
  With single-image clusters this makes the same random draws, in the same
  order, as the original stratified_resample_indices()."""
  resampled = []
  for clusters in strata:
    drawn = rng.choice(len(clusters), size=len(clusters), replace=True)
    resampled.extend(clusters[k] for k in drawn)
  return np.concatenate(resampled)
# ==============================================================


def _require_all_classes(y_resampled):
  # macro-F1, macro-AUC and sensitivity need every class present. With patient
  # resampling this is not guaranteed by construction -> fail loudly, never skip.
  if len(np.unique(y_resampled)) < 3:
    raise ValueError('A resample is missing a class; cannot compute macro metrics.')


def paired_cluster_bootstrap_diff(y_true, values_a, values_b, metric_fn, strata,
                                  n_boot=10000, seed=SEED):
  """Original paired_bootstrap_diff, resampling clusters instead of images."""
  rng = np.random.default_rng(seed)
  y_true, values_a, values_b = np.asarray(y_true), np.asarray(values_a), np.asarray(values_b)
  point_diff = metric_fn(y_true, values_a) - metric_fn(y_true, values_b)

  diffs = np.empty(n_boot)
  for i in range(n_boot):
    idx = cluster_resample_indices(strata, rng)
    _require_all_classes(y_true[idx])
    diffs[i] = metric_fn(y_true[idx], values_a[idx]) - metric_fn(y_true[idx], values_b[idx])

  p_value = min(2 * min((diffs >= 0).mean(), (diffs <= 0).mean()), 1.0)
  p_value = max(p_value, 1 / n_boot)
  return {'point_estimate': point_diff, 'ci_lower': np.percentile(diffs, 2.5),
          'ci_upper': np.percentile(diffs, 97.5), 'p_value': p_value}


def unpaired_cluster_bootstrap_diff(y_true_a, values_a, strata_a, y_true_b, values_b, strata_b,
                                    metric_fn, n_boot=10000, seed=SEED):
  """Original unpaired_bootstrap_diff; each side resampled with its own strata
  (BUSI test: image_level_strata; BUS-UCLM: patient_level_strata). Draw order
  per iteration unchanged: side A, then side B."""
  rng = np.random.default_rng(seed)
  y_true_a, values_a = np.asarray(y_true_a), np.asarray(values_a)
  y_true_b, values_b = np.asarray(y_true_b), np.asarray(values_b)
  point_diff = metric_fn(y_true_a, values_a) - metric_fn(y_true_b, values_b)

  diffs = np.empty(n_boot)
  for i in range(n_boot):
    idx_a = cluster_resample_indices(strata_a, rng)
    idx_b = cluster_resample_indices(strata_b, rng)
    _require_all_classes(y_true_a[idx_a])
    _require_all_classes(y_true_b[idx_b])
    diffs[i] = metric_fn(y_true_a[idx_a], values_a[idx_a]) - metric_fn(y_true_b[idx_b], values_b[idx_b])

  p_value = min(2 * min((diffs >= 0).mean(), (diffs <= 0).mean()), 1.0)
  p_value = max(p_value, 1 / n_boot)
  return {'point_estimate': point_diff, 'ci_lower': np.percentile(diffs, 2.5),
          'ci_upper': np.percentile(diffs, 97.5), 'p_value': p_value}


# ---------------- attaching patient IDs to a comparison's rows ----------------
def attach_patient_info(cond, ids):
  """cond: one condition's rows exactly as get_predictions_for_condition()
  returns them (sorted by image_name). ids: data/new/uclm_patient_ids.csv.
  Returns (patient_ids, strata_labels) aligned row-for-row with cond, after
  checking the alignment instead of assuming it."""
  merged = cond[['image_name', 'true_label']].merge(
      ids, on='image_name', how='left', validate='one_to_one', suffixes=('', '_ids'))
  if len(merged) != len(cond) or merged['patient_id'].isna().any():
    raise ValueError('Some images have no patient ID.')
  if not (merged['image_name'].values == cond['image_name'].values).all():
    raise ValueError('Row order changed while attaching patient IDs.')
  if not (merged['true_label'].values == merged['true_label_ids'].values).all():
    raise ValueError('True labels disagree with uclm_patient_ids.csv.')
  if not (merged['patient_id'].values == merged['image_name'].str[:4].values).all():
    raise ValueError('A patient ID does not equal its filename prefix after alignment.')
  return merged['patient_id'].values, merged['stratum'].values
