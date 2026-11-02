"""
original_functions.py -- VERBATIM copies of the statistical functions from
04_evaluation.ipynb (Section 5.1 cell 72; Section 6 cells 81, 84, 88, 91).

DO NOT EDIT the copied function bodies. This file is the frozen reference
implementation: the environment check proves it reproduces the saved
statistical_results_raw.csv, and later tests compare the new patient-level
code against it.

Only additions (marked NEW, not from Notebook 04):
  - the import block (the notebook imported these in cell 5),
  - SEED, CLASS_TO_IDX (the notebook took these from src/utils.py and
    src/dataset_utils.py, which import torch; defined directly here),
  - predictions_df = None, plus set_predictions(), because the copied
    get_predictions_for_condition() reads a global named predictions_df.
"""
# ---- NEW: imports (same as Notebook 04 cell 5, minus unused ones) ----
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from scipy import stats as scipy_stats

# ---- NEW: constants (values from src/utils.py and src/dataset_utils.py) ----
SEED = 42
CLASS_TO_IDX = {'benign': 0, 'malignant': 1, 'normal': 2}
CLASS_NAMES = ['benign', 'malignant', 'normal']   # Notebook 04 cell 60

# ---- NEW: global that get_predictions_for_condition() reads ----
predictions_df = None

def set_predictions(df):
  """Make the loaded predictions table visible to the verbatim functions."""
  global predictions_df
  predictions_df = df


# ================= VERBATIM: Notebook 04, cell 72 =================
MALIGNANT_IDX = CLASS_TO_IDX['malignant']
TARGET_SPECIFICITIES = [0.85, 0.90, 0.95]

def compute_sensitivity_at_specificity(y_true_binary, y_score, targets=TARGET_SPECIFICITIES):
  fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, drop_intermediate=False)
  finite = np.isfinite(thresholds)
  fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
  specificity = 1 - fpr

  # Vertical segments (tied specificity, differing sensitivity) break
  # np.interp's monotonicity assumption. Collapse to one row per unique
  # specificity -- max tpr (best achievable), paired with min threshold
  # (the one that actually achieves it; within a tie group, lower
  # threshold always corresponds to higher tpr).
  # Numpy-only version; ~8x faster than the orignal groupby version.
  unique_spec, inverse = np.unique(specificity, return_inverse=True)
  agg_tpr = np.full(len(unique_spec), -np.inf)
  agg_thresh = np.full(len(unique_spec), np.inf)
  np.maximum.at(agg_tpr, inverse, tpr)
  np.minimum.at(agg_thresh, inverse, thresholds)

  if not np.all(np.diff(agg_tpr) <= 1e-9):
    raise ValueError('tpr not monotonically non-increasing after de-duplication -- do not trust interpolation.')

  results = {}
  for target in targets:
    results[target] = {
        'sensitivity': float(np.interp(target, unique_spec, agg_tpr)),
        'threshold': float(np.interp(target, unique_spec, agg_thresh)),
    }
  return results


# ================= VERBATIM: Notebook 04, cell 81 =================
def mcnemar_test(correct_a, correct_b, exact_threshold=25):
  correct_a = np.asarray(correct_a, dtype=bool)
  correct_b = np.asarray(correct_b, dtype=bool)
  if len(correct_a) != len(correct_b):
    raise ValueError('Arrays must be the same length (paired).')

  b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
  c = int(np.sum(~correct_a & correct_b))   # A wrong, B right
  n_discordant = b + c

  if n_discordant == 0:
    return {'b': b, 'c': c, 'n_discordant': 0, 'method': 'undefined', 'statistic': None, 'p_value': 1.0}

  if n_discordant < exact_threshold:
    k = min(b, c)
    p_value = scipy_stats.binomtest(k, n_discordant, 0.5, alternative='two-sided').pvalue
    return {'b': b, 'c': c, 'n_discordant': n_discordant, 'method': 'exact', 'statistic': None, 'p_value': p_value}
  else:
    chi2_stat = (abs(b - c) - 1) ** 2 / n_discordant
    p_value = scipy_stats.chi2.sf(chi2_stat, df=1)
    return {'b': b, 'c': c, 'n_discordant': n_discordant, 'method': 'chi2_corrected', 'statistic': chi2_stat, 'p_value': p_value}


# ================= VERBATIM: Notebook 04, cell 84 =================
def _psi_matrix(X, Y):
  """psi(x,y) = 1 if y<x, 0.5 if y==x, 0 if y>x -- reward when the positive-class
  score outranks the negative-class score. X: positive scores (m,), Y: negative
  scores (n,). Returns the full (m, n) comparison matrix via broadcasting."""
  diff = X[:, None] - Y[None, :]
  return np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))


def delong_test(y_true_binary, score_a, score_b):
  """DeLong's test for two correlated AUCs on the same binary outcome, same cases.
  Returns AUCs, z-statistic, and two-sided p-value."""
  y_true_binary = np.asarray(y_true_binary)
  score_a = np.asarray(score_a, dtype=float)
  score_b = np.asarray(score_b, dtype=float)

  Xa, Ya = score_a[y_true_binary == 1], score_a[y_true_binary == 0]
  Xb, Yb = score_b[y_true_binary == 1], score_b[y_true_binary == 0]
  m, n = len(Xa), len(Ya)

  psi_a, psi_b = _psi_matrix(Xa, Ya), _psi_matrix(Xb, Yb)
  V10_a, V01_a = psi_a.mean(axis=1), psi_a.mean(axis=0)
  V10_b, V01_b = psi_b.mean(axis=1), psi_b.mean(axis=0)

  auc_a, auc_b = V10_a.mean(), V10_b.mean()

  S10 = np.cov(np.stack([V10_a, V10_b], axis=1), rowvar=False, ddof=1)
  S01 = np.cov(np.stack([V01_a, V01_b], axis=1), rowvar=False, ddof=1)
  S = S10 / m + S01 / n

  var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
  z = (auc_a - auc_b) / np.sqrt(var_diff)
  p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

  return {'auc_a': auc_a, 'auc_b': auc_b, 'z': z, 'p_value': p_value}


# ================= VERBATIM: Notebook 04, cell 88 =================
def stratified_resample_indices(y_true, rng):
  """One stratified resample: each class resampled with replacement from its
  own pool, at its own original size. Guarantees every class is represented
  in every resample, regardless of how rare it is."""
  y_true = np.asarray(y_true)
  resampled = []
  for cls in range(3):
    cls_idx = np.where(y_true == cls)[0]
    if len(cls_idx) == 0:
      continue
    resampled.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
  return np.concatenate(resampled)


def metric_macro_f1(y_true, pred_label):
  return f1_score(y_true, pred_label, average='macro', zero_division=0, labels=[0, 1, 2])


def metric_malignant_sensitivity_90(y_true, y_prob):
  y_true_binary = (y_true == MALIGNANT_IDX).astype(int)
  return compute_sensitivity_at_specificity(y_true_binary, y_prob[:, MALIGNANT_IDX], targets=[0.90])[0.90]['sensitivity']


def metric_macro_auc(y_true, y_prob):
  aucs = [roc_auc_score((y_true == i).astype(int), y_prob[:, i]) for i in range(3)]
  return np.mean(aucs)


def paired_bootstrap_diff(y_true, values_a, values_b, metric_fn, n_boot=10000, seed=SEED):
  """Type 1 only -- same resampled indices applied to both A and B each
  iteration, preserving correlation between their errors on the same images."""
  rng = np.random.default_rng(seed)
  y_true, values_a, values_b = np.asarray(y_true), np.asarray(values_a), np.asarray(values_b)
  point_diff = metric_fn(y_true, values_a) - metric_fn(y_true, values_b)

  diffs = np.empty(n_boot)
  for i in range(n_boot):
    idx = stratified_resample_indices(y_true, rng)
    diffs[i] = metric_fn(y_true[idx], values_a[idx]) - metric_fn(y_true[idx], values_b[idx])

  p_value = min(2 * min((diffs >= 0).mean(), (diffs <= 0).mean()), 1.0)
  p_value = max(p_value, 1 / n_boot)  # floor: a bootstrap p-value can't resolve below ~1/n_boot
  return {'point_estimate': point_diff, 'ci_lower': np.percentile(diffs, 2.5),
          'ci_upper': np.percentile(diffs, 97.5), 'p_value': p_value}


def unpaired_bootstrap_diff(y_true_a, values_a, y_true_b, values_b, metric_fn, n_boot=10000, seed=SEED):
  """Type 2 only: internal-vs-external, different images entirely. Each
  dataset resampled independently; pairing is impossible, not applicable."""
  rng = np.random.default_rng(seed)
  y_true_a, values_a = np.asarray(y_true_a), np.asarray(values_a)
  y_true_b, values_b = np.asarray(y_true_b), np.asarray(values_b)
  point_diff = metric_fn(y_true_a, values_a) - metric_fn(y_true_b, values_b)

  diffs = np.empty(n_boot)
  for i in range(n_boot):
    idx_a = stratified_resample_indices(y_true_a, rng)
    idx_b = stratified_resample_indices(y_true_b, rng)
    diffs[i] = metric_fn(y_true_a[idx_a], values_a[idx_a]) - metric_fn(y_true_b[idx_b], values_b[idx_b])

  p_value = min(2 * min((diffs >= 0).mean(), (diffs <= 0).mean()), 1.0)
  p_value = max(p_value, 1 / n_boot)
  return {'point_estimate': point_diff, 'ci_lower': np.percentile(diffs, 2.5),
          'ci_upper': np.percentile(diffs, 97.5), 'p_value': p_value}


# ===== VERBATIM: Notebook 04, cell 91 (only these two functions) =====
def get_predictions_for_condition(architecture, mode, arm, eval_data):
  group = predictions_df[(predictions_df['architecture'] == architecture) & (predictions_df['mode'] == mode) &
                          (predictions_df['arm'] == arm) & (predictions_df['eval_data'] == eval_data)]
  return group.sort_values('image_name').reset_index(drop=True)


def run_comparison(row):
  cond_a = get_predictions_for_condition(row['arch_a'], row['mode_a'], row['arm_a'], row['eval_data_a'])
  cond_b = get_predictions_for_condition(row['arch_b'], row['mode_b'], row['arm_b'], row['eval_data_b'])
  result = {
      'comparison_id': row['comparison_id'], 'comparison_type': row['comparison_type'],
      'family': row['family'], 'pairing_type': row['pairing_type'],
      'arch_a': row['arch_a'], 'mode_a': row['mode_a'], 'arm_a': row['arm_a'], 'eval_data_a': row['eval_data_a'],
      'arch_b': row['arch_b'], 'mode_b': row['mode_b'], 'arm_b': row['arm_b'], 'eval_data_b': row['eval_data_b'],
  }

  if row['pairing_type'] == 'type1_paired':
    if cond_a['image_name'].tolist() != cond_b['image_name'].tolist():
      raise ValueError(f"comparison_id {row['comparison_id']}: image_name mismatch between paired conditions.")

    y_true = cond_a['true_label'].values
    pred_a, pred_b = cond_a['pred_label'].values, cond_b['pred_label'].values
    prob_a = cond_a[['prob_benign', 'prob_malignant', 'prob_normal']].values
    prob_b = cond_b[['prob_benign', 'prob_malignant', 'prob_normal']].values

    mcnemar_result = mcnemar_test(pred_a == y_true, pred_b == y_true)
    result['mcnemar_p'] = mcnemar_result['p_value']
    result['mcnemar_method'] = mcnemar_result['method']

    for i, cls in enumerate(CLASS_NAMES):
      y_true_binary = (y_true == i).astype(int)
      if len(np.unique(y_true_binary)) < 2:
        result[f'delong_{cls}_p'] = np.nan
      else:
        result[f'delong_{cls}_p'] = delong_test(y_true_binary, prob_a[:, i], prob_b[:, i])['p_value']

    boot_f1 = paired_bootstrap_diff(y_true, pred_a, pred_b, metric_macro_f1, n_boot=10000)
    boot_sens = paired_bootstrap_diff(y_true, prob_a, prob_b, metric_malignant_sensitivity_90, n_boot=10000)
    boot_auc = paired_bootstrap_diff(y_true, prob_a, prob_b, metric_macro_auc, n_boot=10000)

  else:  # type2_unpaired
    y_true_a, y_true_b = cond_a['true_label'].values, cond_b['true_label'].values
    pred_a, pred_b = cond_a['pred_label'].values, cond_b['pred_label'].values
    prob_a = cond_a[['prob_benign', 'prob_malignant', 'prob_normal']].values
    prob_b = cond_b[['prob_benign', 'prob_malignant', 'prob_normal']].values

    result['mcnemar_p'] = np.nan
    result['mcnemar_method'] = 'not_applicable'
    for cls in CLASS_NAMES:
      result[f'delong_{cls}_p'] = np.nan

    boot_f1 = unpaired_bootstrap_diff(y_true_a, pred_a, y_true_b, pred_b, metric_macro_f1, n_boot=10000)
    boot_sens = unpaired_bootstrap_diff(y_true_a, prob_a, y_true_b, prob_b, metric_malignant_sensitivity_90, n_boot=10000)
    boot_auc = unpaired_bootstrap_diff(y_true_a, prob_a, y_true_b, prob_b, metric_macro_auc, n_boot=10000)

  for metric_name, boot in [('macro_f1', boot_f1), ('sensitivity90', boot_sens), ('macro_auc', boot_auc)]:
    result[f'bootstrap_{metric_name}_diff'] = boot['point_estimate']
    result[f'bootstrap_{metric_name}_ci_lower'] = boot['ci_lower']
    result[f'bootstrap_{metric_name}_ci_upper'] = boot['ci_upper']
    result[f'bootstrap_{metric_name}_p'] = boot['p_value']

  return result
