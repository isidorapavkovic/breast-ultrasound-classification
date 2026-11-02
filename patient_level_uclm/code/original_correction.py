"""
original_correction.py -- VERBATIM copies of the multiple-comparison correction from
04_evaluation.ipynb, Section 6.6 (cells 98-101).

Cells 98 (functions) and 99 (their verification) are copied unchanged.
Cells 100 (the procedure) and 101 (its checks) were top-level notebook code working
on a variable named stats_raw; here each is wrapped in a function. The ONLY change is
the added def line, indentation, and a return line (all marked NEW); the original
lines are otherwise character-for-character identical (checked by script).
"""
import numpy as np   # NEW: import (the notebook imported it in cell 5)

# NEW: cell 101 uses P_COLUMNS, which cell 100 defines; inside separate functions it must
# exist at module level too. Identical to the list in cell 100 (checked below).
P_COLUMNS = ['mcnemar_p', 'delong_benign_p', 'delong_malignant_p', 'delong_normal_p',
             'bootstrap_macro_f1_p', 'bootstrap_sensitivity90_p', 'bootstrap_macro_auc_p']


# ================= VERBATIM: Notebook 04, cell 98 =================
def holm_bonferroni(p_values, alpha=0.05):
  p_values = np.asarray(p_values, dtype=float)
  m = len(p_values)
  order = np.argsort(p_values)
  sorted_p = p_values[order]
  terms = np.minimum(1.0, (m - np.arange(m)) * sorted_p)
  adjusted_sorted = np.maximum.accumulate(terms)
  adjusted = np.empty(m)
  adjusted[order] = adjusted_sorted
  return adjusted, adjusted < alpha


def benjamini_hochberg(p_values, alpha=0.05):
  p_values = np.asarray(p_values, dtype=float)
  m = len(p_values)
  order = np.argsort(p_values)
  sorted_p = p_values[order]
  ranks = np.arange(1, m + 1)
  terms = np.minimum(1.0, (m / ranks) * sorted_p)
  adjusted_sorted = np.minimum.accumulate(terms[::-1])[::-1]
  adjusted = np.empty(m)
  adjusted[order] = adjusted_sorted
  return adjusted, adjusted < alpha


# ================= VERBATIM: Notebook 04, cell 99 (wrapped) =================
def verify_correction_functions():   # NEW: wrapper
  # Published reference (adjustp, Rust crate, doctested), independently hand-verified beforehand.
  p_ref = [0.1, 0.2, 0.3, 0.4, 0.1]
  expected_bh_ref = [0.25, 0.33333333333333337, 0.375, 0.4, 0.25]
  expected_holm_hand = [0.5, 0.6, 0.6, 0.6, 0.5]  # hand-derived, same test vector

  bh_adj, _ = benjamini_hochberg(p_ref)
  holm_adj, _ = holm_bonferroni(p_ref)

  checks = [
      ('BH matches published reference', np.allclose(bh_adj, expected_bh_ref)),
      ('Holm matches hand-derivation', np.allclose(holm_adj, expected_holm_hand)),
  ]

  rng_check = np.random.default_rng(3)
  for m_test in [27, 39, 108, 132]:  # actual group sizes used below
    p_rand = rng_check.beta(0.5, 5, size=m_test)
    h_adj, _ = holm_bonferroni(p_rand)
    b_adj, _ = benjamini_hochberg(p_rand)
    checks.append((f'Holm adjusted >= raw p, m={m_test}', np.all(h_adj >= p_rand - 1e-12)))
    checks.append((f'BH adjusted >= raw p, m={m_test}', np.all(b_adj >= p_rand - 1e-12)))

  all_passed = True
  for description, passed in checks:
    print(f'[{"PASS" if passed else "FAIL"}] {description}')
    if not passed:
      all_passed = False

  if not all_passed:
    raise ValueError('Correction function verification failed.')
  print('\nHolm-Bonferroni and Benjamini-Hochberg verified.')


# ================= VERBATIM: Notebook 04, cell 100 (wrapped) =================
def apply_correction(stats_raw):   # NEW: wrapper
  CORRECTION_GROUPS = [
      ('mcnemar_p', 'confirmatory', 'type1_paired'), ('delong_benign_p', 'confirmatory', 'type1_paired'),
      ('delong_malignant_p', 'confirmatory', 'type1_paired'), ('delong_normal_p', 'confirmatory', 'type1_paired'),
      ('bootstrap_macro_f1_p', 'confirmatory', None), ('bootstrap_sensitivity90_p', 'confirmatory', None),
      ('bootstrap_macro_auc_p', 'confirmatory', None),
      ('mcnemar_p', 'exploratory', 'type1_paired'), ('delong_benign_p', 'exploratory', 'type1_paired'),
      ('delong_malignant_p', 'exploratory', 'type1_paired'), ('delong_normal_p', 'exploratory', 'type1_paired'),
      ('bootstrap_macro_f1_p', 'exploratory', None), ('bootstrap_sensitivity90_p', 'exploratory', None),
      ('bootstrap_macro_auc_p', 'exploratory', None),
  ]
  P_COLUMNS = ['mcnemar_p', 'delong_benign_p', 'delong_malignant_p', 'delong_normal_p',
               'bootstrap_macro_f1_p', 'bootstrap_sensitivity90_p', 'bootstrap_macro_auc_p']

  stats_corrected = stats_raw.copy()
  for col in P_COLUMNS:
    stats_corrected[f'{col}_adj'] = np.nan
    stats_corrected[f'{col}_significant'] = False

  group_sizes = {}
  for p_col, family, pairing_filter in CORRECTION_GROUPS:
    mask = (stats_corrected['family'] == family) & stats_corrected[p_col].notna()
    if pairing_filter is not None:
      mask &= stats_corrected['pairing_type'] == pairing_filter

    p_vals = stats_corrected.loc[mask, p_col].values
    method = 'holm' if family == 'confirmatory' else 'BH'
    adj, sig = (holm_bonferroni(p_vals) if method == 'holm' else benjamini_hochberg(p_vals))

    stats_corrected.loc[mask, f'{p_col}_adj'] = adj
    stats_corrected.loc[mask, f'{p_col}_significant'] = sig
    group_sizes[(p_col, family)] = (method, mask.sum())

  for k, (method, n) in group_sizes.items():
    print(f'{k}: method={method}, n={n}')
  return stats_corrected, group_sizes   # NEW


# ================= VERBATIM: Notebook 04, cell 101 (wrapped) =================
def verify_correction(stats_corrected, group_sizes):   # NEW: wrapper
  expected_sizes = {
      ('mcnemar_p', 'confirmatory'): 27, ('delong_benign_p', 'confirmatory'): 27,
      ('delong_malignant_p', 'confirmatory'): 27, ('delong_normal_p', 'confirmatory'): 27,
      ('bootstrap_macro_f1_p', 'confirmatory'): 39, ('bootstrap_sensitivity90_p', 'confirmatory'): 39,
      ('bootstrap_macro_auc_p', 'confirmatory'): 39,
      ('mcnemar_p', 'exploratory'): 108, ('delong_benign_p', 'exploratory'): 108,
      ('delong_malignant_p', 'exploratory'): 108, ('delong_normal_p', 'exploratory'): 108,
      ('bootstrap_macro_f1_p', 'exploratory'): 132, ('bootstrap_sensitivity90_p', 'exploratory'): 132,
      ('bootstrap_macro_auc_p', 'exploratory'): 132,
  }

  checks = []
  for k, expected_n in expected_sizes.items():
    checks.append((f'{k} group size', group_sizes[k][1] == expected_n))

  adj_cols = [f'{c}_adj' for c in P_COLUMNS]
  raw_adj_pairs = [(c, f'{c}_adj') for c in P_COLUMNS]
  for raw_col, adj_col in raw_adj_pairs:
    valid = stats_corrected[raw_col].notna()
    checks.append((f'{adj_col} >= {raw_col} everywhere valid',
                    (stats_corrected.loc[valid, adj_col] >= stats_corrected.loc[valid, raw_col] - 1e-9).all()))
    checks.append((f'{adj_col} in [0,1]', stats_corrected.loc[valid, adj_col].between(0, 1).all()))

  all_passed = True
  for description, passed in checks:
    print(f'[{"PASS" if passed else "FAIL"}] {description}')
    if not passed:
      all_passed = False

  if not all_passed:
    raise ValueError('6.6 verification failed.')
  print('\n6.6 verified: all 14 correction groups correctly sized, adjusted p-values valid.')
