"""
test_patient_bootstrap.py -- prove patient_bootstrap.py is correct BEFORE it
produces any real patient-level number.

  T1 Identity sweep   All 63 BUS-UCLM comparisons x 3 metrics, short runs:
                      new code with every image as its own cluster must equal
                      the original Notebook 04 functions EXACTLY.
  T2 Identity, full   Comparisons 1 and 27 at the real setting (10,000
                      resamples): exact equality again.
  T3 Duplication      Every image copied 5 times, copies form one cluster. The
                      cluster bootstrap on the copies must give the original
                      result on the un-copied data (copies add no information),
                      while the ordinary image bootstrap on the copies must give
                      a much narrower, overconfident interval.
  T4 Structure        Real BUS-UCLM patient strata: 17 + 20 patients, every
                      resample made of WHOLE patients only, IDs aligned with
                      filenames, same seed -> same resamples.

No real patient-level results are computed or saved. Prints a verdict; writes
data/new/bootstrap_tests.csv.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import original_functions as of
import patient_bootstrap as pb

BASE = Path(os.environ.get('UCLM_BASE', Path.home() / 'uclm_rerun'))
ORIG = BASE / 'data' / 'original'
NEW = BASE / 'data' / 'new'
N_SWEEP = int(os.environ.get('UCLM_N_SWEEP', 200))     # env vars: for testing the tests only
N_FULL = int(os.environ.get('UCLM_N_FULL', 10000))
N_DUP = int(os.environ.get('UCLM_N_DUP', 1000))
PROB = ['prob_benign', 'prob_malignant', 'prob_normal']
METRICS = {   # metric name -> (function, which values it uses)
    'macro_f1': (of.metric_macro_f1, 'pred'),
    'sensitivity90': (of.metric_malignant_sensitivity_90, 'prob'),
    'macro_auc': (of.metric_macro_auc, 'prob'),
}
KEYS = ['point_estimate', 'ci_lower', 'ci_upper', 'p_value']

predictions = pd.read_csv(ORIG / 'predictions.csv', float_precision='round_trip')
raw = pd.read_csv(ORIG / 'statistical_results_raw.csv', float_precision='round_trip')
ids = pd.read_csv(NEW / 'uclm_patient_ids.csv')
of.set_predictions(predictions)
uclm = raw[(raw['eval_data_a'] == 'bus_uclm') | (raw['eval_data_b'] == 'bus_uclm')]
if len(uclm) != 63:
  sys.exit(f'[STOP] Expected 63 BUS-UCLM comparisons, found {len(uclm)}.')

report, failures = [], []


def record(test, item, passed, detail=''):
  report.append({'test': test, 'item': item, 'passed': bool(passed), 'detail': detail})
  if not passed:
    failures.append(f'{test} | {item} | {detail}')


def data_for(row):
  a = of.get_predictions_for_condition(row['arch_a'], row['mode_a'], row['arm_a'], row['eval_data_a'])
  b = of.get_predictions_for_condition(row['arch_b'], row['mode_b'], row['arm_b'], row['eval_data_b'])
  return a, b


def values(cond, kind):
  return cond['pred_label'].values if kind == 'pred' else cond[PROB].values


def run_both(row, metric, n_boot):
  """Original function vs new function with single-image clusters."""
  fn, kind = METRICS[metric]
  a, b = data_for(row)
  if row['pairing_type'] == 'type1_paired':
    y = a['true_label'].values
    old = of.paired_bootstrap_diff(y, values(a, kind), values(b, kind), fn, n_boot=n_boot)
    new = pb.paired_cluster_bootstrap_diff(y, values(a, kind), values(b, kind), fn,
                                           pb.image_level_strata(y), n_boot=n_boot)
  else:
    ya, yb = a['true_label'].values, b['true_label'].values
    old = of.unpaired_bootstrap_diff(ya, values(a, kind), yb, values(b, kind), fn, n_boot=n_boot)
    new = pb.unpaired_cluster_bootstrap_diff(ya, values(a, kind), pb.image_level_strata(ya),
                                             yb, values(b, kind), pb.image_level_strata(yb),
                                             fn, n_boot=n_boot)
  return old, new


def exactly_equal(old, new):
  return all(old[k] == new[k] for k in KEYS)


# ---------------- T1: identity sweep ----------------
print(f'T1 identity sweep: 63 comparisons x 3 metrics, {N_SWEEP} resamples each ...')
t0 = time.perf_counter()
for _, row in uclm.iterrows():
  for metric in METRICS:
    old, new = run_both(row, metric, N_SWEEP)
    ok = exactly_equal(old, new)
    record('T1_identity_sweep', f'{row["comparison_id"]}:{metric}', ok,
           '' if ok else f'old={old} new={new}')
n_t1 = sum(r['test'] == 'T1_identity_sweep' for r in report)
n_t1_ok = sum(r['passed'] for r in report if r['test'] == 'T1_identity_sweep')
print(f'  {n_t1_ok}/{n_t1} exactly equal  ({time.perf_counter() - t0:.0f} s)')

# ---------------- T2: identity at full setting ----------------
print(f'\nT2 identity at {N_FULL} resamples: comparisons 1 and 27 ...')
for cid in [1, 27]:
  row = uclm[uclm['comparison_id'] == cid].iloc[0]
  for metric in METRICS:
    t0 = time.perf_counter()
    old, new = run_both(row, metric, N_FULL)
    ok = exactly_equal(old, new)
    record('T2_identity_full', f'{cid}:{metric}', ok, '' if ok else f'old={old} new={new}')
    print(f'  comparison {cid:2d} {metric:14s} {"EXACT" if ok else "DIFFERENT"}  '
          f'({time.perf_counter() - t0:.0f} s)')

# ---------------- T3: duplication known-answer ----------------
print(f'\nT3 duplication test ({N_DUP} resamples), data of comparison 1 ...')
row = uclm[uclm['comparison_id'] == 1].iloc[0]
a, b = data_for(row)
y = a['true_label'].values
COPIES = 5
y_dup = np.repeat(y, COPIES)
# copies of original row i sit at rows COPIES*i ... COPIES*i+COPIES-1 and form one cluster;
# clusters are grouped by class in ascending original row order (as image_level_strata)
dup_strata = [[np.arange(COPIES * i, COPIES * i + COPIES) for i in np.where(y == cls)[0]]
              for cls in range(3) if (y == cls).any()]
for metric, (fn, kind) in METRICS.items():
  va, vb = values(a, kind), values(b, kind)
  va_dup, vb_dup = np.repeat(va, COPIES, axis=0), np.repeat(vb, COPIES, axis=0)
  reference = of.paired_bootstrap_diff(y, va, vb, fn, n_boot=N_DUP)
  clustered = pb.paired_cluster_bootstrap_diff(y_dup, va_dup, vb_dup, fn, dup_strata, n_boot=N_DUP)
  naive = of.paired_bootstrap_diff(y_dup, va_dup, vb_dup, fn, n_boot=N_DUP)
  max_diff = max(abs(reference[k] - clustered[k]) for k in KEYS)
  width_ref = reference['ci_upper'] - reference['ci_lower']
  width_naive = naive['ci_upper'] - naive['ci_lower']
  ratio = width_naive / width_ref if width_ref > 0 else np.nan
  ok_cluster = max_diff <= 1e-12
  ok_naive = ratio < 0.6          # theory 1/sqrt(5) = 0.45; measured 0.43-0.46 at 1000 resamples
  record('T3_duplication_clustered', metric, ok_cluster, f'max_diff={max_diff:.3g}')
  record('T3_duplication_naive_narrower', metric, ok_naive, f'width_ratio={ratio:.3f}')
  print(f'  {metric:14s} clustered-on-copies vs original: max diff {max_diff:.3g} '
        f'[{"OK" if ok_cluster else "FAIL"}] | naive width ratio {ratio:.2f} '
        f'[{"OK" if ok_naive else "FAIL"}]')

# ---------------- T4: structure of the real patient strata ----------------
print('\nT4 structure of the real BUS-UCLM patient strata ...')
patient_ids, stratum_labels = pb.attach_patient_info(a, ids)
record('T4_alignment', 'ids attach to comparison 1 rows', True)
strata = pb.patient_level_strata(patient_ids, stratum_labels)
sizes = [len(s) for s in strata]
record('T4_strata_sizes', '17 has_malignant, 20 no_malignant', sizes == [17, 20], f'sizes={sizes}')
all_rows = np.sort(np.concatenate([c for s in strata for c in s]))
record('T4_coverage', 'every row in exactly one cluster',
       np.array_equal(all_rows, np.arange(len(y))), f'{len(all_rows)} rows')
record('T4_clusters_are_patients', 'each cluster = one patient, all their rows',
       all(len(set(patient_ids[c])) == 1 and len(c) == (patient_ids == patient_ids[c[0]]).sum()
           for s in strata for c in s))

rng = np.random.default_rng(of.SEED)
whole_ok, malignant_ok = True, True
for _ in range(1000):
  idx = pb.cluster_resample_indices(strata, rng)
  counts = np.bincount(idx, minlength=len(y))        # times each row was drawn
  for s, expected in zip(strata, [17, 20]):
    per_cluster = [np.unique(counts[c]) for c in s]
    # whole patient: every row of a cluster drawn the same number of times
    if any(len(u) != 1 for u in per_cluster):
      whole_ok = False
    # that number = how often the patient was drawn; must sum to the stratum size
    if sum(int(u[0]) for u in per_cluster) != expected:
      whole_ok = False
  malignant_ok &= (y[idx] == 1).any()
record('T4_whole_patients', '1000 resamples: whole patients only, 17 + 20 draws', whole_ok)
record('T4_malignant_always_present', '1000 resamples contain malignant images', malignant_ok)

r1, r2 = np.random.default_rng(of.SEED), np.random.default_rng(of.SEED)
same = all(np.array_equal(pb.cluster_resample_indices(strata, r1),
                          pb.cluster_resample_indices(strata, r2)) for _ in range(100))
record('T4_deterministic', 'same seed -> same resamples', same)
for r in report:
  if r['test'].startswith('T4'):
    print(f'  {"OK  " if r["passed"] else "FAIL"} {r["item"]} {r["detail"]}')

# ---------------- verdict ----------------
NEW.mkdir(parents=True, exist_ok=True)
pd.DataFrame(report).to_csv(NEW / 'bootstrap_tests.csv', index=False)
print(f'\nReport saved to {NEW / "bootstrap_tests.csv"}')
if failures:
  print('\nFAILURES:')
  for f in failures:
    print('  ' + f)
  sys.exit('\n[STOP] PATIENT BOOTSTRAP TESTS FAILED. Do not continue; investigate the output above.')
print(f'\nALL {len(report)} PATIENT BOOTSTRAP TESTS PASSED.')
