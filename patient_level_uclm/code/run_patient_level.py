"""
run_patient_level.py -- the patient-level reanalysis of the 63 BUS-UCLM comparisons.

Mirrors the original Notebook 04 run_comparison() exactly, with three replacements:
  McNemar           -> Durkalski et al. 2003  (R: clust.bin.pair 0.1.2)
  DeLong (per class)-> Obuchowski 1997        (R: funcs_clusteredROC.R)
  image bootstrap   -> patient bootstrap, strata has_malignant (17) / no_malignant (20)
                       (unpaired comparisons: BUSI test side keeps the original
                       class-stratified image bootstrap; BUS-UCLM side is patient-level)
Every component was verified beforehand (test_patient_bootstrap.py, test_clustered_tests.py).

Phases:
  1. R tests for the 45 paired comparisons, in one R run (a few minutes).
  2. Bootstraps, one comparison at a time. Each finished comparison is saved
     immediately to its own file -> safe to interrupt (Ctrl+C) and resume by
     running the same command again; finished comparisons are skipped.
  3. Assemble the 63 rows into the final file (write-once).

Hard stops (nothing final is written if any fails):
  - every R job used 629 images / 37 patients
  - R's AUCs equal the ordinary pooled AUCs (the clustered method does not change AUCs)
  - every bootstrap point estimate equals the ORIGINAL saved point estimate
    (point estimates use all images, so they must not change)

Output: data/new/patient_level/statistical_results_patient_level_uclm.csv
        (same 29 columns as statistical_results_raw.csv, plus pl_* columns)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import clustered_tests as ct
import original_functions as of
import patient_bootstrap as pb

BASE = Path(os.environ.get('UCLM_BASE', Path.home() / 'uclm_rerun'))
ORIG, NEW = BASE / 'data' / 'original', BASE / 'data' / 'new'
OUT = NEW / 'patient_level'
ROWS = OUT / 'rows'                          # one file per finished comparison
R_RESULTS = OUT / 'r_results.csv'
N_BOOT = int(os.environ.get('UCLM_N_BOOT', 10000))
TEST_MODE = N_BOOT != 10000                  # only ever used to test this script
FINAL = OUT / ('statistical_results_patient_level_uclm' + ('_TESTMODE' if TEST_MODE else '') + '.csv')
TOL = 1e-12
PROB = ['prob_benign', 'prob_malignant', 'prob_normal']
METRICS = [('macro_f1', of.metric_macro_f1, 'pred'),
           ('sensitivity90', of.metric_malignant_sensitivity_90, 'prob'),
           ('macro_auc', of.metric_macro_auc, 'prob')]
REGISTRY_COLS = ['comparison_id', 'comparison_type', 'family', 'pairing_type',
                 'arch_a', 'mode_a', 'arm_a', 'eval_data_a', 'arch_b', 'mode_b', 'arm_b', 'eval_data_b']


def stop(msg):
  sys.exit(f'\n[STOP] {msg}\nNo final file was written. Investigate before continuing.')


if TEST_MODE:
  print(f'*** TEST MODE: {N_BOOT} resamples instead of 10000. Output is NOT a real result. ***\n')

predictions = pd.read_csv(ORIG / 'predictions.csv', float_precision='round_trip')
raw = pd.read_csv(ORIG / 'statistical_results_raw.csv', float_precision='round_trip')
ids = pd.read_csv(NEW / 'uclm_patient_ids.csv')
of.set_predictions(predictions)

uclm = raw[(raw['eval_data_a'] == 'bus_uclm') | (raw['eval_data_b'] == 'bus_uclm')].sort_values('comparison_id')
paired = uclm[uclm['pairing_type'] == 'type1_paired']
unpaired = uclm[uclm['pairing_type'] == 'type2_unpaired']
if (len(uclm), len(paired), len(unpaired)) != (63, 45, 18):
  stop(f'Expected 63 BUS-UCLM comparisons (45 paired, 18 unpaired); found {len(uclm)} ({len(paired)}, {len(unpaired)}).')
if not ((unpaired['eval_data_a'] == 'busi_test') & (unpaired['eval_data_b'] == 'bus_uclm')).all():
  stop('Unpaired comparisons are not all BUSI test (A) vs BUS-UCLM (B).')


def load(row, side):
  return of.get_predictions_for_condition(row[f'arch_{side}'], row[f'mode_{side}'],
                                          row[f'arm_{side}'], row[f'eval_data_{side}'])


# ================= PHASE 1: R tests for the paired comparisons =================
OUT.mkdir(parents=True, exist_ok=True)
expected_jobs = [f'c{cid}_{t}' for cid in paired['comparison_id']
                 for t in ['durk'] + [f'obu_{c}' for c in of.CLASS_NAMES]]

if R_RESULTS.exists():
  r_res = pd.read_csv(R_RESULTS, float_precision='round_trip', dtype={'job_id': str})
  if sorted(r_res['job_id']) != sorted(expected_jobs):
    stop(f'{R_RESULTS.name} exists but is incomplete or different. Delete it and run again.')
  print(f'Phase 1: reusing {R_RESULTS.name} ({len(r_res)} R jobs).')
else:
  print('Phase 1: clustered McNemar (Durkalski) and clustered DeLong (Obuchowski) in R ...')
  t0 = time.perf_counter()
  jobs = []
  for _, row in paired.iterrows():
    a, b = load(row, 'a'), load(row, 'b')
    if a['image_name'].tolist() != b['image_name'].tolist():
      stop(f'comparison {row["comparison_id"]}: paired conditions not aligned.')
    y = a['true_label'].values
    pid, _ = pb.attach_patient_info(a, ids)
    cid = row['comparison_id']
    jobs.append({'job_id': f'c{cid}_durk', 'test': 'durkalski', 'expected_n': 629, 'expected_clusters': 37,
                 'data': ct.durkalski_job(pid, y, a['pred_label'].values, b['pred_label'].values)})
    for i, cls in enumerate(of.CLASS_NAMES):
      jobs.append({'job_id': f'c{cid}_obu_{cls}', 'test': 'obuchowski', 'expected_n': 629,
                   'expected_clusters': 37,
                   'data': ct.obuchowski_job(pid, y, a[PROB].values, b[PROB].values, i)})
  r_res = ct.run_r_jobs(jobs, OUT / 'r_jobs')
  if not ((r_res['n_obs'] == 629).all() and (r_res['n_clusters'] == 37).all()):
    stop('An R job did not use exactly 629 images and 37 patients.')
  if r_res['p_value'].isna().any():
    stop('An R job returned no p-value.')
  r_res.to_csv(R_RESULTS, index=False)
  print(f'  {len(r_res)} R jobs done in {time.perf_counter() - t0:.0f} s.')

r_res = r_res.set_index('job_id')

# Hard check: clustered AUC == ordinary pooled AUC, for every paired comparison and class
worst = 0.0
for _, row in paired.iterrows():
  a, b = load(row, 'a'), load(row, 'b')
  y = a['true_label'].values
  for i, cls in enumerate(of.CLASS_NAMES):
    d = of.delong_test((y == i).astype(int), a[PROB].values[:, i], b[PROB].values[:, i])
    got = r_res.loc[f'c{row["comparison_id"]}_obu_{cls}']
    worst = max(worst, abs(got['auc_a'] - d['auc_a']), abs(got['auc_b'] - d['auc_b']))
if worst > TOL:
  stop(f'R AUCs differ from the pooled AUCs by {worst:.3g}.')
print(f'  check: R AUCs equal the pooled AUCs (max diff {worst:.2g})')

# ================= PHASE 2: bootstraps, one comparison at a time =================
ROWS.mkdir(parents=True, exist_ok=True)
todo = [r for _, r in uclm.iterrows() if not (ROWS / f'comparison_{r["comparison_id"]}.csv').exists()]
print(f'\nPhase 2: patient-level bootstraps ({N_BOOT} resamples). '
      f'{63 - len(todo)} already done, {len(todo)} to go.')

durations = []
for n_done, row in enumerate(todo, start=1):
  t0 = time.perf_counter()
  cid = row['comparison_id']
  a, b = load(row, 'a'), load(row, 'b')
  result = {c: row[c] for c in REGISTRY_COLS}

  if row['pairing_type'] == 'type1_paired':
    if a['image_name'].tolist() != b['image_name'].tolist():
      stop(f'comparison {cid}: paired conditions not aligned.')
    y = a['true_label'].values
    pid, strat = pb.attach_patient_info(a, ids)
    strata = pb.patient_level_strata(pid, strat)
    durk = r_res.loc[f'c{cid}_durk']
    result['mcnemar_p'] = durk['p_value']
    result['mcnemar_method'] = durk['method']
    for cls in of.CLASS_NAMES:
      result[f'delong_{cls}_p'] = r_res.loc[f'c{cid}_obu_{cls}', 'p_value']
    boots = {}
    for name, fn, kind in METRICS:
      va = a['pred_label'].values if kind == 'pred' else a[PROB].values
      vb = b['pred_label'].values if kind == 'pred' else b[PROB].values
      boots[name] = pb.paired_cluster_bootstrap_diff(y, va, vb, fn, strata, n_boot=N_BOOT)
    extras = {'pl_n_discordant_patients': durk['n_discordant_patients'],
              'pl_durkalski_statistic': durk['statistic']}
  else:
    ya, yb = a['true_label'].values, b['true_label'].values
    pid_b, strat_b = pb.attach_patient_info(b, ids)          # side B is BUS-UCLM
    strata_a = pb.image_level_strata(ya)                      # BUSI test: original resampling
    strata_b = pb.patient_level_strata(pid_b, strat_b)
    result['mcnemar_p'] = np.nan
    result['mcnemar_method'] = 'not_applicable'
    for cls in of.CLASS_NAMES:
      result[f'delong_{cls}_p'] = np.nan
    boots = {}
    for name, fn, kind in METRICS:
      va = a['pred_label'].values if kind == 'pred' else a[PROB].values
      vb = b['pred_label'].values if kind == 'pred' else b[PROB].values
      boots[name] = pb.unpaired_cluster_bootstrap_diff(ya, va, strata_a, yb, vb, strata_b, fn, n_boot=N_BOOT)
    extras = {'pl_n_discordant_patients': np.nan, 'pl_durkalski_statistic': np.nan}

  for name, _, _ in METRICS:
    bt = boots[name]
    saved_point = row[f'bootstrap_{name}_diff']
    if abs(bt['point_estimate'] - saved_point) > TOL:
      stop(f'comparison {cid} {name}: point estimate {bt["point_estimate"]!r} differs from the '
           f'original {saved_point!r}.')
    result[f'bootstrap_{name}_diff'] = bt['point_estimate']
    result[f'bootstrap_{name}_ci_lower'] = bt['ci_lower']
    result[f'bootstrap_{name}_ci_upper'] = bt['ci_upper']
    result[f'bootstrap_{name}_p'] = bt['p_value']
  result['analysis_unit'] = 'patient'
  result.update(extras)

  # atomic save: write to a temporary name, then rename (an interruption can never leave half a file)
  tmp = ROWS / f'comparison_{cid}.csv.tmp'
  pd.DataFrame([result]).to_csv(tmp, index=False)
  os.replace(tmp, ROWS / f'comparison_{cid}.csv')

  durations.append(time.perf_counter() - t0)
  eta_min = np.mean(durations) * (len(todo) - n_done) / 60
  print(f'  [{n_done:2d}/{len(todo)}] comparison {cid:3d} ({row["pairing_type"]}) '
        f'{durations[-1]:.0f} s | about {eta_min:.0f} min left', flush=True)

# ================= PHASE 3: assemble, check, write once =================
print('\nPhase 3: assembling the final file ...')
parts = [pd.read_csv(ROWS / f'comparison_{cid}.csv', float_precision='round_trip')
         for cid in uclm['comparison_id']]
final = pd.concat(parts, ignore_index=True)
expected_cols = list(raw.columns) + ['analysis_unit', 'pl_n_discordant_patients', 'pl_durkalski_statistic']
if list(final.columns) != expected_cols:
  stop(f'Column mismatch.\n  got:      {list(final.columns)}\n  expected: {expected_cols}')
if list(final['comparison_id']) != list(uclm['comparison_id']):
  stop('Comparison IDs do not match the 63 BUS-UCLM comparisons.')
for col in REGISTRY_COLS:
  if not (final[col].values == uclm[col].values).all():
    stop(f'Registry column {col} does not match the original.')
p_cols = [c for c in final.columns if c.endswith('_p')]
vals = final[p_cols].values.astype(float)
if np.any((vals[~np.isnan(vals)] < 0) | (vals[~np.isnan(vals)] > 1)):
  stop('A p-value is outside [0, 1].')
if not set(final['mcnemar_method']) <= {'durkalski', 'no_net_discordance', 'not_applicable'}:
  stop(f'Unexpected McNemar method labels: {set(final["mcnemar_method"])}')

if FINAL.exists():
  existing = pd.read_csv(FINAL, float_precision='round_trip')
  if not existing.equals(final):
    stop(f'{FINAL.name} already exists and DIFFERS from this run.')
  print(f'  {FINAL.name} already exists and is identical; left untouched.')
else:
  final.to_csv(FINAL, index=False)
  print(f'  wrote {FINAL}')
print(f'\nPATIENT-LEVEL RERUN COMPLETE: 63 comparisons '
      f'({(final["mcnemar_method"] == "no_net_discordance").sum()} with no net discordance).')
