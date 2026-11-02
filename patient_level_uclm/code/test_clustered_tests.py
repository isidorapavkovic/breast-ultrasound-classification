"""
test_clustered_tests.py -- prove the clustered McNemar (Durkalski) and clustered
DeLong (Obuchowski) pipeline is correct BEFORE any patient-level result is used.

  C1 Published example   MRA data (Obuchowski 1997, Table 3) through the full
                         Python -> CSV -> R -> Python handoff must reproduce the
                         Lerner help document; R must equal the Python translation.
  C2 One image/patient   All 45 paired BUS-UCLM comparisons with every image its own
                         cluster: Obuchowski must equal the ORIGINAL DeLong p-values
                         and AUCs; Durkalski must equal uncorrected McNemar.
  C3 Duplication         Comparison 1, every image copied 5x, copies = one cluster:
                         both tests must give exactly the C2 result.
  C4 Two implementations All 45 comparisons with REAL patient clusters: R must equal
                         the independent Python versions. Only differences are
                         printed or saved, never the p-values themselves.
  C5 Guards              R must refuse: missing values, wrong row count, wrong
                         patient count.

Writes data/new/clustered_tests_checks.csv. Prints a verdict.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import clustered_tests as ct
import original_functions as of
import patient_bootstrap as pb

BASE = Path(os.environ.get('UCLM_BASE', Path.home() / 'uclm_rerun'))
ORIG, NEW = BASE / 'data' / 'original', BASE / 'data' / 'new'
WORK = NEW / 'r_jobs_tests'
TOL = 1e-12
PROB = ['prob_benign', 'prob_malignant', 'prob_normal']
OBU_FIELDS = ['auc_a', 'auc_b', 'se_a', 'se_b', 'abs_diff', 'diff_se', 'p_value']

predictions = pd.read_csv(ORIG / 'predictions.csv', float_precision='round_trip')
raw = pd.read_csv(ORIG / 'statistical_results_raw.csv', float_precision='round_trip')
ids = pd.read_csv(NEW / 'uclm_patient_ids.csv')
of.set_predictions(predictions)
paired = raw[(raw['pairing_type'] == 'type1_paired') & (raw['eval_data_a'] == 'bus_uclm')]
if len(paired) != 45:
  sys.exit(f'[STOP] Expected 45 paired BUS-UCLM comparisons, found {len(paired)}.')

report = []


def record(test, item, passed, max_diff=np.nan, detail=''):
  report.append({'test': test, 'item': item, 'passed': bool(passed),
                 'max_abs_diff': max_diff, 'detail': detail})


def close(a, b, rel=False):
  if pd.isna(a) and pd.isna(b):
    return 0.0
  if pd.isna(a) or pd.isna(b):
    return np.inf
  d = abs(float(a) - float(b))
  return d / max(1.0, abs(float(a))) if rel else d


def data_for(row):
  a = of.get_predictions_for_condition(row['arch_a'], row['mode_a'], row['arm_a'], 'bus_uclm')
  b = of.get_predictions_for_condition(row['arch_b'], row['mode_b'], row['arm_b'], 'bus_uclm')
  if not (a['image_name'].values == b['image_name'].values).all():
    raise ValueError('Paired conditions are not aligned.')
  return a, b


# ---------------- C1: published example ----------------
print('C1 published example (MRA, Obuchowski 1997) through the R handoff ...')
mra = pd.read_csv(BASE / 'r' / 'MRA.csv').dropna(subset=['reader1_result', 'reader2_result'])
mra_job = pd.DataFrame({'patient_id': mra['patient_id'].astype(str), 'y': mra['disease'].astype(int),
                        'score_a': mra['reader1_result'].astype(int), 'score_b': mra['reader2_result'].astype(int)})
r = ct.run_r_jobs([{'job_id': 'mra', 'test': 'obuchowski', 'data': mra_job,
                    'expected_n': 65, 'expected_clusters': 36}], WORK / 'c1').iloc[0]
help_doc = {'auc_a': 0.9837, 'auc_b': 0.9852, 'se_a': 0.0108, 'se_b': 0.0097,
            'abs_diff': 0.0014, 'diff_se': 0.0066, 'p_value': 0.8271}
ok = all(round(float(r[k]), 4) == v for k, v in help_doc.items()) and r['n_obs'] == 65 and r['n_clusters'] == 36
record('C1_help_document', 'R reproduces all 7 published values (4 dp), 65 obs, 36 patients', ok)
py = ct.obuchowski_py(mra_job['score_a'], mra_job['score_b'], mra_job['y'], mra_job['patient_id'])
md = max(close(r[k], py[k]) for k in OBU_FIELDS)
record('C1_R_vs_python', 'R equals the Python translation on MRA', md <= TOL, md)
print(f'  help document: {"OK" if ok else "FAIL"} | R vs Python max diff {md:.2g}')

# ---------------- C2: one image per patient ----------------
print('\nC2 one image per patient, 45 comparisons (one R run) ...')
jobs, expected = [], {}
for _, row in paired.iterrows():
  cid = row['comparison_id']
  a, b = data_for(row)
  y = a['true_label'].values
  singles = a['image_name'].values                       # every image its own "patient"
  for i, cls in enumerate(of.CLASS_NAMES):
    jid = f'c{cid}_obu_{cls}'
    jobs.append({'job_id': jid, 'test': 'obuchowski', 'expected_n': len(y), 'expected_clusters': len(y),
                 'data': ct.obuchowski_job(singles, y, a[PROB].values, b[PROB].values, i)})
    expected[jid] = of.delong_test((y == i).astype(int), a[PROB].values[:, i], b[PROB].values[:, i])
  jid = f'c{cid}_durk'
  jobs.append({'job_id': jid, 'test': 'durkalski', 'expected_n': len(y), 'expected_clusters': len(y),
               'data': ct.durkalski_job(singles, y, a['pred_label'].values, b['pred_label'].values)})
  expected[jid] = ct.mcnemar_uncorrected((a['pred_label'].values == y).astype(int),
                                         (b['pred_label'].values == y).astype(int))
res_c2 = ct.run_r_jobs(jobs, WORK / 'c2').set_index('job_id')
worst_obu, worst_durk, n_obu_ok, n_durk_ok = 0.0, 0.0, 0, 0
for jid, exp in expected.items():
  got = res_c2.loc[jid]
  if '_obu_' in jid:
    md = max(close(got['p_value'], exp['p_value']), close(got['auc_a'], exp['auc_a']),
             close(got['auc_b'], exp['auc_b']))
    worst_obu = max(worst_obu, md)
    n_obu_ok += md <= TOL
    record('C2_obuchowski_equals_original_delong', jid, md <= TOL, md)
  else:
    md = max(close(got['p_value'], exp['p_value']), close(got['statistic'], exp['statistic'], rel=True))
    worst_durk = max(worst_durk, md)
    n_durk_ok += md <= TOL
    record('C2_durkalski_equals_uncorrected_mcnemar', jid, md <= TOL, md)
print(f'  Obuchowski = original DeLong:      {n_obu_ok}/135  (max diff {worst_obu:.2g})')
print(f'  Durkalski  = uncorrected McNemar:  {n_durk_ok}/45   (max diff {worst_durk:.2g})')

# ---------------- C3: duplication ----------------
print('\nC3 duplication, comparison 1 (each image x5, copies = one cluster) ...')
row = paired[paired['comparison_id'] == 1].iloc[0]
a, b = data_for(row)
y = a['true_label'].values
COPIES = 5
dup_ids = np.repeat(a['image_name'].values, COPIES)
y_dup = np.repeat(y, COPIES)
pa, pb_ = np.repeat(a[PROB].values, COPIES, axis=0), np.repeat(b[PROB].values, COPIES, axis=0)
jobs = [{'job_id': f'dup_obu_{cls}', 'test': 'obuchowski', 'expected_n': len(y_dup),
         'expected_clusters': len(y), 'data': ct.obuchowski_job(dup_ids, y_dup, pa, pb_, i)}
        for i, cls in enumerate(of.CLASS_NAMES)]
jobs.append({'job_id': 'dup_durk', 'test': 'durkalski', 'expected_n': len(y_dup), 'expected_clusters': len(y),
             'data': ct.durkalski_job(dup_ids, y_dup, np.repeat(a['pred_label'].values, COPIES),
                                      np.repeat(b['pred_label'].values, COPIES))})
res_c3 = ct.run_r_jobs(jobs, WORK / 'c3').set_index('job_id')
for cls in of.CLASS_NAMES:
  md = max(close(res_c3.loc[f'dup_obu_{cls}', k], res_c2.loc[f'c1_obu_{cls}', k]) for k in OBU_FIELDS)
  record('C3_duplication_obuchowski', cls, md <= TOL, md)
  print(f'  Obuchowski {cls:9s}: max diff vs uncopied {md:.2g} [{"OK" if md <= TOL else "FAIL"}]')
md = max(close(res_c3.loc['dup_durk', 'p_value'], res_c2.loc['c1_durk', 'p_value']),
         close(res_c3.loc['dup_durk', 'statistic'], res_c2.loc['c1_durk', 'statistic'], rel=True))
record('C3_duplication_durkalski', 'statistic and p', md <= TOL, md)
print(f'  Durkalski          : max diff vs uncopied {md:.2g} [{"OK" if md <= TOL else "FAIL"}]')

# ---------------- C4: R vs Python on real patient clusters ----------------
print('\nC4 real patient clusters, 45 comparisons: R vs independent Python ...')
jobs, py_results = [], {}
for _, row in paired.iterrows():
  cid = row['comparison_id']
  a, b = data_for(row)
  y = a['true_label'].values
  patient_ids, _ = pb.attach_patient_info(a, ids)        # alignment re-checked every time
  for i, cls in enumerate(of.CLASS_NAMES):
    jid = f'c{cid}_obu_{cls}'
    data = ct.obuchowski_job(patient_ids, y, a[PROB].values, b[PROB].values, i)
    jobs.append({'job_id': jid, 'test': 'obuchowski', 'expected_n': 629, 'expected_clusters': 37, 'data': data})
    py_results[jid] = ct.obuchowski_py(data['score_a'], data['score_b'], data['y'], data['patient_id'])
  jid = f'c{cid}_durk'
  data = ct.durkalski_job(patient_ids, y, a['pred_label'].values, b['pred_label'].values)
  jobs.append({'job_id': jid, 'test': 'durkalski', 'expected_n': 629, 'expected_clusters': 37, 'data': data})
  py_results[jid] = ct.durkalski_py(data['patient_id'], data['correct_a'], data['correct_b'])
res_c4 = ct.run_r_jobs(jobs, WORK / 'c4').set_index('job_id')
counts_ok = bool((res_c4['n_obs'] == 629).all() and (res_c4['n_clusters'] == 37).all())
record('C4_counts', 'every R job used 629 images and 37 patients', counts_ok)
worst_obu, worst_durk, n_obu_ok, n_durk_ok = 0.0, 0.0, 0, 0
for jid, py in py_results.items():
  got = res_c4.loc[jid]
  if '_obu_' in jid:
    md = max(close(got[k], py[k]) for k in OBU_FIELDS)
    worst_obu = max(worst_obu, md)
    n_obu_ok += md <= TOL
    record('C4_obuchowski_R_vs_python', jid, md <= TOL, md)
  else:
    same_label = got['method'] == py['method'] and got['n_discordant_patients'] == py['n_discordant_patients']
    md = max(close(got['p_value'], py['p_value']), close(got['statistic'], py['statistic'], rel=True))
    worst_durk = max(worst_durk, md)
    n_durk_ok += (md <= TOL) and same_label
    record('C4_durkalski_R_vs_python', jid, (md <= TOL) and same_label, md)
print(f'  every job 629 images / 37 patients: {"OK" if counts_ok else "FAIL"}')
print(f'  Obuchowski R = Python: {n_obu_ok}/135  (max diff {worst_obu:.2g})')
print(f'  Durkalski  R = Python: {n_durk_ok}/45   (max diff {worst_durk:.2g})')

# ---------------- C5: guards ----------------
print('\nC5 R must refuse bad input ...')
good = mra_job.copy()
bad_cases = {
    'missing value': (good.assign(score_a=good['score_a'].astype(float).where(good.index != good.index[0])), 65, 36),
    'wrong row count': (good, 64, 36),
    'wrong patient count': (good, 65, 35),
}
for name, (data, n_exp, k_exp) in bad_cases.items():
  try:
    ct.run_r_jobs([{'job_id': 'guard', 'test': 'obuchowski', 'data': data,
                    'expected_n': n_exp, 'expected_clusters': k_exp}], WORK / 'c5')
    refused = False
  except RuntimeError:
    refused = True
  record('C5_guard', name, refused)
  print(f'  {name:20s}: {"refused (OK)" if refused else "ACCEPTED (FAIL)"}')

# ---------------- verdict ----------------
NEW.mkdir(parents=True, exist_ok=True)
rep = pd.DataFrame(report)
rep.to_csv(NEW / 'clustered_tests_checks.csv', index=False)
print(f'\nReport saved to {NEW / "clustered_tests_checks.csv"}')
failed = rep[~rep['passed']]
if len(failed):
  print('\nFAILURES:')
  print(failed.to_string(index=False))
  sys.exit('\n[STOP] CLUSTERED TESTS FAILED. Do not continue; investigate the output above.')
print(f'\nALL {len(rep)} CLUSTERED TEST CHECKS PASSED.')
