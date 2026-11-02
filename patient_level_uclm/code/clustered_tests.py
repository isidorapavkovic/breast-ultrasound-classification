"""
clustered_tests.py -- clustered replacements for McNemar and DeLong on BUS-UCLM.

  Durkalski et al. (2003)  replaces McNemar   -> R package clust.bin.pair 0.1.2
  Obuchowski (1997)        replaces DeLong    -> Lerner Institute funcs_clusteredROC.R

The OFFICIAL results come from R (run_r_jobs). The Python functions at the bottom
are independent re-implementations used ONLY to cross-check R in the tests.

Handoff design (Python -> CSV -> R -> CSV -> Python):
  - One CSV per job + a manifest; R runs once for all jobs.
  - Obuchowski needs only the ORDER of scores (its kernel compares a positive score
    with a negative score: >, =, <). Scores are therefore sent as dense integer
    ranks per model: order and ties are kept exactly, and integers cross the CSV
    boundary without any rounding. (Tested: identical to raw-probability DeLong.)
  - Durkalski needs only correct/incorrect per image: sent as 0/1 integers.
  - R writes results with 17 significant digits; Python reads them back exactly.
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

BASE = Path(os.environ.get('UCLM_BASE', Path.home() / 'uclm_rerun'))
R_RUNNER = BASE / 'r' / 'clustered_tests_runner.R'
LERNER_SCRIPT = BASE / 'r' / 'funcs_clusteredROC.R'


# ---------------- building job inputs ----------------
def obuchowski_job(patient_ids, y_true, prob_a, prob_b, cls):
  """One-vs-rest input for class index cls, exactly as the original DeLong call:
  y_binary = (y_true == cls), scores = probability of class cls."""
  y = (np.asarray(y_true) == cls).astype(int)
  return pd.DataFrame({
      'patient_id': np.asarray(patient_ids, dtype=str),
      'y': y,
      'score_a': rankdata(np.asarray(prob_a)[:, cls], method='dense').astype(int),
      'score_b': rankdata(np.asarray(prob_b)[:, cls], method='dense').astype(int),
  })


def durkalski_job(patient_ids, y_true, pred_a, pred_b):
  """Correct/incorrect per image, exactly as the original McNemar call:
  correct = (pred == y_true)."""
  y_true = np.asarray(y_true)
  return pd.DataFrame({
      'patient_id': np.asarray(patient_ids, dtype=str),
      'correct_a': (np.asarray(pred_a) == y_true).astype(int),
      'correct_b': (np.asarray(pred_b) == y_true).astype(int),
  })


# ---------------- running R ----------------
def run_r_jobs(jobs, workdir):
  """jobs: list of dicts with keys job_id (str), test ('obuchowski'|'durkalski'),
  data (DataFrame), expected_n (int), expected_clusters (int).
  Returns a DataFrame of R results, one row per job, in the order given."""
  workdir = Path(workdir)
  workdir.mkdir(parents=True, exist_ok=True)
  ids = [j['job_id'] for j in jobs]
  if len(set(ids)) != len(ids):
    raise ValueError('Duplicate job_id.')
  rows = []
  for j in jobs:
    path = workdir / f'job_{j["job_id"]}.csv'
    j['data'].to_csv(path, index=False)
    rows.append({'job_id': j['job_id'], 'test': j['test'], 'file': str(path),
                 'expected_n': int(j['expected_n']), 'expected_clusters': int(j['expected_clusters'])})
  manifest = workdir / 'manifest.csv'
  results_path = workdir / 'results.csv'
  if results_path.exists():
    results_path.unlink()                       # never read a stale result
  pd.DataFrame(rows).to_csv(manifest, index=False)

  proc = subprocess.run(['Rscript', str(R_RUNNER), str(manifest), str(results_path), str(LERNER_SCRIPT)],
                        capture_output=True, text=True)
  if proc.returncode != 0:
    raise RuntimeError(f'R failed (exit {proc.returncode}):\n{proc.stderr}')

  res = pd.read_csv(results_path, float_precision='round_trip', dtype={'job_id': str})
  if list(res['job_id']) != ids:
    raise RuntimeError('R results do not match the submitted jobs one-to-one, in order.')
  return res


# ================= INDEPENDENT PYTHON IMPLEMENTATIONS (cross-checks only) =================
def durkalski_py(patient_ids, correct_a, correct_b):
  """Durkalski et al. 2003, as implemented in clust.bin.pair 0.1.2 (R/durkalski.R):
  X2 = (sum_k (b_k - c_k)/n_k)^2 / sum_k ((b_k - c_k)/n_k)^2,  p = 1 - pchisq(X2, 1)."""
  df = pd.DataFrame({'p': np.asarray(patient_ids, dtype=str),
                     'b': (np.asarray(correct_a) == 1) & (np.asarray(correct_b) == 0),
                     'c': (np.asarray(correct_a) == 0) & (np.asarray(correct_b) == 1)})
  g = df.groupby('p').agg(n=('b', 'size'), b=('b', 'sum'), c=('c', 'sum'))
  d = (g['b'] - g['c']) / g['n']
  denom = float((d ** 2).sum())
  n_disc_patients = int(((g['b'] + g['c']) > 0).sum())
  if denom == 0.0:
    return {'statistic': np.nan, 'p_value': 1.0, 'method': 'no_net_discordance',
            'n_discordant_patients': n_disc_patients}
  stat = float(d.sum()) ** 2 / denom
  return {'statistic': stat, 'p_value': 1 - stats.chi2.cdf(stat, 1), 'method': 'durkalski',
          'n_discordant_patients': n_disc_patients}


def mcnemar_uncorrected(correct_a, correct_b):
  """Plain McNemar WITHOUT continuity correction: (b-c)^2/(b+c). With one image per
  patient, Durkalski's statistic equals exactly this."""
  a, b_ = np.asarray(correct_a) == 1, np.asarray(correct_b) == 1
  b, c = int(np.sum(a & ~b_)), int(np.sum(~a & b_))
  if b + c == 0:
    return {'statistic': np.nan, 'p_value': 1.0}
  stat = (b - c) ** 2 / (b + c)
  return {'statistic': stat, 'p_value': 1 - stats.chi2.cdf(stat, 1)}


def obuchowski_py(score_a, score_b, y, patient_ids):
  """Line-by-line translation of the two-curve branch of funcs_clusteredROC.R
  (lines 118-171). Returns the same quantities plus the intermediates the
  help document prints with print.all = TRUE."""
  s1, s2 = np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
  y, cid = np.asarray(y), np.asarray(patient_ids, dtype=str)
  clusters = list(dict.fromkeys(cid))                       # unique(), order of appearance
  I = len(clusters)
  I01, I10 = len(set(cid[y == 0])), len(set(cid[y == 1]))
  m = np.array([np.sum((cid == c) & (y == 1)) for c in clusters])
  n = np.array([np.sum((cid == c) & (y == 0)) for c in clusters])
  M, N = m.sum(), n.sum()

  def components(s):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    diff = s[pos][:, None] - s[neg][None, :]
    K = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))   # kern()
    auc = K.mean()
    v10 = pd.Series(K.mean(axis=1), index=pos)                    # V10 per positive
    v01 = pd.Series(K.mean(axis=0), index=neg)                    # V01 per negative
    X = np.array([v10[pos[cid[pos] == c]].sum() for c in clusters])
    Y = np.array([v01[neg[cid[neg] == c]].sum() for c in clusters])
    return auc, X, Y

  a1, X1, Y1 = components(s1)
  a2, X2, Y2 = components(s2)
  f10, f01, f11 = I10 / ((I10 - 1) * M), I01 / ((I01 - 1) * N), I / (I - 1)
  S = {
      'S10_1': f10 * np.sum((X1 - m * a1) ** 2), 'S01_1': f01 * np.sum((Y1 - n * a1) ** 2),
      'S11_1': f11 * np.sum((X1 - m * a1) * (Y1 - n * a1)),
      'S10_2': f10 * np.sum((X2 - m * a2) ** 2), 'S01_2': f01 * np.sum((Y2 - n * a2) ** 2),
      'S11_2': f11 * np.sum((X2 - m * a2) * (Y2 - n * a2)),
      'S10_12': f10 * np.sum((X1 - m * a1) * (X2 - m * a2)),
      'S01_12': f01 * np.sum((Y1 - n * a1) * (Y2 - n * a2)),
      'S11_12': f11 * np.sum((X1 - m * a1) * (Y2 - n * a2)),
      'S11_21': f11 * np.sum((X2 - m * a2) * (Y1 - n * a1)),
  }
  var1 = S['S10_1'] / M + S['S01_1'] / N + 2 * S['S11_1'] / (M * N)
  var2 = S['S10_2'] / M + S['S01_2'] / N + 2 * S['S11_2'] / (M * N)
  cov = S['S10_12'] / M + S['S01_12'] / N + S['S11_12'] / (M * N) + S['S11_21'] / (M * N)
  diff = abs(a1 - a2)
  diff_se = np.sqrt(var1 + var2 - 2 * cov)
  return {'auc_a': a1, 'auc_b': a2, 'se_a': np.sqrt(var1), 'se_b': np.sqrt(var2),
          'abs_diff': diff, 'diff_se': diff_se, 'p_value': 2 * (1 - stats.norm.cdf(diff / diff_se)),
          'I': I, 'I10': I10, 'I01': I01, 'M': int(M), 'N': int(N), **S}
