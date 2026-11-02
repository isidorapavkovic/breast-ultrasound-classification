"""
env_diag.py -- WHERE do the ~1e-16 differences from env_check.py come from?

Diagnostic only: reads the original files, writes one report to data/new/.
Changes nothing and decides nothing; it only shows, column by column, which
numbers differ from the saved Colab results and by how many "last-digit units".

Part A: how many probabilities parse differently with pandas' default reader
        vs its exact (round-trip) reader, for the models in comparisons 1 and 2.
Part B: recompute comparisons 1 and 2 (default reader, as in env_check.py) and
        list every column with its difference.
Part C: recompute comparison 1 with the exact reader, to test whether parsing
        is the cause.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import original_functions as of

BASE = Path(os.environ.get('UCLM_BASE', Path.home() / 'uclm_rerun'))
ORIG = BASE / 'data' / 'original'
NEW = BASE / 'data' / 'new'

PROB_COLS = ['prob_benign', 'prob_malignant', 'prob_normal']
NUMERIC_COLS = [
    'mcnemar_p', 'delong_benign_p', 'delong_malignant_p', 'delong_normal_p',
    'bootstrap_macro_f1_diff', 'bootstrap_macro_f1_ci_lower',
    'bootstrap_macro_f1_ci_upper', 'bootstrap_macro_f1_p',
    'bootstrap_sensitivity90_diff', 'bootstrap_sensitivity90_ci_lower',
    'bootstrap_sensitivity90_ci_upper', 'bootstrap_sensitivity90_p',
    'bootstrap_macro_auc_diff', 'bootstrap_macro_auc_ci_lower',
    'bootstrap_macro_auc_ci_upper', 'bootstrap_macro_auc_p',
]

pred_default = pd.read_csv(ORIG / 'predictions.csv')
pred_exact = pd.read_csv(ORIG / 'predictions.csv', float_precision='round_trip')
raw = pd.read_csv(ORIG / 'statistical_results_raw.csv', float_precision='round_trip')

# ---------------- Part A ----------------
print('PART A: probabilities that parse differently (default vs exact reader)')
key = ['architecture', 'mode', 'arm', 'eval_data']
if not (pred_default[key + ['image_name']].equals(pred_exact[key + ['image_name']])):
    raise ValueError('Row order differs between the two reads -- unexpected, stop.')
differs = (pred_default[PROB_COLS].values != pred_exact[PROB_COLS].values).any(axis=1)
sub = (pred_default['architecture'] == 'resnet50') & (pred_default['arm'] == 'inpainted')
summary = (pd.DataFrame({'mode': pred_default['mode'], 'eval_data': pred_default['eval_data'],
                         'differs': differs})[sub]
           .groupby(['mode', 'eval_data'])['differs'].agg(['sum', 'count']))
print(summary.rename(columns={'sum': 'rows_differing', 'count': 'rows'}).to_string())


# ---------------- helpers ----------------
def compare(cid, predictions, label):
    of.set_predictions(predictions)
    saved = raw[raw['comparison_id'] == cid].iloc[0]
    t0 = time.perf_counter()
    new = of.run_comparison(saved)
    print(f'\n{label}: comparison {cid} ({saved["eval_data_a"]}), {time.perf_counter() - t0:.0f} s')
    rows = []
    for col in NUMERIC_COLS:
        a, b = saved[col], new[col]
        if pd.isna(a) and pd.isna(b):
            diff, ulps = 0.0, 0.0
        else:
            diff = abs(float(a) - float(b))
            ulps = diff / np.spacing(abs(float(a))) if float(a) != 0 else (0.0 if diff == 0 else np.inf)
        rows.append({'run': label, 'comparison_id': cid, 'column': col,
                     'saved': a, 'new': b, 'abs_diff': diff, 'last_digit_units': ulps})
        flag = 'exact' if diff == 0 else f'diff {diff:.2e} ({ulps:.0f} last-digit units)'
        print(f'  {col:36s} {flag}')
    return rows


# ---------------- Parts B and C ----------------
print('\nPART B: column-by-column, default reader (same as env_check.py)')
results = compare(1, pred_default, 'B_default') + compare(2, pred_default, 'B_default')

print('\nPART C: comparison 1 again, exact reader')
results += compare(1, pred_exact, 'C_exact')

NEW.mkdir(parents=True, exist_ok=True)
out = NEW / 'environment_diagnostic.csv'
pd.DataFrame(results).to_csv(out, index=False)
print(f'\nSaved to {out}')
