"""
env_check.py -- Does this laptop reproduce the ORIGINAL Colab statistics?

Recomputes three saved comparisons from scratch, using the verbatim Notebook 04
functions in original_functions.py, and compares every statistical column with
statistical_results_raw.csv. Reads the original files only; writes one small
report to data/new/.

Pass rule: every number differs by at most 1e-12 (we expect exactly 0.0),
and every McNemar method label is identical.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import original_functions as of

# ---------------- paths ----------------
BASE = Path(os.environ.get('UCLM_BASE', Path.home() / 'uclm_rerun'))
ORIG = BASE / 'data' / 'original'
NEW = BASE / 'data' / 'new'
REPORT_PATH = NEW / 'environment_check.csv'

# ---------------- which comparisons ----------------
# Each one exercises a different code path. The expected descriptions are a
# guard: if the IDs ever pointed at different comparisons, we stop.
CHECKS = {
    1:  ('type1_paired',   'bus_uclm',  'bus_uclm'),  # paired: McNemar, DeLong, paired bootstrap
    27: ('type2_unpaired', 'busi_test', 'bus_uclm'),  # unpaired bootstrap, BUS-UCLM side
    2:  ('type1_paired',   'breast',    'breast'),    # paired, second external dataset
}
TOL = 1e-12

NUMERIC_COLS = [
    'mcnemar_p', 'delong_benign_p', 'delong_malignant_p', 'delong_normal_p',
    'bootstrap_macro_f1_diff', 'bootstrap_macro_f1_ci_lower',
    'bootstrap_macro_f1_ci_upper', 'bootstrap_macro_f1_p',
    'bootstrap_sensitivity90_diff', 'bootstrap_sensitivity90_ci_lower',
    'bootstrap_sensitivity90_ci_upper', 'bootstrap_sensitivity90_p',
    'bootstrap_macro_auc_diff', 'bootstrap_macro_auc_ci_lower',
    'bootstrap_macro_auc_ci_upper', 'bootstrap_macro_auc_p',
]


def stop(msg):
    print(f'\n[STOP] {msg}')
    sys.exit(1)


# ---------------- 1. load and sanity-check inputs ----------------
# default reader on purpose: Notebook 04 (cell 56) loaded predictions.csv this way
predictions = pd.read_csv(ORIG / 'predictions.csv')
# round_trip: read saved numbers back bit-for-bit (pandas' default reader can be
# off in the last binary digit, ~1e-16, which would hide whether results are exact)
raw = pd.read_csv(ORIG / 'statistical_results_raw.csv', float_precision='round_trip')

needed = {'architecture', 'mode', 'arm', 'eval_data', 'image_name', 'true_label',
          'pred_label', 'prob_benign', 'prob_malignant', 'prob_normal'}
missing = needed - set(predictions.columns)
if missing:
    stop(f'predictions.csv is missing columns: {sorted(missing)}')
if len(predictions) != 17370:
    stop(f'predictions.csv has {len(predictions)} rows, expected 17370.')
if len(raw) != 171 or raw['comparison_id'].duplicated().any():
    stop('statistical_results_raw.csv is not 171 unique comparisons.')
print(f'Loaded predictions ({len(predictions)} rows) and raw results ({len(raw)} rows).')

of.set_predictions(predictions)

# ---------------- 2. recompute and compare ----------------
report = []
for cid, (pairing, ed_a, ed_b) in CHECKS.items():
    saved = raw[raw['comparison_id'] == cid].iloc[0]
    if (saved['pairing_type'], saved['eval_data_a'], saved['eval_data_b']) != (pairing, ed_a, ed_b):
        stop(f'comparison_id {cid} is not the expected {pairing} {ed_a} vs {ed_b}.')

    print(f'\nRecomputing comparison {cid} ({saved["comparison_type"]}, {ed_a} vs {ed_b}) ...')
    t0 = time.perf_counter()
    new = of.run_comparison(saved)          # saved row carries every registry column
    elapsed = time.perf_counter() - t0

    diffs = {}
    for col in NUMERIC_COLS:
        a, b = saved[col], new[col]
        if pd.isna(a) and pd.isna(b):
            diffs[col] = 0.0                # both "not applicable": agree
        elif pd.isna(a) or pd.isna(b):
            diffs[col] = np.inf             # one missing, one present: disagree
        else:
            diffs[col] = abs(float(a) - float(b))
    method_ok = saved['mcnemar_method'] == new['mcnemar_method']
    max_diff = max(diffs.values())
    n_exact = sum(d == 0.0 for d in diffs.values())
    passed = method_ok and max_diff <= TOL

    print(f'  {elapsed:.1f} s | max abs diff = {max_diff:.3g} | '
          f'exactly equal: {n_exact}/{len(NUMERIC_COLS)} | McNemar method same: {method_ok} | '
          f'{"PASS" if passed else "FAIL"}')
    if not passed:
        for col, d in diffs.items():
            if d > TOL:
                print(f'    {col}: saved={saved[col]!r} new={new[col]!r}')

    report.append({'comparison_id': cid, 'seconds': round(elapsed, 1),
                   'max_abs_diff': max_diff, 'n_exactly_equal': n_exact,
                   'n_columns': len(NUMERIC_COLS), 'mcnemar_method_same': method_ok,
                   'passed': passed})

# ---------------- 3. save the report and give the verdict ----------------
NEW.mkdir(parents=True, exist_ok=True)
report = pd.DataFrame(report)
report.to_csv(REPORT_PATH, index=False)
print(f'\nReport saved to {REPORT_PATH}')
print(f'Mean time per comparison: {report["seconds"].mean():.1f} s')

if report['passed'].all():
    print('\nENVIRONMENT CHECK PASSED: this laptop reproduces the original statistics.')
else:
    stop('ENVIRONMENT CHECK FAILED. Do not continue; investigate the output above.')
