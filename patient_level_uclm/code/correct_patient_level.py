"""
correct_patient_level.py -- redo the multiple-comparison correction with the
patient-level BUS-UCLM results, using the ORIGINAL Notebook 04 procedure
(original_correction.py: Holm for confirmatory, Benjamini-Hochberg for
exploratory, 14 groups pooled across datasets).

  Step A  Reproduce the original correction EXACTLY from the original raw file
          (read the way Notebook 04 cell 96 read it: default pandas reader).
          Must equal statistical_results_corrected.csv in every cell, or STOP.
  Step B  Build the updated raw table: the 63 BUS-UCLM rows replaced by the
          patient-level rows; the 108 other rows untouched (all files read exactly).
  Step C  Apply the same correction to the updated table; run the notebook's own
          checks (group sizes 27/39/108/132, adjusted p valid).
  Step D  Report every change in significance, separately for BUS-UCLM rows and
          for the other rows (whose ADJUSTED values can move via the pooled groups).

Outputs (write-once) in data/new/patient_level/:
  statistical_results_raw_updated.csv
  statistical_results_corrected_updated.csv
  significance_changes.csv   (one row per comparison x test)
"""
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import original_correction as oc

BASE = Path(os.environ.get('UCLM_BASE', Path.home() / 'uclm_rerun'))
ORIG, OUT = BASE / 'data' / 'original', BASE / 'data' / 'new' / 'patient_level'
PL_FILE = OUT / 'statistical_results_patient_level_uclm.csv'
RAW_UPD = OUT / 'statistical_results_raw_updated.csv'
CORR_UPD = OUT / 'statistical_results_corrected_updated.csv'
CHANGES = OUT / 'significance_changes.csv'
TOL = 1e-12
REGISTRY = ['comparison_id', 'comparison_type', 'family', 'pairing_type', 'arch_a', 'mode_a',
            'arm_a', 'eval_data_a', 'arch_b', 'mode_b', 'arm_b', 'eval_data_b']
POINT = ['bootstrap_macro_f1_diff', 'bootstrap_sensitivity90_diff', 'bootstrap_macro_auc_diff']
EXTRAS = ['analysis_unit', 'pl_n_discordant_patients', 'pl_durkalski_statistic']


def stop(msg):
  sys.exit(f'\n[STOP] {msg}\nNothing was written. Investigate before continuing.')


def quiet(fn, *args):
  """Run a notebook function, capturing its printed PASS/FAIL lines."""
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    out = fn(*args)
  text = buf.getvalue()
  if '[FAIL]' in text:
    stop(f'Notebook verification failed:\n{text}')
  return out, text.count('[PASS]')


# ---------------- Step A: reproduce the original correction exactly ----------------
print('Step A: reproduce the original correction ...')
_, n_pass = quiet(oc.verify_correction_functions)
raw_as_notebook = pd.read_csv(ORIG / 'statistical_results_raw.csv')        # as cell 96 did
saved_corrected = pd.read_csv(ORIG / 'statistical_results_corrected.csv', float_precision='round_trip')
(repro, sizes), _ = quiet(oc.apply_correction, raw_as_notebook)
_, n_pass2 = quiet(oc.verify_correction, repro, sizes)
if not repro.equals(saved_corrected):
  stop('The original correction was NOT reproduced exactly.')
print(f'  notebook checks: {n_pass + n_pass2} PASS | reproduced statistical_results_corrected.csv '
      f'exactly (all {saved_corrected.shape[0]} rows x {saved_corrected.shape[1]} columns)')

# ---------------- Step B: updated raw table ----------------
print('\nStep B: merge the 63 patient-level rows ...')
raw = pd.read_csv(ORIG / 'statistical_results_raw.csv', float_precision='round_trip')
pl = pd.read_csv(PL_FILE, float_precision='round_trip')
if len(pl) != 63 or pl['comparison_id'].duplicated().any() or not (pl['analysis_unit'] == 'patient').all():
  stop('Patient-level file is not 63 unique patient-level comparisons.')
is_uclm = (raw['eval_data_a'] == 'bus_uclm') | (raw['eval_data_b'] == 'bus_uclm')
if sorted(raw.loc[is_uclm, 'comparison_id']) != sorted(pl['comparison_id']):
  stop('Patient-level comparison IDs do not match the BUS-UCLM comparisons in the raw file.')

updated = raw.copy()
updated['analysis_unit'] = 'image'
updated['pl_n_discordant_patients'] = np.nan
updated['pl_durkalski_statistic'] = np.nan
pl_by_id = pl.set_index('comparison_id')
for idx in updated.index[is_uclm]:
  cid = updated.at[idx, 'comparison_id']
  for col in updated.columns:
    if col != 'comparison_id':
      updated.at[idx, col] = pl_by_id.at[cid, col]

stat_cols = [c for c in raw.columns if c not in REGISTRY]
checks = {
    'columns = original 29 + 3 extras': list(updated.columns) == list(raw.columns) + EXTRAS,
    'registry columns unchanged (all 171 rows)': all((updated[c].values == raw[c].values).all() for c in REGISTRY),
    '108 other rows unchanged in every statistic': updated.loc[~is_uclm, stat_cols].equals(raw.loc[~is_uclm, stat_cols]),
    'point estimates unchanged (all 171 rows)': all(np.nanmax(np.abs(updated[c].values - raw[c].values)) <= TOL for c in POINT),
    'BUS-UCLM paired rows use Durkalski': set(updated.loc[is_uclm & (updated['pairing_type'] == 'type1_paired'), 'mcnemar_method']) <= {'durkalski', 'no_net_discordance'},
    'analysis_unit: 63 patient, 108 image': (updated['analysis_unit'] == 'patient').sum() == 63,
}
for name, ok in checks.items():
  print(f'  [{"PASS" if ok else "FAIL"}] {name}')
if not all(checks.values()):
  stop('Merge checks failed.')

# ---------------- Step C: correction on the updated table ----------------
print('\nStep C: apply the original correction to the updated table ...')
(corrected, sizes_new), _ = quiet(oc.apply_correction, updated)
_, n_pass3 = quiet(oc.verify_correction, corrected, sizes_new)
print(f'  notebook checks: {n_pass3} PASS (14 groups correctly sized; adjusted p valid)')
borderline = []
for col in oc.P_COLUMNS:
  near = corrected[(corrected[f'{col}_adj'] - 0.05).abs() < 1e-9]
  borderline += [(int(r['comparison_id']), col) for _, r in near.iterrows()]
print(f'  adjusted p-values within 1e-9 of 0.05: {len(borderline)}' + (f' -> {borderline}' if borderline else ''))

# ---------------- Step D: what changed ----------------
rows = []
for _, new in corrected.iterrows():
  old = saved_corrected.loc[saved_corrected['comparison_id'] == new['comparison_id']].iloc[0]
  # raw p: from the original raw file read EXACTLY (the corrected file's copies of the raw
  # p-values carry the default reader's last-digit changes; adjusted values come from it)
  old_raw = raw.loc[raw['comparison_id'] == new['comparison_id']].iloc[0]
  uclm = 'bus_uclm' in (new['eval_data_a'], new['eval_data_b'])
  data = new['eval_data_a'] if new['eval_data_a'] == new['eval_data_b'] else f'{new["eval_data_a"]} vs {new["eval_data_b"]}'
  for col in oc.P_COLUMNS:
    if pd.isna(new[col]):
      continue
    rows.append({'comparison_id': new['comparison_id'], 'comparison_type': new['comparison_type'],
                 'family': new['family'], 'data': data, 'bus_uclm_row': uclm, 'test': col.replace('_p', ''),
                 'raw_p_old': old_raw[col], 'raw_p_new': new[col],
                 'adj_p_old': old[f'{col}_adj'], 'adj_p_new': new[f'{col}_adj'],
                 'significant_old': bool(old[f'{col}_significant']),
                 'significant_new': bool(new[f'{col}_significant'])})
ch = pd.DataFrame(rows)
ch['change'] = np.select([ch['significant_old'] & ~ch['significant_new'], ~ch['significant_old'] & ch['significant_new']],
                         ['lost significance', 'gained significance'], 'unchanged')

print('\nStep D: significance (adjusted p < 0.05), before -> after')
summary = (ch.groupby(['family', 'test', 'bus_uclm_row'])
           .agg(tests=('change', 'size'), significant_before=('significant_old', 'sum'),
                significant_after=('significant_new', 'sum'),
                lost=('change', lambda s: (s == 'lost significance').sum()),
                gained=('change', lambda s: (s == 'gained significance').sum())))
print(summary.to_string())
others = ch[~ch['bus_uclm_row'] & (ch['change'] != 'unchanged')]
print(f'\nBUSI / BrEaST rows whose significance changed through the pooled correction: {len(others)}')
if len(others):
  print(others[['comparison_id', 'comparison_type', 'family', 'data', 'test', 'adj_p_old', 'adj_p_new', 'change']]
        .to_string(index=False))
raw_moved = ch[~ch['bus_uclm_row'] & ((ch['raw_p_old'] - ch['raw_p_new']).abs() > 0)]
print(f'BUSI / BrEaST raw p-values changed: {len(raw_moved)} (must be 0)')
if len(raw_moved):
  stop('A raw p-value outside BUS-UCLM changed.')

# ---------------- write once ----------------
OUT.mkdir(parents=True, exist_ok=True)
for df, path in [(updated, RAW_UPD), (corrected, CORR_UPD), (ch, CHANGES)]:
  if path.exists():
    existing = pd.read_csv(path, float_precision='round_trip')
    fresh = pd.read_csv(io.StringIO(df.to_csv(index=False)), float_precision='round_trip')
    if not existing.equals(fresh):
      stop(f'{path.name} already exists and DIFFERS from this run.')
    print(f'  {path.name} already exists and is identical; left untouched.')
  else:
    df.to_csv(path, index=False)
    print(f'  wrote {path.name}')
print('\nCORRECTION REDONE.')
