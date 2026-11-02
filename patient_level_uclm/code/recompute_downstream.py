"""
recompute_downstream.py -- rerun the significance-dependent conclusions of Notebook 04
(Tier 1 pairwise table, ResNet-50 TL-vs-scratch, Clever Hans ablation) on the updated
patient-level results, using the VERBATIM notebook cells (original_downstream.py).

  Step A  Run the cells on the ORIGINAL corrected results; they must reproduce the
          saved notebook outputs and saved files exactly, or STOP:
            - cell 109 printed table (12 rows)          -> embedded below
            - cell 113 printed values (ResNet-50)       -> embedded below
            - ablation_arm_comparisons.csv (36 rows)    -> data/original/
            - direction counts in ablation_clever_hans_summary.json
  Step B  Run the same cells on the UPDATED corrected results.
  Step C  Check every hand-typed claim of cells 114 and 121, old vs new.

Unaffected by the reanalysis (no significance involved): Tier 0 (safety floor) and
Tier 2 (tiebreak) -- they use per-model metrics only.

Writes (write-once) to data/new/patient_level/:
  ablation_arm_comparisons_updated.csv, tier1_pairwise_updated.csv, downstream_claims_check.csv
"""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import original_downstream as od

BASE = Path(os.environ.get('UCLM_BASE', Path.home() / 'uclm_rerun'))
ORIG, OUT = BASE / 'data' / 'original', BASE / 'data' / 'new' / 'patient_level'
REGISTRY = ['comparison_id', 'comparison_type', 'family', 'pairing_type', 'arch_a', 'mode_a',
            'arm_a', 'eval_data_a', 'arch_b', 'mode_b', 'arm_b', 'eval_data_b']
EVAL_DATASETS = ['busi_test', 'bus_uclm', 'breast']          # Notebook 04 cell 40

# Saved output of cell 109 (sensitivity90_diff printed to 6 dp; all 12 not significant)
SAVED_109 = [('TL', 'resnet50', 'efficientnet_b0', 'bus_uclm', -0.056338), ('TL', 'resnet50', 'efficientnet_b0', 'breast', -0.059184),
             ('TL', 'resnet50', 'swin_tiny', 'bus_uclm', -0.154930), ('TL', 'resnet50', 'swin_tiny', 'breast', 0.048980),
             ('TL', 'efficientnet_b0', 'swin_tiny', 'bus_uclm', -0.098592), ('TL', 'efficientnet_b0', 'swin_tiny', 'breast', 0.108163),
             ('scratch', 'resnet50', 'efficientnet_b0', 'bus_uclm', 0.152113), ('scratch', 'resnet50', 'efficientnet_b0', 'breast', -0.006122),
             ('scratch', 'resnet50', 'swin_tiny', 'bus_uclm', 0.109859), ('scratch', 'resnet50', 'swin_tiny', 'breast', 0.014286),
             ('scratch', 'efficientnet_b0', 'swin_tiny', 'bus_uclm', -0.042254), ('scratch', 'efficientnet_b0', 'swin_tiny', 'breast', 0.020408)]
# Saved output of cell 113
SAVED_113 = {'breast': (0.075019, 0.2736, False), 'bus_uclm': (0.019053, 1.0, False), 'busi_test': (0.085866, 0.5202, False)}


def stop(msg):
  sys.exit(f'\n[STOP] {msg}\nNothing was written. Investigate before continuing.')


def run_cells(stats_corrected):
  """Execute the verbatim cells on one corrected table; return their namespace."""
  ns = {'pd': pd, 'np': np, 'stats_corrected': stats_corrected, 'EVAL_DATASETS': EVAL_DATASETS,
        'comparison_registry': stats_corrected[REGISTRY].copy()}
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    exec(od.TL_CHECK_RECONSTRUCTION, ns)
    for i in [109, 113, 117, 118, 119, 120]:
      exec(od.CELLS[i], ns)
  ns['_printed'] = buf.getvalue()
  return ns


def facts(ns):
  """The computed quantities behind every hand-typed claim."""
  sc, tl, ch, se = ns['stats_corrected'], ns['tl_check'], ns['clever_hans_df'], ns['sig_external']
  f1_sig_pairs = []
  for mode in ['TL', 'scratch']:
    for a1, a2 in [('resnet50', 'efficientnet_b0'), ('resnet50', 'swin_tiny'), ('efficientnet_b0', 'swin_tiny')]:
      for ed in ['bus_uclm', 'breast']:
        if ns['find_arch_comparison'](a1, a2, mode, ed)['macro_f1_significant']:
          f1_sig_pairs.append(f'{mode} {a1} vs {a2} on {ed}')
  tl_sig = {arch: sorted(tl[(tl['arch_a'] == arch) & tl['bootstrap_macro_f1_p_significant']]['eval_data_a'])
            for arch in ['resnet50', 'efficientnet_b0', 'swin_tiny']}
  internal, external = ch[~ch['is_external']], ch[ch['is_external']]
  dirs = se['direction'].value_counts().to_dict() if len(se) else {}
  return {
      'tier1_sensitivity90_significant_of_12': int(ns['pairwise_df']['sensitivity90_significant'].sum()),
      'tier1_macro_f1_significant_pairs': f1_sig_pairs,
      'resnet50_tl_ahead_all_3_datasets': bool(ns['resnet50_all_favor_tl']),
      'resnet50_tl_significant_any_dataset': bool(ns['resnet50_any_significant']),
      'tl_significantly_ahead_macro_f1_by_architecture': tl_sig,
      'clever_hans_internal_significant_of_12': int((internal['sensitivity90_significant'] | internal['macro_f1_significant']).sum()),
      'clever_hans_external_significant_of_24': int(len(se)),
      'clever_hans_significant_on_bus_uclm': int((se['eval_data'] == 'bus_uclm').sum()) if len(se) else 0,
      'clever_hans_significant_on_breast': int((se['eval_data'] == 'breast').sum()) if len(se) else 0,
      'clever_hans_directions': {k: int(v) for k, v in dirs.items()},
      'clever_hans_sensitivity90_ever_significant': bool(ch['sensitivity90_significant'].any()),
  }


# ---------------- Step A: reproduce the original downstream outputs ----------------
print('Step A: reproduce the original downstream outputs ...')
orig_corr = pd.read_csv(ORIG / 'statistical_results_corrected.csv', float_precision='round_trip')
old = run_cells(orig_corr)

pw = old['pairwise_df']
ok_109 = (len(pw) == 12 and not pw['sensitivity90_significant'].any() and
          all((r.mode, r.arch1, r.arch2, r.eval_data) == s[:4] and round(r._5, 6) == s[4]
              for r, s in zip(pw.itertuples(), SAVED_109)))
tl_old = old['resnet50_tl_vs_scratch']
ok_113 = all((round(r.bootstrap_macro_f1_diff, 6), round(r.bootstrap_macro_f1_p_adj, 4),
              bool(r.bootstrap_macro_f1_p_significant)) == SAVED_113[r.eval_data_a] for r in tl_old.itertuples())
saved_ablation = pd.read_csv(ORIG / 'ablation_arm_comparisons.csv', float_precision='round_trip')
ch_old = old['clever_hans_df'].reset_index(drop=True)
num = ['sensitivity90_diff_(inpainted-other)', 'macro_f1_diff_(inpainted-other)']
ok_csv = (list(ch_old.columns) == list(saved_ablation.columns) and len(ch_old) == 36 and
          all((ch_old[c].values == saved_ablation[c].values).all() for c in ch_old.columns if c not in num) and
          max(np.max(np.abs(ch_old[c].values - saved_ablation[c].values)) for c in num) <= 1e-12)
saved_summary = json.loads((ORIG / 'ablation_clever_hans_summary.json').read_text())
ok_dirs = facts(old)['clever_hans_directions'] == saved_summary['direction_among_significant']
for name, ok in [('cell 109 table (12 rows) as saved in the notebook', ok_109),
                 ('cell 113 ResNet-50 values as saved in the notebook', ok_113),
                 ('ablation_arm_comparisons.csv (36 rows)', ok_csv),
                 ('Clever Hans direction counts in the saved summary', ok_dirs)]:
  print(f'  [{"PASS" if ok else "FAIL"}] reproduces {name}')
if not all([ok_109, ok_113, ok_csv, ok_dirs]):
  stop('The original downstream outputs were NOT reproduced.')

# ---------------- Step B: rerun on the updated results ----------------
print('\nStep B: rerun the same cells on the updated (patient-level) results ...')
upd_corr = pd.read_csv(OUT / 'statistical_results_corrected_updated.csv', float_precision='round_trip')
if not (upd_corr[REGISTRY].values == orig_corr[REGISTRY].values).all():
  stop('Updated table rows do not line up with the original table.')
new = run_cells(upd_corr)
f_old, f_new = facts(old), facts(new)

print('\n  Tier 1 pairwise (sensitivity90), updated:')
print('  ' + new['pairwise_df'].to_string(index=False).replace('\n', '\n  '))
print('\n  ResNet-50 TL vs scratch (macro-F1), updated:')
print('  ' + new['resnet50_tl_vs_scratch'].to_string(index=False).replace('\n', '\n  '))
print('\n  Clever Hans: significant external rows, updated:')
se = new['sig_external']
print('  ' + (se[['architecture', 'mode', 'eval_data', 'comparison', 'sensitivity90_diff_(inpainted-other)',
                  'macro_f1_diff_(inpainted-other)', 'direction']].to_string(index=False).replace('\n', '\n  ')
              if len(se) else 'none'))

# ---------------- Step C: the hand-typed claims, old vs new ----------------
CLAIMS = [
    ('Tier 1: "significant_comparisons: 0" (of 12, sensitivity90)', 'tier1_sensitivity90_significant_of_12'),
    ('Tier 1: "No architecture significantly outperforms another" -- macro-F1 pairs significant', 'tier1_macro_f1_significant_pairs'),
    ('ResNet-50: TL numerically ahead on all 3 datasets', 'resnet50_tl_ahead_all_3_datasets'),
    ('ResNet-50: TL significantly ahead on any dataset (claim: False)', 'resnet50_tl_significant_any_dataset'),
    ('"unlike efficientnet_b0 and swin_tiny, where TL\'s advantage IS significant" (datasets)', 'tl_significantly_ahead_macro_f1_by_architecture'),
    ('Clever Hans: "No significant arm effect internally (0/12)"', 'clever_hans_internal_significant_of_12'),
    ('Clever Hans: "Externally, 9/24 significant"', 'clever_hans_external_significant_of_24'),
    ('Clever Hans: "All 9 significant results are on bus_uclm" -- on bus_uclm', 'clever_hans_significant_on_bus_uclm'),
    ('Clever Hans: ... -- on breast', 'clever_hans_significant_on_breast'),
    ('Clever Hans: "split roughly evenly confirming vs. disconfirming (5 vs 4)"', 'clever_hans_directions'),
    ('Clever Hans: "sensitivity90 never reaches significance anywhere"', 'clever_hans_sensitivity90_ever_significant'),
]
print('\nStep C: every hand-typed claim, original computation vs updated computation')
rows = []
for text, key in CLAIMS:
  o, n = f_old[key], f_new[key]
  rows.append({'claim': text, 'original': json.dumps(o), 'updated': json.dumps(n), 'changed': o != n})
  print(f'\n  {text}\n    original: {o}\n    updated:  {n}{"   <-- CHANGED" if o != n else ""}')

# ---------------- write once ----------------
OUT.mkdir(parents=True, exist_ok=True)
outputs = [(new['clever_hans_df'], OUT / 'ablation_arm_comparisons_updated.csv'),
           (new['pairwise_df'], OUT / 'tier1_pairwise_updated.csv'),
           (pd.DataFrame(rows), OUT / 'downstream_claims_check.csv')]
print()
for df, path in outputs:
  if path.exists():
    existing = pd.read_csv(path, float_precision='round_trip')
    fresh = pd.read_csv(io.StringIO(df.to_csv(index=False)), float_precision='round_trip')
    if not existing.equals(fresh):
      stop(f'{path.name} already exists and DIFFERS from this run.')
    print(f'  {path.name} already exists and is identical; left untouched.')
  else:
    df.to_csv(path, index=False)
    print(f'  wrote {path.name}')
print('\nDOWNSTREAM RECOMPUTED. The hand-typed summaries must now be rewritten by you from Step C.')
