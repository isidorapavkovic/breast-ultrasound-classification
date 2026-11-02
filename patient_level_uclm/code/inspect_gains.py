"""
inspect_gains.py -- WHY did comparisons 80, 87 and 148 gain significance under
Durkalski's test? Read-only; prints, writes nothing.

For each comparison: the image-level picture (b = A right/B wrong, c = A wrong/B right,
which the original McNemar used) and the patient-level picture Durkalski uses: every
patient with any disagreement, how many images they have, and which model they favour.
Durkalski sums (b_k - c_k) / n_k, so each patient counts equally, however many images.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import original_functions as of
import patient_bootstrap as pb

BASE = Path(os.environ.get('UCLM_BASE', Path.home() / 'uclm_rerun'))
ORIG, NEW = BASE / 'data' / 'original', BASE / 'data' / 'new'
GAINS = [80, 87, 148]

predictions = pd.read_csv(ORIG / 'predictions.csv', float_precision='round_trip')
raw = pd.read_csv(ORIG / 'statistical_results_raw.csv', float_precision='round_trip').set_index('comparison_id')
ids = pd.read_csv(NEW / 'uclm_patient_ids.csv')
of.set_predictions(predictions)

for cid in GAINS:
  row = raw.loc[cid]
  a = of.get_predictions_for_condition(row['arch_a'], row['mode_a'], row['arm_a'], 'bus_uclm')
  b = of.get_predictions_for_condition(row['arch_b'], row['mode_b'], row['arm_b'], 'bus_uclm')
  y = a['true_label'].values
  pid, _ = pb.attach_patient_info(a, ids)
  ca, cb = a['pred_label'].values == y, b['pred_label'].values == y
  df = pd.DataFrame({'patient': pid, 'b': ca & ~cb, 'c': ~ca & cb})
  g = df.groupby('patient').agg(n=('b', 'size'), b=('b', 'sum'), c=('c', 'sum'))
  g['net_per_image'] = (g['b'] - g['c']) / g['n']
  disc = g[(g['b'] + g['c']) > 0].sort_values('n', ascending=False)

  B, C = int(g['b'].sum()), int(g['c'].sum())
  fav_a, fav_b, tied = int((disc['b'] > disc['c']).sum()), int((disc['c'] > disc['b']).sum()), int((disc['b'] == disc['c']).sum())
  top3 = disc.head(3)
  print(f'\n=== Comparison {cid}: A = {row["arch_a"]}/{row["mode_a"]}/{row["arm_a"]}   B = {row["arch_b"]}/{row["mode_b"]}/{row["arm_b"]}')
  print(f'  Image level (original McNemar, {row["mcnemar_method"]}, raw p = {row["mcnemar_p"]:.4f}):')
  print(f'    A right & B wrong: {B} images | A wrong & B right: {C} images | net {B - C:+d} images')
  print(f'  Patient level (Durkalski):')
  print(f'    patients with any disagreement: {len(disc)}/37 -> favour A: {fav_a}, favour B: {fav_b}, tied: {tied}')
  print(f'    sum of per-patient net proportions: {disc["net_per_image"].sum():+.3f}')
  print(f'    3 largest disagreeing patients ({", ".join(top3.index)}): '
        f'{int(top3["n"].sum())} images, net {int((top3["b"] - top3["c"]).sum()):+d} images')
  print('    per patient (largest first): patient, images, A-better, B-better, net per image')
  for p, r in disc.iterrows():
    print(f'      {p}  {int(r.n):3d}  {int(r.b):3d}  {int(r.c):3d}  {r.net_per_image:+.3f}')
