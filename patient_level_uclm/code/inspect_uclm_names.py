"""
inspect_uclm_names.py -- LOOK before building patient IDs.

Read-only. Prints how the BUS-UCLM image names look, checks that every model
was evaluated on the same 629 images with the same labels, and shows a
CANDIDATE patient table (text before the first underscore). Nothing is decided
here; the real patient-ID builder comes after this output has been reviewed.
"""
import os
from pathlib import Path

import pandas as pd

BASE = Path(os.environ.get('UCLM_BASE', Path.home() / 'uclm_rerun'))
ORIG = BASE / 'data' / 'original'
CLASS_NAMES = ['benign', 'malignant', 'normal']
KEY = ['architecture', 'mode', 'arm']

pred = pd.read_csv(ORIG / 'predictions.csv', float_precision='round_trip')
uclm = pred[pred['eval_data'] == 'bus_uclm']

# ---- 1. structure: 18 models x 629 images ----
sizes = uclm.groupby(KEY).size()
print(f'BUS-UCLM rows: {len(uclm)} (expected 18 x 629 = 11322)')
print(f'Models evaluated on BUS-UCLM: {len(sizes)} (expected 18)')
print(f'Images per model: {sorted(int(x) for x in sizes.unique())} (expected [629])')

# ---- 2. every model saw the same images, with the same labels ----
reference = None
same_images, same_labels, any_duplicates = True, True, False
for key, grp in uclm.groupby(KEY):
    any_duplicates |= grp['image_name'].duplicated().any()
    labels = grp.set_index('image_name')['true_label'].sort_index()
    if reference is None:
        reference = labels
        continue
    if list(labels.index) != list(reference.index):
        same_images = False
    elif not labels.equals(reference):
        same_labels = False
print(f'Duplicate image names within any model: {any_duplicates} (expected False)')
print(f'All models share the same image names: {same_images} (expected True)')
print(f'All models share the same true labels: {same_labels} (expected True)')

# ---- 3. what the names look like ----
names = pd.Series(reference.index)
print('\nFirst 10 names (sorted):', names.head(10).tolist())
print('Last 5 names (sorted): ', names.tail(5).tolist())
print('5 random names:        ', names.sample(5, random_state=0).tolist())
print(f'Names without an underscore: {(~names.str.contains("_")).sum()}')
print(f'Name lengths seen: {sorted(int(x) for x in names.str.len().unique())}')
print(f'File extensions seen: {sorted(names.str.extract(r"(\.[^.]+)$")[0].dropna().unique())}')

# ---- 4. CANDIDATE patient table (prefix = text before the first underscore) ----
table = pd.DataFrame({'image_name': names.values,
                      'true_label': reference.values,
                      'candidate_id': names.str.split('_').str[0].values})
counts = (table.groupby('candidate_id')['true_label']
          .value_counts().unstack(fill_value=0)
          .reindex(columns=[0, 1, 2], fill_value=0))
counts.columns = CLASS_NAMES
counts['total'] = counts.sum(axis=1)
print(f'\nCandidate patients: {len(counts)}')
print(f'Candidate patients with >=1 malignant image: {(counts["malignant"] > 0).sum()}')
print(f'"HESN" present (the excluded patient): {"HESN" in counts.index} (expected False)')
print(f'Class totals: {counts[CLASS_NAMES].sum().to_dict()} '
      f'(expected benign 154, malignant 71, normal 404)')
print('\nCandidate patient table:')
print(counts.to_string())
