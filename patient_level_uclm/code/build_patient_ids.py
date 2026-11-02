"""
build_patient_ids.py -- the single source of truth for BUS-UCLM patient IDs.

Writes (only if EVERY check passes):
  data/new/uclm_patient_ids.csv    one row per image: image_name, patient_id,
                                   true_label, stratum
  data/new/uclm_patient_table.csv  one row per patient: class counts, stratum

Write-once: if an output already exists, the script does not overwrite it; it
checks the existing file is identical to what it would write, and stops if not.

Expected values come from inspect_uclm_names.py (run 2026-10-04) and the
exclusions documented in Supplementary S2 (683 -> 629 images; patient HESN,
13 images, excluded entirely -> 37 patients).
"""
import hashlib
import os
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(os.environ.get('UCLM_BASE', Path.home() / 'uclm_rerun'))
ORIG = BASE / 'data' / 'original'
NEW = BASE / 'data' / 'new'
IDS_PATH = NEW / 'uclm_patient_ids.csv'
TABLE_PATH = NEW / 'uclm_patient_table.csv'
KEY = ['architecture', 'mode', 'arm']

# Expected per-patient counts (benign, malignant, normal), from the inspection run.
EXPECTED_TABLE = {
    'ALWI': (21,  0,  8), 'ANAT': ( 0,  0, 21), 'ANFO': ( 6,  0, 10),
    'ASSC': (10,  0, 18), 'CAWI': (10,  0, 14), 'CHCO': ( 0,  1,  3),
    'CHSP': ( 0,  3, 10), 'CHVI': ( 2, 12,  8), 'CODE': ( 0,  4,  3),
    'COPE': ( 0,  4, 12), 'COST': ( 2,  0, 10), 'COVA': ( 0,  2, 13),
    'CRCI': ( 0,  3,  0), 'DAPA': (22,  0, 14), 'ELCO': ( 0,  3,  4),
    'FLBA': ( 1,  3, 15), 'FLKA': ( 0,  3, 11), 'FUHI': ( 1,  2,  4),
    'HUBL': ( 1,  5, 13), 'KIFO': ( 5,  0,  5), 'LOTI': ( 0,  5,  4),
    'MENE': ( 1,  4, 11), 'NIRO': ( 1,  0, 22), 'ORPE': (10,  0, 20),
    'OSCU': ( 9,  0, 14), 'PAGY': ( 0,  4,  8), 'PLBA': (11,  0, 14),
    'POFR': ( 4,  0, 22), 'RARE': ( 3,  0,  0), 'SECH': ( 5,  0,  9),
    'SHST': ( 3,  0,  7), 'SIBA': ( 0,  8,  0), 'STSP': ( 6,  0, 15),
    'TOCI': (11,  0, 26), 'UNCU': ( 0,  5,  8), 'VITR': ( 9,  0, 20),
    'WAQU': ( 0,  0,  8),
}
EXPECTED = {'images': 629, 'patients': 37, 'with_malignant': 17,
            'without_malignant': 20, 'benign': 154, 'malignant': 71, 'normal': 404}
EXCLUDED_PATIENT = 'HESN'


def stop(msg):
    print(f'\n[STOP] {msg}\nNothing was written.')
    sys.exit(1)


def check(condition, ok_msg, fail_msg):
    if not condition:
        stop(fail_msg)
    print(f'  OK  {ok_msg}')


# ---------------- 1. load; all 18 models must agree ----------------
pred = pd.read_csv(ORIG / 'predictions.csv', float_precision='round_trip')
uclm = pred[pred['eval_data'] == 'bus_uclm']
print('Checks:')
check(uclm.groupby(KEY).ngroups == 18, '18 models on BUS-UCLM',
      'Expected 18 models evaluated on BUS-UCLM.')

reference = None
for key, grp in uclm.groupby(KEY):
    labels = grp.set_index('image_name')['true_label'].sort_index()
    if labels.index.duplicated().any():
        stop(f'Duplicate image names in model {key}.')
    if reference is None:
        reference = labels
    elif not labels.equals(reference):
        stop(f'Model {key} has different images or labels from the first model.')
print('  OK  all 18 models: same images, same labels, no duplicates')

images = pd.DataFrame({'image_name': reference.index, 'true_label': reference.values})
check(len(images) == EXPECTED['images'], f'{len(images)} images',
      f'{len(images)} images, expected {EXPECTED["images"]}.')

# ================= CORE: the patient-ID rule =================
# Name = 4 uppercase letters (patient), underscore, 3 digits (image), .png.
# fullmatch, not match: the WHOLE name must fit, or we stop.
NAME_PATTERN = re.compile(r'([A-Z]{4})_(\d{3})\.png')
parsed = images['image_name'].map(NAME_PATTERN.fullmatch)
bad = images.loc[parsed.isna(), 'image_name'].tolist()
check(not bad, 'every name matches LLLL_DDD.png', f'Names not matching the pattern: {bad[:10]}')
images['patient_id'] = [m.group(1) for m in parsed]
# Stratum for the stratified patient bootstrap: does the patient have any malignant image?
has_malignant = images.groupby('patient_id')['true_label'].apply(lambda s: (s == 1).any())
images['stratum'] = images['patient_id'].map(
    lambda p: 'has_malignant' if has_malignant[p] else 'no_malignant')
# =============================================================

# ---------------- 2. patient-level checks ----------------
table = (images.groupby('patient_id')['true_label'].value_counts()
         .unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0))
table.columns = ['benign', 'malignant', 'normal']
table['total'] = table.sum(axis=1)
table['stratum'] = ['has_malignant' if has_malignant[p] else 'no_malignant' for p in table.index]

check(EXCLUDED_PATIENT not in table.index, f'excluded patient {EXCLUDED_PATIENT} absent',
      f'Excluded patient {EXCLUDED_PATIENT} is present.')
check(len(table) == EXPECTED['patients'], f'{len(table)} patients',
      f'{len(table)} patients, expected {EXPECTED["patients"]}.')
n_with = int(has_malignant.sum())
check(n_with == EXPECTED['with_malignant'] and len(table) - n_with == EXPECTED['without_malignant'],
      f'strata: {n_with} with malignant, {len(table) - n_with} without',
      f'Strata are {n_with}/{len(table) - n_with}, expected 17/20.')
for cls in ['benign', 'malignant', 'normal']:
    check(table[cls].sum() == EXPECTED[cls], f'{cls}: {table[cls].sum()} images',
          f'{cls}: {table[cls].sum()} images, expected {EXPECTED[cls]}.')

observed = {p: (int(r.benign), int(r.malignant), int(r.normal)) for p, r in table.iterrows()}
mismatch = sorted(p for p in set(observed) | set(EXPECTED_TABLE)
                  if observed.get(p) != EXPECTED_TABLE.get(p))
check(not mismatch, 'per-patient counts identical to the inspected table',
      f'Per-patient counts differ for: {mismatch}')
check(images.groupby('image_name')['patient_id'].nunique().max() == 1,
      'each image belongs to exactly one patient', 'An image maps to more than one patient.')

# ---------------- 3. write once ----------------
NEW.mkdir(parents=True, exist_ok=True)
ids_out = images[['image_name', 'patient_id', 'true_label', 'stratum']].sort_values('image_name')
table_out = table.reset_index()


def write_once(df, path):
    if path.exists():
        existing = pd.read_csv(path, float_precision='round_trip')
        if not existing.equals(df.reset_index(drop=True)):
            stop(f'{path.name} already exists and DIFFERS from what this run produced.')
        print(f'  {path.name} already exists and is identical; left untouched.')
    else:
        df.to_csv(path, index=False)
        print(f'  wrote {path.name}')
    print(f'    sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}')


print('\nOutputs:')
write_once(ids_out, IDS_PATH)
write_once(table_out, TABLE_PATH)
print('\nPATIENT IDS BUILT: 629 images -> 37 patients (17 with malignant, 20 without).')
