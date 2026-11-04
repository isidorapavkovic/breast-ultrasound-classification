# Breast ultrasound classification: code, splits, predictions and statistics

Code, data splits, per-image predictions and statistical results accompanying the paper

> **Good Internal Performance Does Not Mean Clinical Safety: External Validation of Convolutional and Transformer Models for Breast Ultrasound Classification**, Isidora Pavković and Nemanja Milošević, *Cancers*. Manuscript under peer review.

The work was developed as part of an MSc thesis. This repository contains everything needed to
follow the analysis from raw images to the numbers reported in the paper, except the
datasets themselves and the trained model weights (see "What is not included").

## Where is what

| Folder | Contents |
|---|---|
| `notebooks/` | Five notebooks, run in order: `01_data_audit`, `02_preprocessing`, `03_training`, `04_evaluation`, `05_xai`. |
| `src/` | Python modules used by the notebooks (models, utilities, dataset handling, artefact/colour/text detection, mask generation) and `decisions_manual.csv`. |
| `splits_and_config/` | Train/test splits, normalisation statistics, sweep winners, final model configurations and training results. |
| `results/data_audit/` | Dataset audit tables and the mask generation log. |
| `results/metrics/` | Technical and clinical metrics, batch latencies, computational cost, confusion matrices. |
| `results/statistics/` | Best-model summary. |
| `results/gradcam/` | Grad-CAM localisation results and the Clever Hans summary. |
| `results/radiologist_review/` | Radiologist's qualitative review of Grad-CAM cases and the answer key. |
| `patient_level_uclm/` | Patient-level re-analysis of the external BUS-UCLM data: `code/`, `r/`, `data/original/` (per-image predictions and image-level statistics), `data/new/` and `data/new/patient_level/` (outputs; `statistical_results_corrected_updated.csv` is the version of record, Table S6). |

## Datasets (not included)

The images are not redistributed here. Download them from the original sources and cite the original papers.

| Dataset | Role in this study | Images used* | Source | Reference |
|---|---|---|---|---|
| BUSI | training and internal testing | 530 | [Cairo University dataset page](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) | Al-Dhabyani et al., *Data in Brief* 28 (2020) 104863, [doi:10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863) |
| BUS-UCLM | external validation | 629 (37 patients) | [Mendeley Data, version 3](https://data.mendeley.com/datasets/7fvgj4jsp7/3) | Vallez et al., *Scientific Data* 12 (2025) 242, [doi:10.1038/s41597-025-04562-3](https://doi.org/10.1038/s41597-025-04562-3) |
| BrEaST | external validation | 256 | [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/) | Pawłowska et al., *Scientific Data* 11 (2024) 148, [doi:10.1038/s41597-024-02984-z](https://doi.org/10.1038/s41597-024-02984-z) |

\* After the exclusions described in the paper, 1,415 of the 1,719 original images were used. BrEaST and BUS-UCLM are released under CC BY 4.0. The BUSI source page does not state a licence and asks users to cite the paper listed above.

## Running the notebooks

The notebooks were run in Google Colab (GPU: NVIDIA L4) with the project files on Google Drive. Run them in order, `01` to `05`.

1. **Data.** Download the three datasets (see above) onto your Google Drive.
2. **Comment list (not redistributed).** Notebook 02 reads `src/dataset_comment_list.csv`, the list of comments on BUSI images published by Pawłowska, Karwat and Żołek (*Data in Brief* 48 (2023) 109247; [Mendeley Data, doi:10.17632/k8t3gnx9h6.1](https://doi.org/10.17632/k8t3gnx9h6.1), CC BY 4.0). Download it and save it as `src/dataset_comment_list.csv`. The file used in this study is semicolon-separated (columns `ID;Filename;Objection;Annotation`, 588 lines) and has SHA-256 `e3c2abc0ce027401dc717bf8a7ec76442e30c3cda2df32a0bc8aa197ff1c74a4`.
3. **BUSClean.** Notebook 02 clones [BUSClean](https://github.com/hawaii-ai/bus-cleaning) (MIT licence) and pins it to commit `d2630a435cd257e366a516a3c5c5cf26c27b3243`; no manual installation is needed.
4. **Weights & Biases.** The notebooks that log to W&B read the API key from Colab Secrets.
5. **Paths and names.** Set the constants listed below.

### Settings to change

| Notebook | Setting | What to set |
|---|---|---|
| `01_data_audit`, `02_preprocessing`, `05_xai` | `ROOT` (near the top, in the path constants) | Your Google Drive project folder. |
| `03_training` | `DRIVE_FINAL_DIR` (section 1.5) and `ROOT` (section 2.2) | Your Google Drive project folder. Both contain the full path, so change both. |
| `04_evaluation` | `ROOT` (section 1.3) | The same project folder. |
| `03_training`, `04_evaluation` | `WANDB_ENTITY`, `WANDB_PROJECT` | Your own W&B account and project. The values in the notebooks (`thesis-mine`) belong to the original run. |
| `05_xai` | folder name `thesis_figures` | Legacy name of an output folder; it can be left as it is. |

### Software versions

Versions used in the study (Supplementary S6):

| Component | Version |
|---|---|
| PyTorch | 2.11.0 (CUDA 12.8) |
| torchvision | 0.26.0 |
| Weights & Biases | 0.28.1 |
| OpenCV | 5.0.0 |
| EasyOCR | 1.7.2 |
| pytorch-grad-cam | 1.5.7 |
| NumPy | 2.1.3 |
| SciPy | 1.16.3 |
| scikit-learn | 1.6.1 |
| pandas | 2.2.2 |
| R (patient-level analysis) | 4.6.1 |
| clust.bin.pair (patient-level analysis) | 0.1.2 |

The Python environment of the patient-level analysis is in `patient_level_uclm/requirements-lock.txt`.

## What is not included

- **Trained model weights.** The 18 final checkpoints total about 1.3 GB, and the Swin files are over GitHub's 100 MB limit for a single file. `03_training` retrains them.
- **Grad-CAM++ heatmap arrays** (`gradcam_heatmaps.npz`, about 72 MB). `05_xai` regenerates them.
- **Segmentation mask images** (1,159 PNG files).
- **Radiologist packet images.** The case ratings and comments are in `results/radiologist_review/`.
- **The three datasets and the comment list** (see "Datasets" and "Running the notebooks" above).
- **The Lerner clustered-ROC R code and `MRA.csv`**, which are not ours to redistribute. `patient_level_uclm/r/DOWNLOADED.txt` lists where to get them and their SHA-256 checksums.

## Notes on the result files

- **Google Drive paths.** Some tables (`audit_master.csv`, `df_clean_post_exclusion.csv`, `final_models.json`, `notebook03_results_summary.csv`, `computational_cost_full.csv`) contain paths starting with `/content/drive/MyDrive/...`. They record where the dataset images and checkpoints were during the study. They are not needed to read the tables and do not point to anything you can open.
- **Version of record for the statistics.** `patient_level_uclm/data/new/patient_level/statistical_results_corrected_updated.csv` is the version of record and the source of Table S6 in the Supplementary Material. It holds image-level rows and patient-level rows, told apart by the `analysis_unit` column. `patient_level_uclm/data/original/statistical_results_corrected.csv` is an input to the patient-level scripts, not a final result.
- **`avg_ms_per_image`** is the median of the per-batch medians, then the median over the three evaluation datasets (Supplementary S5).

## Radiologist review

- **Files.** `results/radiologist_review/radiologist_qualitative_review.csv` holds the radiologist's ratings and comments on the reviewed cases. `radiologist_packet_answer_key.csv` in the same folder is the answer key of the 20-case packet.
- **How the comments were recorded.** The comments were spoken by the radiologist, transcribed by the first author and read back to her for confirmation.
- **Columns.** `radiologist_agreement` is the "Rating" column of Table S5 in the Supplementary Material. `flag` holds additional remarks by the radiologist for `case_04`, `case_05`, `case_09` and `case_14`; they are not in Table S5.

## Patient-level re-analysis

- **What it is.** `patient_level_uclm/` repeats the 63 BUS-UCLM comparisons at the patient level (37 patients), using cluster-aware tests and a patient-level bootstrap. A patient ID is the four-letter prefix of a BUS-UCLM image name. The results are the patient-level rows of `statistical_results_corrected_updated.csv` (see "Notes on the result files").
- **Setup.** The scripts in `code/` find their folders through the environment variable `UCLM_BASE` (default `~/uclm_rerun`). Point it at the `patient_level_uclm/` folder, or copy that folder to `~/uclm_rerun`. It must contain `r/`, `data/original/` and `data/new/`. You need R 4.6.1 with `clust.bin.pair` 0.1.2 (the R script refuses other versions) and the Python packages in `requirements-stats.txt` (exact versions in `requirements-lock.txt`).
- **Order.** Run `build_patient_ids.py`, then `run_patient_level.py` (it takes a while and can be interrupted and resumed), then `correct_patient_level.py`. Each script writes its outputs once and stops if an existing output differs from what it would write.
- **Files that are not ours to redistribute.** `funcs_clusteredROC.R` and `MRA.csv` come from the Lerner Research Institute (https://lerner.ccf.org/quantitative-health/software/) and must be placed in `r/`. `r/DOWNLOADED.txt` has the SHA-256 checksums of the versions used. The R function is sourced unchanged. `MRA.csv` is the published example data (Obuchowski 1997) used by check C1 in `test_clustered_tests.py`, which must reproduce the values in the Lerner help document before any result is used.
- **Provenance.** The files in `patient_level_uclm/data/` are byte-identical to those used in the 6 October 2026 run (SHA-256 verified), except for one sentence in `data/original/ablation_clever_hans_summary.json` (`finding`), which was corrected to state the counts (8 on BUS-UCLM, 1 on BrEaST). `clinical_metrics.csv` and `technical_metrics.csv` are in `results/metrics/`.
