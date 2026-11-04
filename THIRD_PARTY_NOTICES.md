# Third-party notices

This repository uses the following third-party material. The repository's own code is released under the MIT licence (see `LICENSE`).

## BUSClean (modified code included)

- Source: https://github.com/hawaii-ai/bus-cleaning, commit `d2630a435cd257e366a516a3c5c5cf26c27b3243`, MIT licence.
- `src/artefact_detection.py` and `src/text_detection.py` are modified versions of BUSClean modules. BUSClean itself is not included; notebook `02_preprocessing` clones it at the commit above.
- Reference: Bunnell A, Hung K, Shepherd JA, Sadowski P. *BUSClean: Open-source software for breast ultrasound image pre-processing and knowledge extraction for medical AI.* arXiv:2407.11316.

The BUSClean licence text follows.

~~~
MIT License

Copyright (c) 2024 Hawaii AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~

## Comment list on the BUSI dataset (not included)

- Pawłowska A, Karwat P, Żołek N. Letter to the Editor. Re: "Dataset of Breast Ultrasound Images by W. Al-Dhabyani, M. Gomaa, H. Khaled & A. Fahmy, Data in Brief, 2020, 28, 104863". *Data in Brief* 48 (2023) 109247, doi:10.1016/j.dib.2023.109247. The accompanying file is on Mendeley Data, doi:10.17632/k8t3gnx9h6.1, licence CC BY 4.0.
- The file is not redistributed here. `notebooks/02_preprocessing.ipynb` expects it at `src/dataset_comment_list.csv` (see the README for the checksum of the file used in this study).

## Clustered ROC code and `clust.bin.pair` (not included)

- `funcs_clusteredROC.R` and `MRA.csv` are from the Department of Quantitative Health Sciences, Lerner Research Institute, Cleveland Clinic (https://www.lerner.ccf.org/quantitative-health/software/). They are not redistributed. `patient_level_uclm/r/DOWNLOADED.txt` lists their SHA-256 checksums. The function implements: Obuchowski NA. Nonparametric Analysis of Clustered ROC Curve Data. *Biometrics* 1997, 53, 567–578, doi:10.2307/2533958.
- The R package `clust.bin.pair` is installed separately: Gopstein D. *clust.bin.pair: Statistical Methods for Analyzing Clustered Matched Pair Data*, 2018, R package version 0.1.2, doi:10.32614/CRAN.package.clust.bin.pair. It implements: Durkalski VL, Palesch YY, Lipsitz SR, Rust PF. Analysis of Clustered Matched-Pair Data. *Statistics in Medicine* 2003, 22, 2417–2428, doi:10.1002/sim.1438.
