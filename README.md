# RBSVD--QIM Color Image Watermarking

This repository contains the reference implementation, experimental scripts,
test images, and supplementary results accompanying the manuscript
*Blind Color Image Watermarking Based on Reduced Biquaternion SVD and
Quantization Index Modulation* by Emre Kişi and Mihriban Yüksek.

The proposed method combines reduced biquaternion singular value decomposition
(RBSVD) with quantization index modulation (QIM) for fully blind watermarking
of color images. The supplied material reproduces the operating-point study,
controlled comparisons, Kodak24 published-method benchmark, and supplementary
analyses reported in the manuscript.

## Repository contents

All reproducibility files are stored in
[`Supplementary_Material`](Supplementary_Material/).

- `4.2/`: operating-point, rank, mechanism, and correction-budget experiments.
- `4.3/`: controlled robustness and end-to-end runtime comparisons.
- `4.4/`: Kodak24 benchmark against nine disclosed author reimplementations.
- `Supplementary_Results/`: scripts for Sections S1--S3. The auxiliary
  `Delta=120` and `Delta=160` experiments reported in Section S4 use the
  Section 4.3 and 4.4 scripts.
- `core/`: RBSVD--QIM implementation and shared experiment utilities.
- `images/`: the eight host images, four watermark images, and Kodak24 set used
  in the experiments.
- `checksums.sha256`: SHA-256 digest of every supplied package file except the
  manifest itself.

The detailed experiment commands and protocol notes are given in the
[package README](Supplementary_Material/README.md).

## Quick start

The reported experiments used Python 3.12.10. From the package directory,
create an environment and install the pinned dependencies:

```bash
cd Supplementary_Material
python -m venv .venv
python -m pip install -r requirements.txt
```

Run a small installation check with:

```bash
python run_all_experiments.py
```

Reproduce the complete reported experiment set with:

```bash
python run_all_experiments.py --full --retune-kodak --workers 8
```

The full run is computationally intensive. The package README also lists the
commands for running Sections 4.2, 4.3, 4.4, and S1--S4 separately.

## Reproducibility notes

- The retained main operating point is
  `(d, Delta, r, K_max) = (8, 140, 1, 2)`.
- Supplementary Results Section S4 reports the corresponding controlled and
  Kodak24 experiments at `Delta=120` and `Delta=160`.
- Generated CSV, JSON, and figure files are written to the selected output
  directory and are not included in the repository.
- Image-source information and common-protocol adaptations are recorded in
  `THIRD_PARTY_NOTICE.txt` and the separate Supplementary Results document.

## License and citation

The supplied research code is distributed under the terms stated in
[`LICENSE.txt`](Supplementary_Material/LICENSE.txt). Citation
details will be added when the manuscript receives its final bibliographic
record.
