# Supplementary Material

This package contains the Python implementation and the complete image set
used for the experiments in the article.  The experiment folders follow the
section numbering of the manuscript.  Generated results are not included;
each script creates its own CSV, JSON, and figure files under the output
directory supplied on the command line.

## Installation

Python 3.12.10 was used for the reported experiments.

```text
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows
PowerShell or `source .venv/bin/activate` on Linux and macOS, and then run:

```text
python -m pip install -r requirements.txt
```

The JPEG2000 experiments require a Pillow build with JPEG2000/OpenJPEG
support.  This support is available in the pinned environment used for the
reported results.

The file `checksums.sha256` lists the SHA-256 digest of every supplied file
except the manifest itself.

Run the commands below from the package root.  When no custom grid is
supplied, omitting `--full` performs a small one-pair check.  Add `--full` to
reproduce the complete experiment.  Independent host-watermark pairs can be
evaluated concurrently with `--workers N` where the option is available.

To reproduce all reported experiments and supplementary analyses, run:

```text
python run_all_experiments.py --full --retune-kodak --workers 8
```

In the Kodak24 step, this command first repeats the PSNR-only global-strength
selection and then runs the benchmark with the selected strengths.  Omit
`--retune-kodak` to reproduce the benchmark directly with the reported global
strengths.  A full all-experiments run also evaluates the auxiliary
`Delta=120` and `Delta=160` settings reported in Supplementary Results
Section S4.  Those files are written under `results/S4_delta120` and
`results/S4_delta160`; neither run overwrites the primary `Delta=140`
results.

## Section 4.2

```text
python 4.2/correction_budget_elbow.py --full --workers 8
python 4.2/operating_point_sensitivity.py --full --workers 8
python 4.2/rank_and_mechanism_ablation.py --full --workers 8
```

- `correction_budget_elbow.py` evaluates `K_max` on the 42-cell grid and
  performs the ten-seed check around the selected budget.
- `operating_point_sensitivity.py` evaluates the complete 42-cell
  `d x Delta` grid, including JPEG2000, and creates the main heatmaps.
- `rank_and_mechanism_ablation.py` evaluates ranks 1--5 and the
  order-preservation/fallback mechanism combinations.

The retained reference point is recorded in `core/experiment_settings.py`.
The operating-point script reports the strictly admissible cells and writes
`operating_point_reference.json`, which records the author-retained
`(d, Delta)=(8, 140)` reference point separately.  The script does not impose
a universal PSNR threshold or select a grid-wide optimum.

To evaluate only specified boundary values while retaining all 32 pairs, use
`--block-sizes` and `--deltas`; for example,
`--block-sizes 8 --deltas 140 160`.

## Section 4.3

```text
python 4.3/controlled_robustness_comparison.py --full --workers 8
python 4.3/end_to_end_runtime.py --full
```

The first command compares RBSVD-QIM, Channel-SVD-QIM, and dense-adjoint
QSVD-QIM on the eight host images and four payloads.  The second performs one
serial in-memory embed-and-extract measurement for every pair and method.

## Section 4.4

```text
python 4.4/kodak24_published_method_benchmark.py --full --workers 8
```

This command uses the global strength values selected by the PSNR-only tuning
experiment and evaluates the proposed method together with nine disclosed
author reimplementations on all 24 x 4 Kodak-watermark pairs.  To repeat the
global-strength selection before the benchmark, run:

```text
python 4.4/kodak24_published_method_benchmark.py --full --retune --max-evals 18 --workers 8
```

The tuner evaluates at most 18 strength candidates per comparison method.
The benchmark also writes the nonnegative-target feasibility counts used in
Supplementary Results.

## Supplementary Results

The separate Supplementary Results document accompanying the article contains
the attack-wise operating-point maps, the brightness-offset analysis, the
published-method implementation details, and the auxiliary `Delta=120` and
`Delta=160` controlled and Kodak24 results. The scripts used to generate these
results are included below.

After the Section 4.2 operating-point experiment, create the attack-wise
Section S1 atlas with:

```text
python Supplementary_Results/S1/attack_wise_heatmaps.py results/operating_point/operating_point_attack_summary.csv
```

The atlas command requires the full Section 4.2 summary in the same output
root.  The all-experiments command above creates the atlas immediately after
that summary is available.

Run the Section S2 brightness-offset experiment with:

```text
python Supplementary_Results/S2/brightness_offset_sweep.py --full --workers 8
```

`Supplementary_Results/S3` contains one source file per
published method, except that WQSD and WQHD share one file because their
quaternion QIM and block-selection operations are common.

Section S4 uses the existing Section 4.3 and 4.4 experiment files with two
explicit auxiliary QIM steps.  Run all four S4 experiments with:

```text
python run_all_experiments.py --section S4 --s4-deltas 120 160 --full --workers 8
```

The equivalent individual commands are:

```text
python 4.3/controlled_robustness_comparison.py --full --delta 120 --workers 8 --output results/S4_delta120/controlled_comparison
python 4.4/kodak24_published_method_benchmark.py --full --delta 120 --retune --max-evals 18 --workers 8 --output results/S4_delta120/kodak_benchmark
python 4.3/controlled_robustness_comparison.py --full --delta 160 --workers 8 --output results/S4_delta160/controlled_comparison
python 4.4/kodak24_published_method_benchmark.py --full --delta 160 --retune --max-evals 18 --workers 8 --output results/S4_delta160/kodak_benchmark
```

The controlled comparison uses the same eight hosts and four payloads as the
primary experiment.  In the Kodak24 run, all nine comparison strengths are
selected again by the same PSNR-only rule at the corresponding RBSVD target.
A non-default `--delta` is therefore rejected by the Kodak24 script unless
`--retune` is also supplied.

## Inputs and outputs

- `images/hosts`: eight host images used in Sections 4.2 and 4.3.
- `images/watermarks`: two binary and two RGB121 watermark images.
- `images/kodak24`: the 24 Kodak benchmark images.
- `core`: the proposed method, payload codec, shared settings, and method
  adapters used by the section-specific scripts.

The host set is Airplane, Barbara, Boats, Fruits, Goldhill, Mandrill, Peppers,
and Sailboat.  The exact files were obtained from the
[Test Images Collection](https://www.hlevkin.com/hlevkin/06testimages.htm)
and the
[Standard Test Images for Image Processing](https://github.com/mohammadimtiazz/standard-test-images-for-Image-Processing)
repository.

The two 16 x 16 color watermarks are already on the RGB121 reconstruction
lattice.  They are serialized in row-major order using one red bit, two green
bits, and one blue bit per pixel.  The two binary watermarks are serialized as
32 x 32 row-major bit arrays.  SAU is supplied as the exact binary array;
the Peugeot source is centre-cropped, resized, and thresholded by the fixed
rules in `core/watermark_codec.py`.  Every payload therefore contains 1024
bits.

The complete experiments are computationally intensive, particularly the
42-cell correction-budget study and the ten-method Kodak24 benchmark.  The
one-pair checks are intended only to verify installation and file access; they
are not publication results.

The proposed-method fallback is evaluated when the primary trajectory either
does not satisfy its residual threshold or gives an incorrect final read-back.
For published methods whose QIM carrier is nonnegative, a negative nearest
representative is shifted by one period to the first nonnegative representative
on the same-bit lattice; the manuscript records this common-protocol
feasibility adaptation.
