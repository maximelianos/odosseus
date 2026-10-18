# Running the camera localization pipeline

How to set up and run the localization pipeline (iteration 1): the submission for a test run, and
the validation on the training bag. What the method does and why is in [README.md](README.md) and
[docs/DECISIONS.md](docs/DECISIONS.md).

## 1. Requirements

- macOS or Linux, CPU only: no GPU and no pretrained models. Developed on an Apple M4 (10 cores, 16 GB RAM).
- [conda](https://docs.conda.io/en/latest/miniconda.html), or Python 3.12 with pip.
- **Disk**: the training bag (7.5 GB) plus about 0.8 GB of derived data for a test run (caches, car-body
  masks and the map of all laps, 0.74 GB). Each additional validation map takes 0.4–0.5 GB.
- **RAM**: 16 GB is enough; a run uses about 1.5 GB.

## 2. Folder layout

```
odosseus/
├── hackFiles/                                hackathon material: briefs (PDF), data, organizers' tools
│   ├── c0_2026-10-13_22-53-42_camloc/        the training bag
│   │   ├── c0_2026-10-13_22-53-42_camloc_0.mcap
│   │   └── tools/evaluate.py                 the official scorer
│   └── 23-00-38_cut_120m_update/             the organizers' track-frame update
└── localization/                             this pipeline: run every command from here
```

If the bag is somewhere else, set `bag:` and `official_eval:` in `configs/iter1.yaml`.

## 3. Setup

```bash
cd localization
conda env create -f environment.yml          # or: pip install -r requirements.txt  (Python 3.12)
conda activate odosseus
python -m unittest discover -s tests -v      # 7 rule tests: allowed inputs, causality, no GT at inference
```

## 4. Test run: the submission

```bash
./run.sh /path/to/test_run.mcap results/submission.csv
python scripts/check_submission.py results/submission.csv --bag /path/to/test_run.mcap
```

- **First use** builds the derived data from the training bag, in `artifacts/`:
  1. the caches, lap table and per-lap GT (`scripts/prepare.py`);
  2. the car-body masks (`scripts/build_masks.py`);
  3. the map from all 9 laps (`scripts/build_map.py`, about 10 minutes on the M4).

  Later runs reuse them. Delete a folder in `artifacts/` to rebuild it.
- Then it localizes the test run, using only past and present data. A 20-minute run takes about
  12 minutes: about 60 ms per image set, five cameras on the CPU.
- **Output**: `timestamp_ns,x,y,yaw`, the pose of `constructor0/base_link` in `map`, one row per
  image set. Next to the CSV come `<name>_diag.npz` (filter diagnostics per image set) and
  `<name>_runtime.json` (time per image set and the hardware).
- `check_submission.py` checks the header and the stamps (strictly increasing, real image stamps of
  the bag). It also checks that every value is finite, that yaw is within ±π, and the coverage: at
  least 90 % of the run's 1 s bins, needed to be ranked.
- **First pose**: read from the first `map → constructor0/base_link` transform of the test bag. If
  it is given separately, run the localization step directly:
  `python scripts/run.py --config configs/final.yaml --bag /path/to/test_run.mcap --out results/submission.csv --first-pose X Y YAW`
- `run.sh` reads two environment variables: `CONFIG` (default `configs/final.yaml`) and `PYTHON`
  (default `python`).

## 5. Validation on the training bag

All validation numbers come from one continuous run of the whole bag from its first pose, scored
lap by lap. Ground truth is only used for building maps and for scoring.

```bash
python scripts/prepare.py                    # once: caches, lap table, per-lap GT, data checks
python scripts/build_masks.py                # once: car-body masks (images only)
```

**Cross-validation**: three folds, the protocol behind the README's headline numbers. It builds
three maps of about 0.46 GB each and takes about 1 hour on the M4.

```bash
PYTHON=python scripts/run_cv.sh
python scripts/cv_score.py configs/cv_f1.yaml configs/cv_f2.yaml configs/cv_f3.yaml
```

**One split, step by step**: for example split B, the pit-lane stress test.

```bash
B=../hackFiles/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
python scripts/build_map.py --config configs/iter1b.yaml --name map_b
python scripts/run.py --config configs/iter1b.yaml --bag $B --out results/iter1b/fullrun/submission.csv
python scripts/score_run.py results/iter1b/fullrun/submission.csv --config configs/iter1b.yaml
python scripts/plot_lap.py --config configs/iter1b.yaml --run results/iter1b/fullrun/submission.csv --laps L0 L4 L7
```

**Ablations**:

- `scripts/run.py ... --no-cameras` runs on the car-state odometry and the track constraint only.
- `--set key=value` overrides a config value, for example `--set track_constraint.enabled=false`.

**Diagnostics**:

| script | what it does |
|---|---|
| `scripts/validate.py` | lap restarts, as a fast tuning loop on the tuning lap L5 |
| `scripts/reloc_diag.py` | fix rate and accuracy per camera |
| `scripts/plot_coverage.py` | errors against the distance from the mapped line |
| `scripts/match_external.py` | frames of an unknown camera against the map |
| `scripts/lap_table.py` | the lap table |

## 6. Configuration

| file | what it is |
|---|---|
| `configs/iter1.yaml` | every parameter; each block names its decision in `docs/DECISIONS.md` |
| `configs/final.yaml` | the test run: the same method, map from all laps |
| `configs/iter1b.yaml` | split B (pit-lane stress test) |
| `configs/cv_f1.yaml` … `cv_f3.yaml` | the three cross-validation folds |
| `configs/*_swapped.yaml`, `configs/*_untuned.yaml` | the swapped-tuning-lap and untuned-parameter checks (D-022, D-023) |

A config can extend another one (`extends:`). `run.py`, `validate.py` and `build_map.py` accept
`--set a.b.c=value` overrides.

## 7. What is not in the repository

- `artifacts/`: derived data, rebuilt from the training bag by `run.sh` or the scripts above.
- `results/`: outputs of runs (submissions, scores, plots), rebuilt by the commands above. The
  numbers quoted in the README come from these runs.
- The bags (`*.mcap`) and the GT export (`gt.csv`).

## 8. Troubleshooting

- **"WARNING: camera calibration of this run differs from the map's"**: the test bag's camera
  calibration (camera_info or tf_static) differs from the training bag's, which was used to build
  the map. Check the camera setup before trusting the run.
- **Slow runs** (more than about 100 ms per image set): usually memory pressure. Close other programs;
  times grow when the machine swaps.
- **Disk full while building a map**: each map takes 0.4–0.75 GB, written per camera at the end of
  the build.
