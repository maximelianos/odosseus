# Camera-based localization on Yas Marina — iteration 1

Causal estimate of the race car's pose (x, y, yaw of `constructor0/base_link` in `map`) at camera
rate, from the 7 cameras, the track borders, the car's own motion signals and the first pose.

Every design choice, with its evidence and the alternatives that were measured and dropped, is in
[docs/DECISIONS.md](docs/DECISIONS.md) (IDs like D-006 are referenced from code and config).

## How it works

```
car state (vx, vy, yaw rate, ~105 Hz) ──► EKF predict: x, y, yaw, gyro bias            D-006
                                           │  standstill: no turning, bias update
cameras (10 Hz, hardware-synced) ──► per camera, in parallel threads:                   D-009..D-013
    SIFT (CLAHE, half res, car-body mask) → match to 3D points of the nearest keyframes
    of a map built from the training laps with GT poses → keep matches consistent with
    the predicted pose → PnP RANSAC → base_link pose → height / attitude checks
                                           │
                                           ▼
    EKF update per camera fix, gated (chi-square 99.9 %); gate lock-out guard              D-014
track borders ──► stay ≥ 1 m inside the track / pit lane (one-sided update)              D-008
                                           │
                                           ▼
                       pose at the image-set stamp → CSV (timestamp_ns,x,y,yaw)           D-005
```

## Results

Every number below comes from **one continuous run of the training bag**, exactly like the test
run: `scripts/run.py` starts from the bag's first pose and never reads GT again; `scripts/score_run.py`
scores the laps inside it with the official math (D-004). Approaches are compared on **three
cross-validation folds** with both pit laps in every map (D-021); split **B**, which holds out the
out-lap L0 with its 82 s of pit lane, stays as the stress test for an unmapped stretch (D-003, D-020).
L5 is the tuning lap and is in no map.

### Cross-validation (D-021): each of L1–L4, L6, L7 validated once

| fold | validation | map | pos RMSE | lateral | heading | < 1 m | runtime mean |
|---|---|---|---|---|---|---|---|
| F1 | L1 + L6 | L0 L2 L3 L4 L7 L8 | 0.119 m | 0.044 m | 0.079° | 100 % | 61.9 ms |
| F2 | L2 + L7 | L0 L1 L3 L4 L6 L8 | 0.220 m | 0.180 m | 0.215° | 100 % | 58.0 ms |
| F3 | L3 + L4 | L0 L1 L2 L6 L7 L8 | 0.123 m | 0.066 m | 0.114° | 100 % | 62.1 ms |
| **pooled, 6 laps** | | | **0.162 m** | **0.114 m** | **0.148°** | **100 %** | |

Pooled longitudinal 0.114 m, median 0.093 m, p95 0.330 m, max 0.816 m. Per lap: L1 0.127, L2 0.265,
L3 0.106, L4 0.136, L6 0.104, L7 0.132 m.

**L2 drove a different line.** For 60 % of the lap L2 is more than 1 m (up to 11–14 m) from every
other lap, which share one line within ~0.1 m; L5 drives L2's line for s ≈ 1,000–1,665 m. Holding
L2 out (F2) removes the only keyframes of that line: L2's fix rate drops to 66–98 % and its error to
0.2–0.4 m, and L5 in F2 scores 0.204 m instead of 0.12 m. Still 100 % within 1 m, but F2 is the fold
that shows what an unfamiliar line costs (D-021, open issue 7).

### Summary across folds and split B: every error metric of the validation laps

Weights are those of the evaluation scheme. "Map laps" counts training laps only (L5, the tuning
lap, is in no map). Runtime is per image set over the whole run of the fold (M4 CPU). Coverage is
100 % in every run.

| scope | split / fold | map laps | poses | position RMSE (25 %) | lateral RMSE (20 %) | heading RMSE (10 %) | within 1 m (10 %) | longitudinal RMSE | median | p95 | max | runtime mean / p95 (5 %) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | F1 | 6 | 1,802 | 0.127 m | 0.043 m | 0.064° | 100 % | 0.120 m | 0.102 m | 0.232 m | 0.378 m | |
| L6 | F1 | 6 | 1,115 | 0.104 m | 0.045 m | 0.098° | 100 % | 0.094 m | 0.068 m | 0.221 m | 0.357 m | |
| **F1: L1 + L6** | F1 | 6 | 2,917 | **0.119 m** | **0.044 m** | **0.079°** | **100 %** | 0.111 m | 0.087 m | 0.229 m | 0.378 m | 61.9 / 83.5 ms |
| L2 | F2 | 6 | 1,578 | 0.265 m | 0.230 m | 0.262° | 100 % | 0.132 m | 0.184 m | 0.526 m | 0.816 m | |
| L7 | F2 | 6 | 1,131 | 0.132 m | 0.058 m | 0.119° | 100 % | 0.119 m | 0.085 m | 0.262 m | 0.435 m | |
| **F2: L2 + L7** | F2 | 6 | 2,709 | **0.220 m** | **0.180 m** | **0.215°** | **100 %** | 0.127 m | 0.128 m | 0.442 m | 0.816 m | 58.0 / 75.8 ms |
| L3 | F3 | 6 | 1,106 | 0.106 m | 0.039 m | 0.070° | 100 % | 0.099 m | 0.064 m | 0.226 m | 0.329 m | |
| L4 | F3 | 6 | 1,182 | 0.136 m | 0.085 m | 0.144° | 100 % | 0.107 m | 0.090 m | 0.277 m | 0.392 m | |
| **F3: L3 + L4** | F3 | 6 | 2,288 | **0.123 m** | **0.066 m** | **0.114°** | **100 %** | 0.103 m | 0.075 m | 0.255 m | 0.392 m | 62.1 / 85.6 ms |
| **CV pooled, 6 laps** | F1–F3 | 6 | **7,914** | **0.162 m** | **0.114 m** | **0.148°** | **100 %** | **0.114 m** | **0.093 m** | **0.330 m** | **0.816 m** | 58.0–62.1 / 75.8–85.6 ms |
| L0 | split B | 5 | 1,854 | 0.576 m | 0.448 m | 0.330° | 93.2 % | 0.362 m | 0.268 m | 1.180 m | 2.218 m | |
| L4 | split B | 5 | 1,182 | 0.143 m | 0.093 m | 0.144° | 100 % | 0.108 m | 0.098 m | 0.272 m | 0.425 m | |
| L7 | split B | 5 | 1,131 | 0.127 m | 0.056 m | 0.116° | 100 % | 0.115 m | 0.083 m | 0.267 m | 0.437 m | |
| L4 + L7 | split B | 5 | 2,313 | 0.135 m | 0.077 m | 0.131° | 100 % | 0.111 m | 0.089 m | 0.270 m | 0.437 m | |
| **split B: L0 + L4 + L7** | split B | 5 | **4,167** | **0.397 m** | **0.304 m** | **0.241°** | **97.0 %** | 0.255 m | 0.141 m | 0.804 m | 2.218 m | 55.6 / 75.9 ms |
| *L5, tuning lap* | *split B* | *5* | *1,439* | *0.103 m* | *0.075 m* | *0.119°* | *100 %* | *0.071 m* | *0.071 m* | *0.200 m* | *0.376 m* | |

- L4 and L7 are validated in both designs and barely move between a 5-lap and a 6-lap map
  (L4 0.143 vs 0.136 m, L7 0.127 vs 0.132 m).
- The two designs fail differently: split B on an unmapped *stretch* (the pit lane, L0), the folds
  on an unmapped *line* (L2 in F2).
- The tuning lap L5 scores 0.103 m in split B and 0.116–0.123 m in F1/F3 (L2, which covers L5's
  off-line stretch, is in those maps), but 0.204 m in F2 (L2 not in the map).

### Swapped tuning lap (D-022): L2 instead of L5 in F1 and F2

Parameters are not re-tuned (tuning is a separate offline step, D-015; map files hold only keyframes,
3D points and descriptors), so "tuning lap" here means held out of the map and scored for information.

| run | validation | tuning lap | map | pos RMSE | lateral | heading | < 1 m | longitudinal | median | p95 | max | runtime mean / p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | L1 + L6 | L5 | L0 **L2** L3 L4 L7 L8 | 0.119 m | 0.044 m | 0.079° | 100 % | 0.111 m | 0.087 m | 0.229 m | 0.378 m | 61.9 / 83.5 ms |
| **F1_swapped** | L1 + L6 | L2 | L0 L3 L4 **L5** L7 L8 | 0.123 m | 0.046 m | 0.080° | 100 % | 0.114 m | 0.084 m | 0.253 m | 0.455 m | 58.8 / 76.6 ms |
| **F2_swapped** | L5 + L7 | L2 | L0 L1 L3 L4 L6 L8 (= F2) | 0.176 m | 0.137 m | 0.175° | 100 % | 0.111 m | 0.093 m | 0.361 m | 0.900 m | 57.2 / 74.1 ms |
| pooled F1_swapped + F2_swapped + F3 | L1 L3 L4 L5 L6 L7 | | | 0.143 m | 0.091 m | 0.128° | 100 % | 0.110 m | 0.084 m | 0.285 m | 0.900 m | |

Per lap: L1 0.130, L6 0.112, L5 0.204, L7 0.132 m; tuning lap L2 0.252 m (F1_swapped) and 0.265 m
(F2_swapped).

- **F2_swapped is byte-identical to F2**: same map laps, same parameters; only the roles change.
- **One lap on the same line removes the off-line penalty.** On the 16 % of L2 where L5 drove the
  same line, L2's error falls from 0.347 m (85 % fixes, F2) to 0.108 m (100 %, F1_swapped); on the
  43 % no other lap drove it stays at 0.32–0.35 m.
- For common-line laps the choice of the sixth map lap hardly matters (L1 + L6: 0.119 → 0.123 m).
- The pooled 0.143 m is **not** better than D-021's 0.162 m: it trades the hardest validation lap
  (L2) for L5, whose score is optimistic because the parameters were tuned on it.

### What the tuning buys (D-023): the folds with the first-guess values

Same maps, but the values used before tuning on L5 (fix σ 0.3 m / 0.3°, process noise halved).

| | position RMSE | lateral | longitudinal | heading | median | p95 | max | < 1 m |
|---|---|---|---|---|---|---|---|---|
| tuned (D-021), pooled 6 laps | **0.162 m** | **0.114 m** | 0.114 m | 0.148° | 0.093 m | 0.330 m | 0.816 m | 100 % |
| untuned, pooled 6 laps | 0.186 m | 0.128 m | 0.135 m | 0.153° | 0.128 m | 0.342 m | 0.899 m | 100 % |
| **gain from tuning** | **−0.024 m (−13 %)** | −0.014 m (−11 %) | −0.021 m | −0.005° | −0.035 m | −0.012 m | −0.083 m | |

Per fold, tuned → untuned: F1 0.119 → 0.131, F2 0.220 → 0.250, F3 0.123 → 0.155 m. Per lap: L1
0.127 → 0.125, L2 0.265 → 0.299, L3 0.106 → 0.138, L4 0.136 → 0.169, L6 0.104 → 0.141, L7 0.132 →
0.158 m.

- The gain transfers to unseen laps, but is smaller than on the tuning lap itself (−20 % on L5).
- Off the mapped line (L2), tuning improves the longitudinal error (0.220 → 0.145 m) but not the
  lateral one (0.299 → 0.296 m): the off-line lateral error is a bias of the fixes (D-022), which
  noise values cannot remove.
- Both settings are overconfident (actual RMSE 1.6–2.3× the filter's σ tuned, 1.7–2.8× untuned).

### Split B (pit-lane stress test): validation laps L0 + L4 + L7

| method | pos RMSE | lateral | longitudinal | heading | median | p95 | max | < 1 m |
|---|---|---|---|---|---|---|---|---|
| car-state odometry only | 21.8 m | 16.8 m | 13.9 m | 2.65° | 9.39 m | 47.5 m | 54.4 m | 3.4 % |
| + track constraint | 3.43 m | 2.48 m | 2.37 m | 1.00° | 1.90 m | 7.16 m | 9.5 m | 16.2 % |
| **+ camera relocalization (iteration 1)** | **0.397 m** | **0.304 m** | 0.255 m | **0.241°** | 0.141 m | 0.804 m | 2.22 m | **97.0 %** |

| lap | kind | pos RMSE | lateral | heading | < 1 m |
|---|---|---|---|---|---|
| L0 | out-lap, pit lane not in the map | 0.576 m | 0.448 m | 0.330° | 93.2 % |
| L4 | fast, slow last ~40 s | 0.143 m | 0.093 m | 0.144° | 100 % |
| L7 | flying lap, 40 m/s | 0.127 m | 0.056 m | 0.116° | 100 % |
| L5 (tuning) | slow start, fast finish | 0.103 m | 0.075 m | 0.119° | 100 % |

**Per evaluation-scheme metric** (split B, iteration 1):

| metric (weight) | validation L0 + L4 + L7 | L4 + L7 only | tuning L5 |
|---|---|---|---|
| position RMSE (25 %) | 0.397 m | 0.135 m | 0.103 m |
| lateral RMSE (20 %, also the tie-break) | 0.304 m | 0.077 m | 0.075 m |
| heading RMSE (10 %) | 0.241° | 0.131° | 0.119° |
| poses within 1 m (10 %) | 97.0 % | 100 % | 100 % |
| runtime per image set (5 %), whole run | 55.6 ms mean, p95 75.9 ms | | |
| coverage (≥ 90 % to be ranked), whole run | 100 % | | |

**The pit lane without map coverage (L0).** The two pit passes of the bag cover different
stretches of the pit lane (L0 garage → exit, L8 entry → stop), so in split B the 82 s L0 spends in
the pit lane are in no lap of the map. There the car-motion data carries the estimate (error
0.4–1.0 m); camera fixes were found in only 3 frames, from track keyframes seen across the pit wall.
Two were good (1.00 → 0.09 m, 0.48 → 0.25 m), but one wrong fix at 81.4 s moved the error from
0.61 to 1.65 m and shrank the filter's σ to 0.37 m. With the filter that sure of itself, the guided
matching rejected the correct matches for the next 8 s on mapped track: worst pose 2.22 m at 90 s,
then 0.22 m RMSE for the rest of the lap (D-020). In the hidden test run the all-laps map covers both
pit stretches; this is the case where a stretch is not covered. Plot: `results/iter1b/fullrun/L0.png`.

### Split A (first validation): both pit laps in the map

Validation laps L4 + L7: 0.137 m position, 0.077 m lateral, 0.132° heading, 100 % within 1 m;
odometry only 29.1 m, + track constraint 4.43 m. With L0 in the map it scored 0.075 m. Results in
`results/iter1/fullrun/`; L4 + L7 are the same under both splits (0.137 vs 0.135 m).

### Other results

**Runtime**: 55.6–59.7 ms per image set (p95 76–77 ms) on an Apple M4 CPU (10 cores), five cameras
in parallel threads, no GPU; up to 92–95 ms (p95 ~124 ms) when the machine was under memory pressure
(D-018). The 1,194 s bag is processed in about 740 s. Re-running gives a byte-identical CSV.

**Submission path** (`./run.sh` on the training bag with the all-laps map, D-016): map built in
10 min, whole bag localized in 12.5 min, all format checks of `scripts/check_submission.py` pass,
100 % coverage. Its score (0.048 m) is the in-map ceiling, not an estimate for the test run.

**Track (s, n, epsi) frame** of the organizers' update (D-019): position and lateral (`n`) scores
are identical to x/y; heading scores 20–45 % worse as `epsi` than as yaw, so we submit x, y, yaw.

**Robustness probes** on L5 (lap restart, D-017): darker / brighter images change RMSE by ≤ 0.01 m;
18 px motion blur gives 0.18 m; a 30 s camera blackout lets the error grow to 0.96 m and recovers to
< 0.14 m within 1 s of the cameras returning.

**What these numbers do not show.** All laps come from one night session, so validation is
optimistic for a test run in different light; the map is night-only and that is the main risk.
The filter is about 2× overconfident (D-014), which is what turned one wrong fix into an 8 s
lock-out on L0 (D-020). In the continuous run the held-out laps L4/L7 are entered from laps whose
own images are in the map. See "Open issues for iteration 2" in [docs/DECISIONS.md](docs/DECISIONS.md).

## Decisions at a glance

| ID | decision |
|---|---|
| D-001 | One whitelist of inputs; `yaw_vel_rad` is the absolute course angle (ground truth), never read |
| D-002 | Laps from a start/finish gate on the GT path (9 laps, L0 and L8 partial); lap table `results/laps.csv` |
| D-003 | Split B (stress test since D-021): validate on L0 + L4 + L7, tune on L5, map from L1–L3, L6, L8 (split A kept L0 in the map) |
| D-004 | Validation = one continuous run from the bag's first pose, scored per lap; lap restarts only for tuning |
| D-005 | One pose per image set, stamped with the newest image; car state only up to that stamp |
| D-006 | EKF on car-state odometry with a gyro-bias state; no visual odometry |
| D-007 | Initial uncertainty from the GT accuracy of the first pose; bias uncertainty 3e-4 rad/s |
| D-008 | One-sided track constraint at 1 m (GT is never closer than 1.51 m to a border) |
| D-009 | Camera method: relocalization against a 3D map of the training laps |
| D-010 | SIFT on CLAHE-equalized half-resolution images |
| D-011 | Masks for features that move with the car (same descriptor at the same pixel) |
| D-012 | Map: keyframe every 2 m, triangulation with GT poses, parallax ≥ 0.5°; bag extrinsics kept |
| D-013 | Guided matching with the predicted pose, PnP RANSAC, sanity checks, five cameras |
| D-014 | One gated EKF update per camera fix; gate lock-out guard |
| D-015 | Measurement and process noise tuned on L5 only |
| D-016 | Hidden test run: same method, map from all laps (`configs/final.yaml`, `run.sh`) |
| D-017 | Robustness probes (lighting, blur, blackout) as checks, not tuning |
| D-018 | Runtime = per image set from decoding to pose, on the M4 CPU |
| D-019 | Submit x, y, yaw (option A); the s/n/epsi frame of the update scores heading worse |
| D-020 | Finding: in an unmapped pit stretch a wrong fix plus an overconfident filter locks out fixes for 8 s |
| D-021 | Cross-validation: 3 folds, L0 + L8 always in the map, L5 tuning; finding: L2 drove a different line |
| D-022 | Swapped tuning lap (L2 for L5 in F1/F2): one lap on the same line removes the off-line penalty |
| D-023 | Tuning buys −0.024 m (−13 %) pooled; it does not reduce the off-line lateral error |
| D-024 | Sanity-check tool: frames of an unknown camera (e.g. daytime onboard screenshots) against the map |

## Setup

Step-by-step instructions for setup, the test run and validation: [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md).

```bash
conda env create -f environment.yml      # or: conda create -n odosseus python=3.12 && pip install -r requirements.txt
conda activate odosseus
```

The training bag is expected at `../hackFiles/c0_2026-10-13_22-53-42_camloc/` (set `bag:` in
`configs/iter1.yaml` otherwise). No GPU and no pretrained models are used.

## Running

Hidden test run (one command; builds the artifacts from the training bag on first use, about
10 minutes for the all-laps map, then ~12 minutes for a 20-minute run on the M4):

```bash
./run.sh /path/to/test_run.mcap results/submission.csv
python scripts/check_submission.py results/submission.csv --bag /path/to/test_run.mcap   # format and coverage
```

The first pose is read from the first `map -> constructor0/base_link` transform of the test bag; pass
`--first-pose X Y YAW` to `scripts/run.py` if it is provided differently.

Validation (the protocol of D-004, split B):

```bash
B=../hackFiles/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
python scripts/prepare.py                 # caches, lap table, per-lap GT, data checks
python scripts/build_masks.py             # car-body masks (images only)
python scripts/build_map.py --config configs/iter1b.yaml --name map_b                 # map of split B (uses GT)
python scripts/run.py --config configs/iter1b.yaml --bag $B --out results/iter1b/fullrun/submission.csv
python scripts/score_run.py results/iter1b/fullrun/submission.csv --config configs/iter1b.yaml   # per lap + scheme
python scripts/plot_lap.py --config configs/iter1b.yaml --run results/iter1b/fullrun/submission.csv --laps L0 L4 L7
python scripts/run.py --config configs/iter1b.yaml --bag $B --out <csv> --no-cameras [--set track_constraint.enabled=false]  # ablations
PYTHON=python scripts/run_cv.sh          # D-021: 3 folds (map + continuous run + score each), ~1 h
python scripts/cv_score.py configs/cv_f1.yaml configs/cv_f2.yaml configs/cv_f3.yaml   # pool the folds
CV_OUT=results/cv/cv_score_swapped_only PYTHON=python scripts/run_cv.sh configs/cv_f1_swapped.yaml configs/cv_f2_swapped.yaml   # D-022
python scripts/match_external.py --images external_frames/day --ignore-bottom 0.3 --out results/external/day   # D-024
python -m unittest discover -s tests -v   # rules: allowed fields, causality, no GT on the inference path
```

Fast tuning loop (lap restart from GT, development only; D-004):
`python scripts/validate.py --laps L5 --tag tune --reloc`. Diagnostics used for the decisions:
`scripts/reloc_diag.py` (per-camera fix rate and accuracy), `scripts/calibrate_extrinsics.py`
(camera rotation refinement, rejected in D-012), `scripts/lap_table.py` (lap table).

## Layout

`artifacts/` and `results/` are not in git: they are rebuilt from the training bag by `run.sh` and
the commands in [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md); the numbers quoted above come from them.

| path | content |
|---|---|
| `camloc/allowed.py` | the whitelist of topics and car-state fields the method may read |
| `camloc/bag.py` | mcap reading through the whitelist; image sets grouped by stamp |
| `camloc/gt.py` | ground truth (map building, laps, scoring only — never imported at inference) |
| `camloc/odometry.py`, `camloc/ekf.py` | causal car-state stream and the pose filter |
| `camloc/trackmap.py`, `camloc/laps.py` | track corridors, inside-track test, lap gate |
| `camloc/reloc/` | features, map building, per-image relocalization |
| `camloc/pipeline.py` | the causal per-image-set loop |
| `camloc/metrics.py` | the official score as numbers |
| `configs/iter1.yaml` | all parameters of iteration 1, split A |
| `configs/iter1b.yaml` | split B: same parameters, L0 held out (pit-lane stress test) |
| `configs/cv_f1.yaml`–`cv_f3.yaml` | the cross-validation folds (D-021); `cv_f1_swapped`, `cv_f2_swapped` (D-022); `cv_f*_untuned` (D-023) |
| `configs/final.yaml` | the hidden test run: map from all laps |
| `scripts/` | entry points (see above) |
| `results/laps.csv` | every lap: ns / UTC / local timestamps, kind, split role (B), image sets, speed metrics |
| `results/cv/` | cross-validation: per-fold runs, scores and plots, pooled `cv_score.json` |
| `results/iter1b/` | split B: continuous runs (with cameras, odometry-only ablations), scores, plots |
| `results/iter1/` | split A: continuous runs, lap-restart tuning runs, robustness probes, s/n analysis, diagnostics |
| `artifacts/` | derived data: caches, masks, maps (rebuilt by `run.sh`) |
| `RUN_INSTRUCTIONS.md` | setup, the test run, validation |
