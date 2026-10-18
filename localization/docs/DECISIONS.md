# Decision log — iteration 1

Every design choice of iteration 1, with the evidence it rests on and what was rejected. Numbers come
from the training bag `c0_2026-10-13_22-53-42_camloc_0.mcap` and the scripts named in each entry; the
raw outputs are in `results/iter1/`. IDs are referenced from the code and `configs/iter1.yaml`.

Status legend: **adopted**, **rejected** (tried, measured, dropped), **deferred** (next iteration).

Two kinds of run appear below (D-004):
- **Continuous run** — the whole bag from its first pose, exactly like the test run. All reported
  validation numbers come from it.
- **Lap restart** — one lap started from the GT pose at its first frame. Not allowed in a ranked
  run; used only as a fast development loop on the tuning lap L5 and for diagnostics.

---

## D-001 Inputs: one whitelist, and `yaw_vel_rad` is ground truth — adopted

**Context.** The brief and INTRO slide 9 allow camera images, calibration, `/tf_static`, track
borders, the first pose, and car-state *motion* fields from `/constructor0/state_estimation`
("speed, accelerations and yaw rate ... and others"). Pose fields, s/n and `/tf` are ground truth;
using them at inference disqualifies. (The README is stricter and calls the whole state topic ground
truth; the newer brief and slides allow the motion fields — to confirm at the Q&A.)

**Decision.** `camloc/allowed.py` is the single list of readable topics and state fields:
`vx_mps, vy_mps, v_mps, wz_radps, ax_mps2, ay_mps2`. `camloc/bag.py` reads the state only through it.
Ground truth lives in `camloc/gt.py`, which the inference path never imports
(`tests/test_rules.py` runs the entry point's imports in a fresh interpreter and checks).

**Evidence.**
- `yaw_vel_rad` equals `yaw_rad + beta_rad` exactly (largest difference over all 126,574 samples:
  9e-16 rad, floating-point rounding): it is the absolute course angle, not a yaw rate. It is listed
  as forbidden. The yaw rate is `wz_radps`.
- `vx_mps`/`vy_mps` are body-frame: `|vy|` never exceeds 0.93 m/s while moving at up to 40 m/s
  (`scripts/prepare.py`, `artifacts/prepare_report.json`). A map-frame velocity would carry the heading.
- `kappa_radpm` equals `wz / v` exactly (safe, but redundant; not used).

---

## D-002 Laps are defined by a start/finish gate on the GT path — adopted

**Decision.** Gate = segment across the track (left to right border) through (-96.6, 697.6), the
point where the car's reference-line coordinate s wraps to 0. A lap starts at each forward crossing
of the GT trajectory (`camloc/laps.py`). This yields 9 laps; L0 and L8 are the partial out-/in-laps.

| lap | start s | dur s | dist m | vmax m/s | vmean | pit lane s | role, split B (A) |
|---|---|---|---|---|---|---|---|
| L0 | 0.0 | 185.5 | 1649 | 14.8 | 8.9 | 80 | **val** (train) |
| L1 | 185.5 | 180.1 | 2979 | 30.1 | 16.5 | 0 | train |
| L2 | 365.6 | 157.8 | 3037 | 28.3 | 19.2 | 0 | train |
| L3 | 523.4 | 110.6 | 2980 | 35.5 | 26.8 | 0 | train |
| L4 | 634.0 | 118.2 | 2980 | 35.4 | 25.1 | 0 | **val** |
| L5 | 752.2 | 143.9 | 2982 | 35.4 | 20.7 | 0 | **tune** |
| L6 | 896.1 | 111.6 | 2981 | 40.3 | 26.6 | 0 | train |
| L7 | 1007.7 | 113.0 | 2978 | 40.3 | 26.2 | 0 | **val** |
| L8 | 1120.7 | 73.7 | 1257 | 30.3 | 17.0 | 28 | train |

The full table with ns / UTC / local timestamps, image-set counts, speed metrics and a rule-based lap
kind is `results/laps.csv` (`scripts/lap_table.py`). Kinds: L0 out-lap, L1–L2 warm-up, L3/L6/L7
flying (≥ 80 % of the distance at ≥ 85 % of L3's speed at the same point), L4/L5 partial (L4 slow
for its last ~40 s, L5 slow in its first ~50 s), L8 in-lap. The car comes to a brief standstill
near the hairpin ~30 s into L2, L6 and L7 (GT speed confirms it: 4.0 s, 0.9 s, 0.3 s); these are
real stops, not speed-signal dropouts.

---

## D-003 Split: validate on L0 + L4 + L7, tune on L5, map from the rest — adopted (revised; comparisons moved to the folds of D-021)

**Decision (split B, `configs/iter1b.yaml`, current).** `train` (map building) = L1, L2, L3, L6, L8;
`tune` = L5; `val` = L0, L4, L7. Validation laps are scored once per split, inside the continuous run
(D-004), and never used to choose a parameter. For the hidden test run the map is rebuilt from all
laps (D-016).

**Why these laps.**
- The test "may include higher speeds": L4 and L7 are fast laps (35.4 and 40.3 m/s; L7 a flying
  lap, L4 fast with a slow last ~40 s, `results/laps.csv`).
- L6 (also 40.3 m/s) stays in training so the map contains the top-speed racing line.
- **One pit lap is held out (the change from split A).** The two pit passes cover different
  stretches of the pit lane, with no overlap: L0 drives from the garage (s ≈ 338 m along the pit
  lane) through the exit (s ≈ 995 m, including the tunnel under the track), 82 s; L8 drives from the
  entry (s ≈ 48 m) to its stop (s ≈ 254 m), 28 s. So whichever is held out, its pit stretch is in no
  lap of the map: validation there measures what the system does without map coverage — a
  pessimistic case, since the test-run map contains both stretches (D-016).
- L0 rather than L8 is held out: it is the situation every run starts in (first pose at standstill
  in the garage), it has three times the pit time, and it includes the pit-exit tunnel.
- A separate tuning lap keeps the validation numbers free of tuning bias (INTRO step 3:
  "never train or tune on them"). The parameters were tuned under split A and are not re-tuned.
- The validation laps are spread over the run (0, 634 and 1008 s), interleaved with training laps.

**Split A (`configs/iter1.yaml`, the first validation).** `train` = L0, L1, L2, L3, L6, L8;
`tune` = L5; `val` = L4, L7. It kept both pit laps in the map, so the pit lane was never validated.
Its results stay in `results/iter1/`.

**Limitation.** All laps are from one night session. Validation is therefore optimistic for a test
run with different lighting; laps on the same track are not independent samples.

---

## D-004 Validation protocol: one continuous run from the first pose — adopted (revised)

**Decision.** Validation numbers come from one continuous run of the whole bag, exactly as for the
test run: `scripts/run.py` starts from the bag's first pose (the only GT pose the rules allow),
never reads GT again, and `scripts/score_run.py` scores the laps inside it per lap and per weighted
metric of the evaluation scheme, with the same math as the official `evaluate.py`
(`camloc/metrics.py`; the official script is also run on the whole run and agrees exactly). The
validation laps (L0, L4, L7 in split B; L4, L7 in split A) and the tuning lap L5 are not in the map.

**Development loop: lap restart.** `scripts/validate.py` runs one lap from the GT pose at its first
image set (2 minutes instead of 13 for the whole bag). Restarting from a GT pose in the middle of a
run is not allowed in a ranked run, so these runs only compare settings on the tuning lap L5 and
serve diagnostics (D-013, D-015, D-017). They never produce the reported validation numbers.

**History.** Iteration 1 first reported lap-restart numbers for L4 / L7 (0.146 / 0.133 m). After
review they were replaced by the continuous run, because only it follows the one-first-pose rule.

**Evidence that the restart does not change the conclusions for the camera method.**

| | L4 | L7 |
|---|---|---|
| lap restart from GT | 0.146 m | 0.133 m |
| continuous run | 0.141 m | 0.133 m |
| first 2 s of the lap: restart / continuous | 0.236 / 0.248 m | 0.223 / 0.234 m |
| 10 s to the end: restart / continuous | 0.143 / 0.137 m | 0.131 / 0.131 m |

The camera fixes re-anchor the estimate in almost every frame, so a GT start helps for well under a
second. The two runs never become identical (gyro-bias estimate, covariance and gate decisions
differ; the estimates stay up to 31 cm (L4) / 12 cm (L7) apart at times), but their RMSE is the same.
The no-camera baselines are different: without fixes the drift accumulates from the start of the
bag (odometry only, L4 + L7: 29.1 m continuous vs 5.0 m with restarts), so the ablation is also
reported from continuous runs (`results/iter1/fullrun_odom*`).

**Caveat.** In the continuous run the held-out laps are entered from laps whose own images are in the
map; in the test run no lap is in the map. Because the estimate re-anchors within a second, this
warm entry matters little.

**Why per-lap GT files** (used by lap restarts and plots). `evaluate.py` measures coverage over the
whole span of the GT file, so a one-lap submission scored against the full `gt.csv` would show
~10 % coverage.

---

## D-005 Output timing and causality — adopted

**Decision.** One pose per image set. Images of different cameras with header stamps within 2 ms form
a set; the pose stamp is the largest image stamp in the set. Car state is integrated only up to that
stamp and the last sample is extrapolated to it (`camloc/odometry.py`). Every pose is written before
the next set is read; nothing is smoothed.

**Evidence.** The cameras are hardware-triggered: the 7 stamps of a set differ by ≤ 0.04 ms
(`tools/camera_sync.py` in the repo root). `OdometryStream` records the largest stamp it touched, the
pipeline asserts it never exceeds the pose stamp, and `tests/test_rules.py` checks it on synthetic
data.

---

## D-006 Motion model: EKF on car-state odometry, no visual odometry — adopted

**Decision.** State `[x, y, yaw, b]` with `b` the yaw-rate bias. Prediction at ~105 Hz integrates the
body velocities `vx, vy` and `wz - b` (midpoint rule). At standstill (`v < 0.05 m/s`) the car cannot
turn: yaw is frozen and `wz` updates `b` directly. `b` starts from the mean yaw rate over the
standstill at the start of the run.

**Evidence.** Dead reckoning with these signals from a known pose drifts 0.5 m after 5 s, 1 m after
10 s, 2.8 m after 30 s; speed is accurate to 0.03 % of distance. Almost all drift is heading: the
bias moves from -1.5e-4 to -3.3e-4 rad/s over the run (standstill value -0.7e-4), giving 16.5° yaw
error and 65 m RMSE over the full run without corrections. So the cameras must correct heading and
position every few seconds, but do not need to provide the motion itself. Visual odometry would add
work without beating this prior.

**Noise values** (spectral densities) are in `configs/iter1.yaml`; see D-015 for tuning.

---

## D-007 Initial uncertainty — adopted

`sigma0_xy = 0.1 m`, `sigma0_yaw = 0.2°` (the first pose is GT, ~0.1 m / 0.2°). `sigma0_bias =
3e-4 rad/s`, because the standstill bias differs from the in-motion bias (D-006).

---

## D-008 Track constraint: stay at least 1 m inside the drivable area — adopted

**Decision.** If the estimate is less than 1.0 m inside the track or pit-lane corridor, a one-sided
position update (σ = 0.3 m) along the inward border normal pushes it back.

**Evidence.**
- The GT base_link is never closer than 1.51 m to a border (0.1th percentile 1.51 m, never outside;
  `prepare_report.json`), so the constraint is never active on the true trajectory.
- Continuous runs without cameras: position RMSE on L4 + L7 29.1 m → 4.4 m with the constraint
  (lateral 22.4 → 3.2 m); on split B's validation laps L0 + L4 + L7 21.8 → 3.4 m; whole bag
  24.8 → 4.9 m. (First measured with a lap restart on L5: 7.47 → 3.50 m.)
- The track-surface height must come from the corridor the car is in: near the pit exit (which runs
  under the track) the track borders are up to 4.6 m above the pit lane.

---

## D-009 Camera method for iteration 1: relocalization against a map of the training laps — adopted

**Decision.** For each camera, a map of 3D points triangulated from training-lap images with GT
camera poses (INTRO method 3a). At run time each image is matched to the nearest map keyframes and
PnP gives an absolute 6-DoF camera pose, converted to base_link `(x, y, yaw)` and fused in the EKF.

**Why first.** It gives absolute fixes (no drift) of lateral, longitudinal and heading at once, and the
car-state prior (D-006) makes the search local, so no global place recognition is needed. It is
classical (OpenCV SIFT + PnP): no GPU, no downloads, no training.

**Deferred.** Learned features (SuperPoint/LightGlue), the lighting-robust track-relative network
(lateral offset and heading from the borders), and global retrieval for recovery are planned for
iteration 2.

---

## D-010 Features: SIFT on CLAHE-equalized half-resolution grayscale — adopted

**Decision.** Decode JPEG at half resolution (libjpeg DCT scaling), CLAHE (clip 2.0, 8×8 tiles), SIFT
with `contrastThreshold = 0.02`, at most 2000 features.

**Evidence** (one image set at t = 700 s): OpenCV-default SIFT finds 36–220 keypoints on the night
images; CLAHE + 0.02 finds 427–1143 (5–17× more) at 12–17 ms per image on the M4 CPU. Descriptors are
integers ≤ 255, so they are stored as uint8 without loss.

---

## D-011 Masks for features that move with the car — adopted

**Decision.** Mask image regions where the *same* feature (ratio-test descriptor match) appears at the
*same* pixel (< 3 px) in frames taken seconds apart at > 20 m/s. Built from 200 frames per camera, no
GT (`scripts/build_masks.py`). Result: `cl` 6.9 %, `cr` 8.6 %, `rr` 3.2 %, `fr` 0.8 % (the car's own
nose decal), `fl`, `r`, `rl` ≈ 0 %.

**Rejected.**
- Temporal intensity variance: at night the body's brightness changes with the floodlights as much
  as the background (temporal std 6–36 grey levels in both), so nothing separates.
- Keypoint-location frequency: keypoints are always dense along the horizon, so it masked the
  horizon band — the most useful region — in every camera (`results/iter1/masks_preview.jpg` shows
  the final masks).

---

## D-012 Map construction — adopted

**Decision.** Per camera, a keyframe every 2 m of GT travel inside the training laps (every frame
above 20 m/s). Each keyframe's features are matched (ratio 0.8) to up to 3 later keyframes of the same
pass (baseline 1–15 m), filtered by the epipolar constraint from the GT relative pose (Sampson
distance < 2 px), triangulated, and kept if the reprojection error is < 2 px in both views, depth is
1–200 m and the parallax is ≥ 0.5°. The largest-parallax triangulation per feature is kept, at most
800 points per keyframe, stored in the map frame (`camloc/reloc/mapbuild.py`).

**Evidence** (diagnostic map from L3 + L6, 4 m spacing, queried on L5 with a 1 m / 0.5° noisy prior):

| parallax ≥ | fl pts/kf | fl fix % | fr fix % | cr fix % | r fix % | median error |
|---|---|---|---|---|---|---|
| 1.0° | 15 | 27 | 58 | 76 | 26 | 0.20–0.35 m |
| 0.5° | 33 | 51 | 69 | 82 | 34 | 0.20–0.34 m |

The lower threshold keeps distant points, which the narrow `fl` and the rear `r` camera need; accuracy
did not change.

**Rejected: refining the camera rotations.** The README warns that extrinsic yaw is reliable only for
`r`/`rl` and roll is uncalibrated. `scripts/calibrate_extrinsics.py` fits a rotation correction per
camera that minimizes the epipolar error of 5-point-RANSAC inliers between consecutive frames, given
the GT motion. With the bag extrinsics the median Sampson error is already 0.8–2.2 px, and the fit
lowers it by at most 0.25 px while proposing corrections of 1–5° that the data barely constrain
(forward motion over 0.1 s hardly depends on the rotation). Because map and query use the same
calibration, rotation errors largely cancel. Kept the bag extrinsics; the refined file is in
`artifacts/` and the report in `results/iter1/extrinsics_refinement.json`.

---

## D-013 Query: guided matching, sanity checks, five cameras — adopted

**Decision** (`camloc/reloc/query.py`, `configs/iter1.yaml: reloc.query/guided/checks`).
1. Candidates: the 3 nearest keyframes of the same camera within 8 m + 3σ (≤ 40 m) of the predicted
   position, heading within 30°, ≥ 30 points.
2. SIFT ratio test (0.8) per keyframe; the best match per query feature is kept.
3. **Guided filter**: a match is kept only if its 3D point is seen within 3σ of where the predicted
   pose expects it. σ combines the filter's position uncertainty scaled by 1/depth, its heading
   uncertainty, 0.8° for the unestimated roll/pitch, 0.2 m for the height, and 6 px.
4. PnP: AP3P inside RANSAC (4 px, ≤ 1000 iterations), LM refinement on the inliers, coordinates
   centred for conditioning; camera pose → base_link pose through the extrinsics.
5. Checks: ≥ 15 inliers, inlier ratio ≥ 0.2, base_link within 1 m of the track surface (offset
   -0.05 m), roll within 3° of the typical -1.25°, |pitch| < 4°.
6. Cameras `cr, cl, rr, fr, rl`, processed in parallel threads.

**Evidence.**
- Without the guided filter only 2–10 % of the 230–500 tentative matches per image were correct and
  RANSAC rarely found them: fix rate 1–13 % per camera (diagnostic map, GT prior). With it: 26–76 %, unchanged when the
  prior is perturbed by 1 m / 0.5° (`scripts/reloc_diag.py`).
- In that diagnostic, fixes rejected by the height check had median errors of 1.0–4.0 m, against
  0.15–0.31 m for accepted fixes: the check removes wrong solutions, not good ones.
- Camera set, tuning lap L5 (lap restart, full map, first-guess filter values):

  | cameras | pos RMSE | lateral | yaw | ms / image set (M4 CPU) |
  |---|---|---|---|---|
  | cr cl rr fr rl fl | 0.128 m | 0.094 m | 0.111° | 106 |
  | **cr cl rr fr rl** | **0.133 m** | **0.100 m** | **0.114°** | **96** |
  | cr cl rr fr | 0.150 m | 0.114 m | 0.134° | 79 |
  | cr cl rr | 0.167 m | 0.132 m | 0.136° | 69 |

  Five cameras keep almost all of the six-camera accuracy inside the 100 ms frame period. `r` was
  not mapped: on the diagnostic map it had the lowest fix rate (34 %) and the worst tail (19 % of
  fixes > 1 m). The side-looking cameras do best, against the expectation that their motion blur
  hurts most: forward motion gives them the best triangulation geometry.

---

## D-014 Fusion: one gated EKF update per camera fix — adopted

**Decision.** Each accepted camera fix is an EKF update of `(x, y, yaw)` with
σ = value · sqrt(100 / inliers), clipped (D-015 for the values), gated at χ²(3) 99.9 % (NIS 16.27).
If fixes keep arriving but all are rejected for 3 s (≥ 6 fixes), the position and heading
uncertainty are inflated by 2 m / 1° so the gate can reopen (gate lock-out guard).

**Evidence.** Median fix error falls with the inlier count (< 30 inliers: 0.30–0.49 m, 30–60:
0.17–0.28 m, > 60: 0.12–0.33 m), and per-camera biases are small (median lateral and longitudinal
error within ±0.08 m, yaw within ±0.03°), so a variance that shrinks with inliers and no bias term
are adequate for now.

**Known simplification.** Fixes are isotropic and treated as independent across cameras, although
they share map errors. The filter is therefore about 2× overconfident (reported σ_xy ≈ 7 cm,
actual RMSE 14 cm). Iteration 2: covariance from the PnP Jacobian and a consistency (NEES) check.

---

## D-015 Tuning on L5 only — adopted

All runs: five cameras, full map, lap restart on L5 (`results/iter1/tune_*`). In the continuous
run the chosen settings give L5 0.108 m, the same as the restart (0.107 m).

| change | pos RMSE | lateral | longitudinal | yaw | fixes accepted / rejected |
|---|---|---|---|---|---|
| σ_xy 0.5 m, σ_yaw 0.5° | 0.170 | 0.118 | 0.122 | 0.114° | 6720 / 11 |
| σ_xy 0.3 m, σ_yaw 0.3° (first guess) | 0.133 | 0.100 | 0.087 | 0.114° | 6673 / 55 |
| σ_xy 0.2 m, σ_yaw 0.2° | 0.117 | 0.087 | 0.078 | 0.116° | 6591 / 146 |
| σ_xy 0.15 m, σ_yaw 0.2° | 0.112 | 0.083 | 0.075 | 0.113° | 6501 / 238 |
| σ_xy 0.10 m, σ_yaw 0.2° | 0.111 | 0.081 | 0.075 | 0.109° | 6231 / 492 |
| **σ_xy 0.15 m + process noise ×2** | **0.107** | **0.072** | 0.079 | 0.121° | 6496 / 233 |
| σ_xy 0.15 m + gate 99 % | 0.111 | 0.082 | 0.075 | 0.107° | 6371 / 361 |

**Decision.** σ_xy = 0.15 m (floor 0.08 m), σ_yaw = 0.2°, process noise doubled
(`v_long = v_lat = 0.1 m/s/√Hz`, `wz = 0.003 rad/s/√Hz`, `bias_rw = 2e-5 rad/s/√s`), gate kept at
99.9 %. Below 0.15 m the gain stops while rejections double, a sign of overconfidence; the larger
process noise helps lateral error most (the most important metric) and makes the filter trust fixes
more when they are sparse. The stricter gate gains nothing and risks lock-out with noisier fixes.
The differences between the last rows are small for one lap; the choice leans to robustness.

---

## D-016 Configuration for the hidden test run — adopted

`configs/final.yaml` = `iter1.yaml` with the map built from **all** laps (L0–L8): validation is
done, and more passes give closer keyframes. `run.sh <test.mcap>` builds the artifacts from the
training bag on first use and runs `scripts/run.py`, which reads the first pose from the first
`map -> constructor0/base_link` transform of the test bag (`--first-pose X Y YAW` overrides it),
uses the test bag's track borders and car state, and warns if its camera calibration differs from
the one the map was built with.

**Check of the submission path.** `./run.sh` on the training bag, exactly as for the test run
(caches and masks reused, `map_final` built from scratch):
- `map_final`: 8,196 keyframes per camera (+55 % over the iteration-1 map), 5.2 M points, 736 MB,
  built in 608 s; the whole command took 22.5 min on the M4.
- `scripts/check_submission.py`: header, one row per image set, strictly increasing stamps that are
  all image header stamps, finite values, yaw in [-π, π], 100 % coverage — all pass.
- Score 0.048 m position RMSE (every lap 0.03–0.08 m). This is **not** an accuracy estimate: every
  lap is in this map, so it is the in-map ceiling. The test-run estimate remains the validation
  result of D-003/D-004 (0.140 m on L4 + L7).

---

## D-017 Robustness probes — adopted as checks, not as tuning

Synthetic changes applied to the images of the tuning lap L5 (lap restart; `reloc.stress`, off in
real runs):

| probe | pos RMSE | lateral | yaw | < 1 m | sets with a fix |
|---|---|---|---|---|---|
| none | 0.107 m | 0.072 m | 0.121° | 100 % | 100 % |
| darker (gamma 2.2) | 0.115 m | 0.083 m | 0.140° | 100 % | 100 % |
| brighter (gamma 0.45) | 0.111 m | 0.079 m | 0.117° | 100 % | 100 % |
| motion blur 18 px (full res) | 0.181 m | 0.139 m | 0.117° | 100 % | 89 % |
| no camera fixes for 30 s | 0.262 m | 0.164 m | 0.158° | 100 % | 79 % |

- CLAHE makes the features insensitive to global brightness. This does **not** show robustness to a
  daylight test run: shadows, sun glare and unlit floodlights change the scene itself, and the map
  was built at night. That is the main open risk.
- During the 30 s blackout the error grows to at most 0.96 m (odometry with the bias learned before
  the blackout, plus the track constraint) and is back under 0.14 m one second after the cameras
  return. Plain dead reckoning drifts 2.8 m (median) in 30 s.

---

## D-018 Runtime measurement — adopted

Time per image set = wall time from the moment the set is available to its pose: JPEG decoding,
features, matching, PnP for five cameras in parallel threads, and the filter. Reading and
decompressing the bag is excluded, because on the car the images arrive as messages;
`scripts/run.py` also reports the total wall time including it. Hardware: Apple M4 (10 cores),
CPU only, no GPU.

| measurement | mean | median | p95 | machine state |
|---|---|---|---|---|
| validation laps L4 + L7 (first run, lap restarts) | 95 ms | 93 ms | 124 ms | ~12 GB swap in use |
| L7 re-timed (lap restart) | 64 ms | 62 ms | 82 ms | ~3 GB swap in use |
| whole bag, `run.sh` with `map_final` | 60 ms | — | 77 ms | ~3 GB swap in use |
| whole bag, split A continuous run (regenerated) | 60 ms | 59 ms | 77 ms | ~3 GB swap in use |
| whole bag, split B continuous run | 56 ms | 57 ms | 76 ms | ~3 GB swap in use |

The whole 1,194 s bag is processed in 690–740 s including bag reading.

The difference is the machine's memory pressure, not the method: the re-timed L7 run produced a
byte-identical CSV (same SHA-256), and frames whose own image is in the map were only ~4 ms faster.
Report the unloaded numbers with the hardware, and note the loaded ones.

**Reproducibility.** Re-running a lap gives a byte-identical CSV: the per-camera threads are
independent and their fixes are fused in a fixed camera order, and OpenCV's RANSAC is seeded.

---

## D-019 Submission format: x, y, yaw in `map` (option A) — adopted

**Context.** The organizers' update (`hackFiles/23-00-38_cut_120m_update/`) adds the track frame of option B:
`s` (along the racing line), `n` (across it, left positive) and `epsi` = car yaw − heading of the
line segment the car projects onto. `convert.py` maps x, y, yaw ↔ s, n, epsi; epsi needs the yaw
(the car's own `epsi` field is always 0). The scheme scores heading only for option A ("Yaw RMSE,
option A"); the updated `evaluate.py` reports an epsi RMSE for option B without saying whether it
counts.

**Decision.** Submit x, y, yaw. The update's racing line can still be added as extra columns later.

**Evidence** (split A continuous run, scored with the update's `evaluate.py`; GT projected on the
update line, `results/iter1/sn_update/`):

| | L4 | L7 | whole run |
|---|---|---|---|
| position RMSE, x/y vs s/n | 0.141 / 0.141 m | 0.133 / 0.133 m | 0.082 / 0.082 m |
| lateral vs n RMSE | 0.095 / 0.094 m | 0.054 / 0.054 m | 0.051 / 0.051 m |
| yaw vs epsi RMSE | 0.145° / 0.173° | 0.116° / 0.168° | 0.077° / 0.120° |

Position scores the same in both frames. Heading scores 20–45 % worse as epsi, because the line's
heading is constant along each ~1 m segment and jumps at vertices (up to 2.4° in the tightest
corner): whenever estimate and GT project onto different segments (6.3 % of poses), epsi inherits
that jump even with a perfect yaw.

**Also found.** The update's racing line (2,960 vertices, 2,976.9 m) is not the line the car used in
the training night (2,929 vertices; the car's own n differs by 0.6 m median, up to 1.4 m), so the
training bag's own (s, n) must not be combined with this map — project x, y instead. The racing line
is the planned reference loop, not the driven path: the car's own n reached 11–15 m on L2, L5, L8.

---

## D-020 Pit lane without map coverage (split B, L0) — finding, fix deferred

**Observation** (continuous run of split B, `results/iter1b/fullrun/`, plot `L0.png`). The 82 s that
L0 spends in the pit lane are in no lap of the map (D-003). L0 scores 0.576 m position RMSE, 0.448 m
lateral, 93.2 % within 1 m, worst pose 2.22 m; the other validation laps are unaffected
(L4 0.143 m, L7 0.127 m).

| time in L0 | what happened | error |
|---|---|---|
| 0–82 s, pit lane | fixes in only 3 frames (matched against track keyframes seen across the pit wall); the car-motion data carries the estimate | 0.4–1.0 m |
| 14.1 s, 25.2 s | two good fixes | 1.00 → 0.09 m, 0.48 → 0.25 m |
| 81.4 s | a wrong fix passes the gate (filter σ 1.53 m, so the gate is wide) and cuts σ to 0.37 m | 0.61 → 1.65 m |
| 82–90 s, mapped track | no fixes produced: guided matching sizes its windows for σ ≈ 0.4 m, the correct matches lie outside; the lock-out guard (D-014) does not fire, it only counts rejected fixes | 1.5 → 2.22 m |
| from ~90 s | σ has grown back, fixes resume | 0.22 m RMSE for the rest of the lap |

**Not fixed in this iteration.** L0 is a validation lap; tuning a remedy on it would leak validation
into the method. Remedies for iteration 2, to be tuned on other data (e.g. a synthetic camera
blackout on L5, D-017):
1. With a weak prior (large σ), accept a fix only if a second camera agrees in the same frame.
2. Extend the lock-out guard to "no fixes produced for N s while keyframes are in range".
3. Covariance that reflects the real error (D-014): the filter claimed 0.37 m while 1.65 m off.

---

## D-021 Cross-validation: three folds with both pit laps in every map — adopted

**Decision** (`configs/cv_f1.yaml`–`cv_f3.yaml`, `scripts/run_cv.sh`, `scripts/cv_score.py`).
Approaches of iteration 2 are compared on three folds. L0 and L8 (both pit passes) are in every
map, L5 stays the tuning lap and is in no map, and each of L1–L4, L6, L7 is validated in exactly
one fold:

| fold | validation | map | why this pair |
|---|---|---|---|
| F1 | L1 + L6 | L0 L2 L3 L4 L7 L8 | slowest warm-up lap + one of the two 40 m/s laps; L1 is entered from the pit exit |
| F2 | L2 + L7 | L0 L1 L3 L4 L6 L8 | warm-up lap with a standstill + the other 40 m/s lap |
| F3 | L3 + L4 | L0 L1 L2 L6 L7 L8 | back to back: L4 is entered from a lap that is not in the map |

- L6 and L7 are in different folds so that every map contains a 40 m/s lap; every fold pairs a slow
  or mixed lap with a fast one.
- Two laps out per fold keeps each map at 6 laps (the test map has 9); leave-one-out (6 folds,
  7-lap maps) is closer to the test but twice the compute — kept for a final check.
- Protocol as D-004: one continuous run of the whole bag per fold from the first pose; only the
  validation laps are scored. The headline is the pool of all six validation laps (each once).
- Split B (D-003) stays as the stress test for a pit stretch without map coverage (D-020), which
  these folds cannot produce.

**Results of iteration 1** (method and parameters unchanged; `results/cv/`):

| | pos RMSE | lateral | longitudinal | heading | < 1 m | median | p95 | max | runtime mean / p95 |
|---|---|---|---|---|---|---|---|---|---|
| F1: L1 + L6 | 0.119 m | 0.044 m | 0.111 m | 0.079° | 100 % | 0.087 m | 0.229 m | 0.378 m | 61.9 / 83.5 ms |
| F2: L2 + L7 | 0.220 m | 0.180 m | 0.127 m | 0.215° | 100 % | 0.128 m | 0.442 m | 0.816 m | 58.0 / 75.8 ms |
| F3: L3 + L4 | 0.123 m | 0.066 m | 0.103 m | 0.114° | 100 % | 0.075 m | 0.255 m | 0.392 m | 62.1 / 85.6 ms |
| **pooled, 6 laps** | **0.162 m** | **0.114 m** | 0.114 m | **0.148°** | **100 %** | 0.093 m | 0.330 m | 0.816 m | |

Per lap: L1 0.127, L2 **0.265**, L3 0.106, L4 0.136, L6 0.104, L7 0.132 m. The tuning lap L5 (in no
map) scores 0.123 / **0.204** / 0.116 m in F1 / F2 / F3. Laps inside the map: 0.03–0.08 m.

**Finding: L2 drove a different line.** For 60 % of L2 its path is more than 1 m from every other
lap (median 2.6 m, up to 11–14 m; s ≈ 181–627 m and most of s ≈ 880–2,800 m). The rest of the laps
share one line within ~0.1 m. L5 drives L2's line for s ≈ 1,000–1,665 m (0.08 m from L2, 5 m from
all others). So only F2 holds out the one lap whose keyframes cover that line:
- L2 in F2: camera fixes in 66–98 % of image sets (100 % elsewhere), error 0.2–0.4 m all lap long.
- L5 in F2: 0.40 m and 73 % fixes on s 1,000–1,500 m; elsewhere unchanged.
- L7 in F2 is 0.13 m, like L4/L7 in the other splits, so the 40 m/s pairing is not the cause.

The method still localizes 5–11 m off the mapped line (100 % within 1 m), but at 2–4× the error.
Consequences: (1) the folds are not equally hard, and F2 is the informative one for a test run
whose line differs from the training laps; (2) the L5 tuning (D-015) was done with L2 in the map, so
L5's off-line stretch was covered while tuning; (3) for the test map (all laps, D-016) L2 and L5
are the only coverage of that line — keep both in.

---

## D-022 Swapped tuning lap: L2 instead of L5 in F1 and F2 — experiment

**Setup** (`configs/cv_f1_swapped.yaml`, `cv_f2_swapped.yaml`, `results/cv/f1_swapped/`,
`f2_swapped/`, pooled `results/cv/cv_score_swapped.json`). L2 becomes the tuning lap (in no map).
Parameters are **not** re-tuned (still `iter1.yaml`, D-015): the tuning role only means "held out of
the map, scored for information". Tuning is a separate offline step, not part of map building; the
map files hold only keyframe poses, 3D points and descriptors.
- **F1_swapped**: map L0 L3 L4 **L5** L7 L8 (new map `map_cv_f1_swapped`), validation L1 + L6.
- **F2_swapped**: map laps identical to F2 (map `map_cv_f2` reused), validation **L5** + L7.

**Results**

| fold | validation | pos RMSE | lateral | heading | < 1 m | longitudinal | median | p95 | max | runtime mean / p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 (map with L2) | L1 + L6 | 0.119 m | 0.044 m | 0.079° | 100 % | 0.111 m | 0.087 m | 0.229 m | 0.378 m | 61.9 / 83.5 ms |
| F1_swapped (map with L5) | L1 + L6 | 0.123 m | 0.046 m | 0.080° | 100 % | 0.114 m | 0.084 m | 0.253 m | 0.455 m | 58.8 / 76.6 ms |
| F2_swapped (= F2 run) | L5 + L7 | 0.176 m | 0.137 m | 0.175° | 100 % | 0.111 m | 0.093 m | 0.361 m | 0.900 m | 57.2 / 74.1 ms |
| pooled F1_swapped + F2_swapped + F3 | L1 L3 L4 L5 L6 L7 | 0.143 m | 0.091 m | 0.128° | 100 % | 0.110 m | 0.084 m | 0.285 m | 0.900 m | |

Tuning lap L2: 0.252 m in F1_swapped (L5 in the map), 0.265 m in F2_swapped (neither L2 nor L5).

**Findings**
1. **The F2_swapped run is byte-identical to F2** (same map laps, same parameters): the pipeline is
   deterministic and only the roles change. Runtime differs by timing noise (57.2 vs 58.0 ms).
2. **Coverage of the driven line is what matters, measured directly.** L2 split by where it drives:

   | part of L2 | share | F2 (no L2, no L5 in map) | F1_swapped (L5 in map) |
   |---|---|---|---|
   | on the common line (< 1 m from L1 L3 L4 L6 L7) | 42 % | 0.127 m, fixes 100 % | 0.149 m, 100 % |
   | off-line where L5 drove the same line | 16 % | 0.347 m, 85 % | **0.108 m, 100 %** |
   | off-line, no other lap nearby | 43 % | 0.324 m, 78.5 % | 0.350 m, 79.6 % |

   One lap that drove the same line brings the off-line error down to the on-line level; nothing
   else changes. This confirms D-021's explanation.
3. **For common-line laps, which lap fills the 6th map slot hardly matters**: L1 + L6 move from
   0.119 m (L2 in the map) to 0.123 m (L5 in the map).
4. **The two pooled numbers are not comparable**: 0.143 m (swapped) vs 0.162 m (D-021) swaps the
   hardest validation lap L2 (0.265 m) for L5 (0.204 m), and L5's score is optimistic because the
   parameters were tuned on it. D-021 remains the reference; the swap is a coverage experiment.

**Side finding (not fixed yet).** Two sanity-check constants, `reloc.checks.z_offset_m` (-0.05 m,
median car height above the track surface) and `roll0_deg` (-1.25°, median roll), were computed by
`scripts/prepare.py` from the GT of the whole bag, validation laps included. They are used only in
the accept/reject checks (tolerances 1 m and 3°) and as the assumed height/roll in guided matching,
so the effect is expected to be negligible, but they should be computed from training laps only.

---

## D-023 What the tuning buys: the folds with the first-guess values — measurement

**Setup** (`configs/cv_f1_untuned.yaml`–`cv_f3_untuned.yaml`, `results/cv/f*_untuned/`, pooled
`results/cv/cv_score_untuned.json`). The three folds of D-021 with their maps unchanged, but with the
values used before tuning (D-015, `results/iter1/tune_cams5/summary.json`): fix σ_xy 0.3 m (floor
0.15 m), σ_yaw 0.3°, process noise halved (v_long = v_lat = 0.05 m/s/√Hz, wz = 0.0015 rad/s/√Hz,
bias_rw = 1e-5). This is a report, not a choice made on the validation laps: the tuned values stay.

| | tuned (D-021) | untuned | gain from tuning |
|---|---|---|---|
| **pooled position RMSE** | **0.162 m** | 0.186 m | **−0.024 m (−13 %)** |
| lateral RMSE | 0.114 m | 0.128 m | −0.014 m (−11 %) |
| longitudinal RMSE | 0.114 m | 0.135 m | −0.021 m (−16 %) |
| heading RMSE | 0.148° | 0.153° | −0.005° (−3 %) |
| median / p95 / max | 0.093 / 0.330 / 0.816 m | 0.128 / 0.342 / 0.899 m | −0.035 / −0.012 / −0.083 m |
| within 1 m | 100 % | 100 % | — |
| F1 (L1 + L6) / F2 (L2 + L7) / F3 (L3 + L4) | 0.119 / 0.220 / 0.123 m | 0.131 / 0.250 / 0.155 m | −0.012 / −0.030 / −0.032 m |
| runtime mean | 58–62 ms | 60 ms | — |

Per lap (position RMSE, tuned → untuned): L1 0.127 → 0.125, L2 0.265 → 0.299, L3 0.106 → 0.138,
L4 0.136 → 0.169, L6 0.104 → 0.141, L7 0.132 → 0.158 m. Tuning lap L5: 0.123 / 0.204 / 0.116 m tuned,
0.134 / 0.227 / 0.129 m untuned (F1 / F2 / F3).

**Findings**
1. **The tuning transfers but is worth less than L5 suggested**: −13 % pooled against −20 % in the
   L5 lap restart where it was chosen. It helps five of six laps by 0.026–0.037 m; the slowest lap,
   L1, is unchanged (0.127 vs 0.125 m; its lateral improves, its longitudinal gets worse).
2. **Off the mapped line, tuning helps along the track but not across it.** L2's off-line part:
   0.371 → 0.330 m position, longitudinal 0.220 → 0.145 m, lateral 0.299 → 0.296 m. This corrects the
   expectation that the smaller fix σ would hurt off-line: it does not. But the off-line lateral
   error does not respond to the noise values at all, which fits the systematic bias of D-022 (fixes
   overshoot away from the mapped line by ~5 % of the offset): noise tuning cannot remove a bias.
3. **Both settings are overconfident; the tuned one slightly less.** RMSE / RMS of the filter's
   σ_xy (= √(P_xx + P_yy); 1.0 = honest): tuned 1.6–2.3, untuned 1.7–2.8. The doubled process noise
   lets σ grow between fixes, so the filter follows the fixes faster.
4. The tuned filter rejects more fixes at the gate (2–12 % against 0.3–4.6 %), without lock-outs.

---

## D-024 Matching frames from an unknown camera against the map — sanity-check tool

**Purpose.** No day data with ground truth exists for this track (the training bag is floodlit night
only; RACECAR has no night sessions; A2RL Vmax has no camera images in any of its 754 files;
4Seasons needs registration and is not a race track). To still see whether daylight frames
*locally match* the night map, `scripts/match_external.py` takes frames of an unknown camera (e.g.
screenshots of onboard footage) and searches the map without a pose prior or intrinsics. Not part
of the method.

**How.** SIFT as in the pipeline on the frame scaled to our half-res width → nearest map points over
all mapped cameras, with the ratio test against the nearest point of a *different place* (> 3 m; each
place is in the map once per lap) → densest place → PnP RANSAC over a grid of fields of view
(24–141°) and k1 → checks (inliers, height, roll, pitch, inside the track). Output: per frame the
ratio matches, the matches agreeing on one place, PnP inliers, the matched night keyframe (lap, time)
and a picture of the matches next to it.

**Control** (6 frames of camera `fr` from L6, against the F1 map, which does not contain L6):

| frames | right place (≤ 2 m) | matches on that place | PnP inliers | pose error |
|---|---|---|---|---|
| as recorded, true lens | 6 / 6 | 62–235 | 20–142 | 0.09–0.37 m |
| as recorded, lens unknown (grid) | 6 / 6 | 62–235 | 19–69 | 0.35–14.8 m |
| re-compressed (JPEG 35), true lens | 6 / 6 | 29–149 | 8–77 | 0.09–0.30 m |
| re-compressed, lens unknown | 6 / 6 | 29–149 | 9–42 | 1.6–27.8 m |
| mirrored (must not match) | 0 / 6 | 0 (6–21 scattered ratio matches) | 0 | — |

**How to read it.** Finding the right place does not need the lens; the pose does. For external
frames the measure is the number of matches that agree on one place, and whether that place is
where the frame was taken; inliers and pose are not meaningful with an unknown lens (a wrong focal
length slides the camera along its viewing direction). Web-like compression costs ~40 % of the
matches. Day and night frames should come from the same video (same camera, compression and
viewpoint) so that only the light differs. Screenshots go to `external_frames/` (gitignored, like
`results/external/`: third-party material is never committed).

---

## Open issues for iteration 2

1. **Lighting**: the map is night-only. Add a lighting-robust measurement that does not depend on
   appearance matching (border/kerb geometry, or the track-relative network from the plan), and
   learned features (SuperPoint + LightGlue) that match better across lighting.
2. **Recovery without a prior**: global retrieval (e.g. DINOv2 descriptors over all keyframes) to
   re-initialize if the filter is lost; today recovery relies on the prior being within ~40 m.
3. **Covariance**: PnP-Jacobian covariance per fix, correlation between cameras, NEES check (D-014).
4. **Runtime**: within the 100 ms frame period on an unloaded M4 (p95 77–82 ms), but p95 reached
   124 ms under memory pressure; batch matching on the GPU, or fewer cameras per frame in rotation,
   would add margin.
5. **Pit lane**: validated on the held-out out-lap L0 since split B (D-003), but only without map
   coverage, because the two pit passes do not overlap; a run with the pit lane mapped by another
   pass does not exist in the training data.
6. **Re-acquisition after long stretches without fixes** (D-020): a wrong fix accepted through the
   wide gate made the filter overconfident and locked out correct fixes for 8 s.
7. **Coverage across the track width** (D-021): keyframes lie on the driven lines only; off the
   mapped line (L2 in fold F2) the fix rate drops and the error doubles to quadruples. Options:
   matching windows that do not assume the keyframe viewpoint, keyframes from all cameras pooled
   (a side camera on one line sees what a front camera sees from another), or synthesizing views.
   D-022 shows one lap on the same line is enough to remove the penalty.
8. **Tuning per fold** (D-022): parameters are tuned once (D-015) and shared by all folds; nested
   tuning (L2 + L5 against a map without L2 in F1/F3, L5 only in F2) is the planned next step.
9. **Check constants from all-bag GT** (D-022): recompute `z_offset_m` / `roll0_deg` from training
   laps only.
