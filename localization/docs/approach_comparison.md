# Design note: this pipeline vs the `main` branch, and a SIFT fallback for `main`

Compares the camera localization of this branch (`localization/`, `camloc`, "iteration 1") with the teammates'
approach on `origin/main` (`odo/`, final method Exp21/Exp24, commit `9433a6a`). It covers:
- whether this pipeline's SIFT map could serve `main` as a fallback;
- which ground-truth (GT) data `main` uses, and whether each use is available at inference.

Sources:
- this branch: `camloc/ekf.py`, `odometry.py`, `pipeline.py`, `reloc/query.py`, `configs/iter1.yaml`, `docs/DECISIONS.md`;
- `main`: `odo/online.py`, `odo/map_build.py`, `odo/data.py`, `scripts/extract_mcap.py`, `scripts/run_submission.sh`, `scripts/tune_l5.py`, `scripts/sweep_online.py`, `docs/*.md`;
- the hackathon brief, the INTRO slides and the organizers' note on the daytime sample.

Date: 2026-10-18.

## 1. Stages

`main` names its work in stages:

| stage | content | in `main`'s final system? |
|---|---|---|
| 1 | visual odometry | no: the car state is ~4× more accurate over 1–10 s |
| 2 | retrieval + SuperPoint/LightGlue + PnP (Exp12, 410 ms per frame) | no: replaced by stage 4 |
| 3 | offline 3D feature map (Exp20) | yes |
| 4 | real-time localiser: EKF, guided matching, PnP, recovery (Exp21, Exp24) | yes |

Notes:
- **LightGlue runs only offline, in stage 3**, to triangulate the map. At run time, `main` tracks and recovers by
  matching SuperPoint *descriptors* against map points projected from the predicted pose. Recovery uses a wider
  window, around MegaLoc place-retrieval candidates.
- **This branch has the same structure without the staging:**
  - a SIFT keyframe map built offline (D-012);
  - an EKF on the car state (D-006);
  - per-camera guided matching and PnP (D-013), with gated updates (D-014).

## 2. Kalman filters

| | this branch (`camloc/ekf.py`) | `main` (`odo/online.py`, `run()`) |
|---|---|---|
| state | x, y, yaw, **gyro bias b** | x, y, yaw |
| prediction | every car-state sample (~105 Hz), midpoint rule, yaw rate minus b | once per image set: samples between frames integrated (zero-order hold, heading midpoint) |
| process noise | continuous white noise, Q ∝ dt: 0.1 m/s/√Hz along and across (+0.2 % of vx), yaw rate 0.003 rad/s/√Hz, bias random walk 2e-5 rad/s/√s | per step, proportional to distance: σ_lon = 1 % of distance + 5 mm, σ_lat = 0.5 % + 5 mm, σ_yaw = 0.0005 rad + 1 % of the turned angle |
| standstill | yaw frozen, zero-rate update of b, initial b from the starting standstill | none |
| time base | image stamp | image stamp − 9.1 ms (camera-vs-GT clock offset fitted on the tuning lap); output shifted back to the stamp |
| initial σ | 0.1 m, 0.2°, b 3e-4 rad/s | 0.05 m, 0.11° |
| camera update | one per camera fix (up to 5 per frame), each gated; R isotropic, σ_xy = 0.15 m · √(100/inliers) clipped 0.08–2 m, σ_yaw = 0.2° · √(100/inliers) clipped 0.15–2° | one per frame, from a joint 7-camera PnP (x, y, yaw + Δroll, Δpitch, Δz with priors); R = 4 × PnP covariance + floor (0.10 m, 0.23°)²; < 60 inliers updates position only |
| update form | Joseph form | (I − K)P |
| gate | NIS χ²(3) 16.27 | Mahalanobis d² 16.0 |
| other updates | one-sided track-border constraint (≥ 1 m inside, σ 0.3 m) | none (borders tried, Exp22, dropped) |
| recovery | lock-out guard: 3 s of only rejected fixes (≥ 6) inflates σ by 2 m / 1° | challenger: 8 rejected fixes agreeing with the first, propagated by dead reckoning, reset the state. Relocalisation: after 1 s without a fix, MegaLoc candidates + wide-window PnP; a jump needs ≥ 80 inliers and must be plausible (0.5 m + 0.3 m/s, 0.3° + 0.1°/s) unless ≥ 300 inliers |

## 3. Camera measurement

| | this branch | `main` |
|---|---|---|
| features | SIFT, CLAHE, half resolution, ≤ 2000, car-body masks | SuperPoint, ≤ 2048, long side 1024 px, body cropped |
| map | keyframe every 2 m per camera, ≤ 800 points, triangulated against up to 3 later keyframes; `map_final` 0.74 GB, CPU | every training frame of every camera, LightGlue to the previous and next frame; 20 M points, 10.6 GB, GPU; frames near GT accuracy spikes avoided |
| matching | ratio test against 3 nearest keyframes, then guided angular filter; PnP RANSAC per camera | projected map points matched within 24 px (+40 px/s without a fix) by descriptor similarity, two passes; joint Gauss–Newton PnP (Huber) |
| cameras | 5 (cr cl rr fr rl) | 7 |
| runtime | ~60 ms per frame at night, ~95 ms by day (Apple M4 CPU) | ~41 ms (H100 GPU); CUDA only |

## 4. Tuning

Both use L5 as the tuning lap and the same three cross-validation folds (F1–F3, identical map laps).

| | this branch | `main` |
|---|---|---|
| tuned | fix σ_xy (0.15 m) and σ_yaw (0.2°), process noise ×2, camera set (5) (D-013, D-015) | cameras (7), 3- vs 6-DoF PnP, noise floor (0.10 m / 0.23°), map passes (3), match radii and similarity thresholds, keypoints (2048), relocalisation gate, challenger (8 frames), selection radius (15 m), clock offset (−9.1 ms); Exp24: relocalisation yaw rule, position-only below 60 inliers |
| set by hand | 20 matches / 15 inliers / 0.2 ratio, map settings, guided-filter σ, track margin | process noise, PnP priors, GT-accuracy cut-off (1 m) |
| tuned on | L5, restarted from GT, 1 map | stage-4 sweep on the old dev laps (≈ L4 + L5); recovery rules re-tuned on L5 under all fold maps |
| validation run | one continuous run from the bag's first pose, scored per lap (D-004) | each lap restarted from its own first GT pose |

Validation position RMSE per lap. The protocols differ: `main` restarts each lap from GT, this branch runs
continuously. D-004 found the difference small for camera fixes.

| lap | L1 | L6 | L2 | L7 | L3 | L4 |
|---|---|---|---|---|---|---|
| `main` (Exp21) | 0.080 | 0.083 | 0.302 (max 1.73) | 0.089 | 0.068 | 0.117 |
| this branch | 0.127 | 0.104 | **0.265 (max 0.82)** | 0.132 | 0.106 | 0.136 |

`main` is better on the five laps that follow the mapped line. This branch is better on L2, which is driven up
to 14 m off the mapped lines.

## 5. Daylight (the daytime sample, `visualization/README.md`)

- **Prior-free smoke test** (two-view inliers, day vs night frames of the same place):
  - SuperPoint + LightGlue: 162 median (324 night-vs-night);
  - SuperPoint descriptor matching, `main`'s tracking matcher without its window: 21 (287);
  - SIFT: 13 (182).
- **This pipeline on the full daytime lap:** a camera fix in 61 % of frames, 0.17 m RMS from the racing line after
  lock-on, 95 ms per frame.
- **`main` has only been tested on synthetic day images, not on this sample.**

## 6. A SIFT fallback for `main`

**Feasible, structurally.** Both produce the same measurement: the base_link x, y, yaw in the map frame, from PnP
against a map triangulated from the same training laps. A SIFT fix can enter `main`'s EKF as an extra measurement.
The new parameters can be tuned with `main`'s own protocol (`scripts/tune_l5.py`, L5 under F1–F3). The per-fold
SIFT maps (`artifacts/map_cv_f1..3`) already exist.

**What has to change:**
1. **Trigger.** Lateral error needs GT and cannot be measured at inference. Observable proxies:
   - the filter's lateral σ: √(nᵀ P_xy n), with n the heading normal. Use a threshold of about 0.25 m for a 0.5 m
     error, since both filters are ~2× overconfident;
   - time since the last accepted fix;
   - weak or rejected fixes (fewer than 60 inliers, gated);
   - a large lateral innovation;
   - `map_dist`, the distance from the mapped lines. `main` computes it already, and off the mapped lines is where
     `main` is weakest and this branch did better.
2. **Clock.**
   - The SIFT map assumes no camera-vs-GT offset, while `main`'s state is 9.1 ms early.
   - That is a 0.36 m along-track disagreement at 40 m/s, and 0.62 m at the daytime lap's 68 m/s.
   - Fix it by shifting each SIFT fix by speed × 9.1 ms, or by rebuilding the SIFT map with the offset.
3. **Measurement model.**
   - Up to 5 isotropic per-camera fixes have to become `main`'s format: one fix with a 3×3 covariance, or several gated updates.
   - Re-tune R inside `main`'s filter, because this branch's σ values belong to its own filter.
4. **Runtime and hardware.**
   - Everything runs on the GPU workstation.
   - SIFT runs on the CPU in parallel. Fallback frames cost about 41 + 60–95 ms, so it must stay rare to keep the
     average under 100 ms.

**Expected benefit:**

| case | benefit |
|---|---|
| mapped laps at night | small: `main` gets a fix in 98.5–100 % of frames |
| lines off the map (L2-type, pit stretches not in the map) | plausible: L2 0.265 vs 0.302 m, worst pose 0.82 vs 1.73 m; triggered by `map_dist` |
| daylight | unlikely: SIFT is the weakest matcher by day. A better daylight fallback inside `main` is LightGlue between the query and the stored map frames (keypoints and descriptors are in the map) when guided matching returns few matches |
| as a second opinion | valuable: require an agreeing SIFT fix before accepting a relocalisation jump. That targets `main`'s documented open failure, an 18 m wrong jump with 328 inliers that passed the ≥ 300-inlier bypass |

**Recommendation.** Not before the deadline. In iteration 2, implement first:
- the clock shift;
- the `map_dist` trigger;
- the second-opinion check on relocalisation jumps.

Tune them on L5 under F1–F3 with `main`'s parameters frozen.

## 7. Ground truth in `main`: what it uses, and whether it exists at inference

What the rules allow:
- **INTRO slide 10:** allowed as input are images, `camera_info`, `/tf_static`, track borders, "speed,
  accelerations and yaw rate from state_estimation", and the first pose of the run. Not allowed are `/tf`, and the
  position, orientation and s, n from `state_estimation`. "Ground truth at inference means disqualification."
- **The brief:** GT poses are "only used for training and for validation"; "Ground truth may not be used at
  inference, except the first pose of the run".
- **INTRO slide 17 (method 3a):** "Every training image has a ground-truth pose … triangulate them with the
  ground-truth poses", so a map from the training run is explicitly intended.
- **The organizers' daytime note:** other fields of the car's state estimation ("SE") may be used.

| GT use in `main` | where | at inference | assessment |
|---|---|---|---|
| training GT 6-DoF poses to triangulate the map, looked up at stamp − 9.1 ms | `map_build.py` | not needed: the map is prebuilt | allowed (slide 17). Same as this branch (D-012) |
| poses of the map frames: height, roll and pitch priors for PnP; selecting map frames | `measure()`, `FeatureMap` | from the map | allowed: training GT inside the map |
| GT `pos_accuracy` of the training run, to drop map frames near accuracy spikes | `FeatureMap` reads the **training** `state.csv` at start-up | the training files must ship with the map | allowed as training metadata, but it is a `state_estimation` field other than motion. **Declare it**; check with the organizers if unsure |
| poses of the MegaLoc database images | `Reloc` (`data.interp_gt`, training) | from the training run | allowed: training GT |
| clock offset −9.1 ms, fitted from longitudinal error vs speed on a tuning lap | `map_build.py`, `online.py` | a fixed constant | allowed (tuning). **Risk:** the test session may have a different offset. Slide 19: "At 40 m/s, 10 ms is 0.4 m"; at 68 m/s 1 ms ≈ 7 cm |
| first pose | `Run.first_pose()`: `first_pose.csv`, **else the first row of the test run's `gt.csv`**; `run_submission.sh` writes `first_pose.csv` from that `gt.csv` | only if the test bag carries `/tf`, or the organizers give the first pose as a file | allowed (the first pose only). **Risk:** the daytime example has no `/tf`, so no `gt.csv`. The script then has no first pose, and test mode needs one supplied by hand (`first_pose.csv` or `cfg['init_pose']`) |
| initial σ 0.05 m / 0.11° | `run()` | assumes the given first pose is GT-accurate | fine if the given pose is that accurate; inflate it otherwise |
| validation: each lap starts from its own first GT pose (`data.first_pose` at the segment start) | CV scripts | **not available**: a test run has only its first pose | not a leak in the method, but **the reported CV numbers rely on mid-run GT restarts**. A continuous run from the bag's first pose is the honest equivalent (D-004) |
| GT error printed during runs | `run()` (`verbose`, not in test mode) | not used | reporting only |
| car-state speed, vx, vy, yaw rate | `CarState`; daytime bag via `/validation/twist` + `/accel`, v = \|(vx, vy)\| | yes | allowed (slide 10). `main`'s `extract_mcap.py` already reads the daytime topics |
| tuning and selection of every parameter on laps with GT | scripts | not needed | allowed: tuning lap L5, validation laps untouched by protocol |

**This branch, for comparison:**
- **Map:** training GT (D-012).
- **Check constants:** `z_offset_m` and `roll0_deg` come from the GT of the whole bag. That is a known small leak
  (D-022), to be recomputed from training laps.
- **Speed scale:** E-5's correction comes from training-lap GT. It's an experiment, not adopted.
- **First pose:** from the first `/tf` message, or `--first-pose`.
- **Gyro bias:** from the starting standstill, no GT. It defaults to 0 at a flying start.

## 8. Questions for the organizers

1. In what form does the hidden test run give its first pose (a single `/tf` message, a file, a number in the
   README)? Both pipelines' default first-pose readers need it.
2. Does the test bag carry state-estimation fields beyond `/validation/twist` and `/accel`?
3. Can the camera-vs-state clock offset differ between sessions? It decides whether `main`'s fixed −9.1 ms carries over.
4. May the training run's `pos_accuracy` field be used to filter the map?
