# Visualization

A dashboard of the camera localization (`localization/`, iteration 1): an overhead track view of the
estimate, live error or health plots, the camera fixes and running metrics. It plays in a window at the
10 Hz image-set rate and renders to mp4 (1920×1080, 10 fps, one video frame per image set).

`localization/camloc` is used as a read-only library: nothing in `localization/` is changed, and ground
truth is used only to score and draw, never as an input of the pipeline.

## Run modes

Two modes: **mapped** replays a run of the night training bag, whose laps make up the map, with ground truth; **test** runs the pipeline on the test-format daytime sample, without ground truth. Every run also saves its final frame (`final_frame.png`) and each panel of it as an image in `plots/` (track, camera, chase, error_scope or camera_matching, lateral_error / heading_error or racing_line_offset / filter_sigma, fixes, metrics).

All commands run from the repository root. Outputs go to `visualization/outputs/<name>/`, which is
gitignored.

| mode | command | what runs | wall time | writes |
|---|---|---|---|---|
| **mapped, video** | `python visualization/run_viz.py mapped` | replay of a stored continuous run, with ground truth | ~15 s | `<name>.mp4`, `final_frame.png`, `plots/*.png`, `metrics.json`, `chunk_poses.csv`, `chunk_gt.csv`, `official_evaluate_chunk.txt` |
| mapped, window | `... mapped --live` | the same, played at 10 Hz in a window, then the video | 30 s + 15 s | the same |
| **test, video** | `python visualization/run_viz.py test` | `pipeline.run` on the daytime bag as fast as it goes, then the video | ~30 s + 35 s | `test_<t0>-<t1>s.mp4`, `final_frame.png`, `plots/*.png`, `metrics.json`, `poses.csv`, `log.pkl` |
| **test, window** | `... test --live` | `pipeline.run` paced at 10 Hz in a worker process, drawn as frames arrive, then the video | 30 s + 35 s | the same |
| test, re-render | `... test --from-log visualization/outputs/test_0-30s/log.pkl` | the video again from a recorded run, without the pipeline | ~15 s | `<name>.mp4`, `final_frame.png`, `plots/*.png`, `metrics.json` |
| test, replay | `... test --from-log <log.pkl> --live` | a recorded run replayed at 10 Hz in a window | 30 s | the same |
| **smoke test** | `conda run -n odosseus-sp python visualization/smoke_matching.py` | day vs night frame matching: SuperPoint + LightGlue next to SIFT | ~90 s | `outputs/smoke_matching/pairs.csv`, `summary.json`, match pictures |

Options:

| option | modes | default | meaning |
|---|---|---|---|
| `--lap`, `--t0`, `--t1` | night | `L7`, `0`, `30` | chunk = lap start + [t0, t1) s |
| `--run` | night | `localization/results/cv/f2/fullrun/submission.csv` | a `scripts/run.py` output CSV, with its `_diag.npz` next to it |
| `--no-thumbs` | night | | no camera thumbnail, so no image reading from the bag |
| `--t0`, `--t1` | day | `0`, `30` | chunk = first image of the bag + [t0, t1) s |
| `--calib training\|bag` | day | `training` | rr / rl extrinsics used at query time. `training` is what the submission does; `bag` uses the day bag's own (rr +2.2° roll / −1.1° pitch, rl +0.5° / +0.3°) |
| `--sigma0-xy`, `--sigma0-yaw` | day | `3.0` m, `1.0` ° | initial uncertainty of the stand-in first pose |
| `--offset-from` | test | `10` | the offset from the racing line from this time on: RMS, mean, p95, max in `metrics.json` (`offset_after`) and the plot `plots/racing_line_offset_after_<s>s.png` |
| `--from-log` | day | | render or replay a recorded run (`log.pkl`) instead of running the pipeline |
| `--live`, `--speed`, `--no-hold` | all | real time | window; playback speed factor; close the window at the end instead of keeping it open |
| `--no-mp4`, `--name` | all | | skip the video; output folder name |

The defaults reproduce the two published runs: `outputs/mapped_L7_0-30s/` and `outputs/test_0-30s/`.

## Requirements

- Env `odosseus` (`localization/environment.yml`): numpy, scipy, matplotlib, opencv, mcap, mcap-ros2-support,
  pyyaml. Plus the `ffmpeg` binary for the mp4. A legend in each chase view names its lines.
- Env `odosseus-sp`, only for the smoke test: a clone of `odosseus` plus `torch`, `kornia` and LightGlue
  (`pip install -r visualization/requirements-smoke.txt`). The SuperPoint and LightGlue weights
  download on first use (cvg/LightGlue releases, ~50 MB).
- Data, not in git:
  - `hackFiles/c0_2026-10-13_22-53-42_camloc/`: the training bag (night thumbnails) and the official `tools/evaluate.py`.
  - `hackFiles/DAYTIME_TEST_rosbag_2026-10-13_16-08-46_cameras_lap/`: the daytime bag.
  - `hackFiles/23-00-38_cut_120m_update/`: `map.mcap` (racing line) and the organizers' `tools/track_frame.py`.
  - `localization/artifacts/` (built by `localization/run.sh` / `scripts/prepare.py`): `map_final/` (the map of
    all 9 night laps), `masks/`, `calibration.json`, `track_bounds.npz`, `gt_laps/`, `laps.json`, `state.npz`.
  - `localization/results/cv/f2/fullrun/`: the stored fold-2 continuous run that the night mode replays
    (`scripts/run_cv.sh`).

## What the views show

**Mapped mode (`TruthDashboard`): the night training bag, with ground truth.** It replays a continuous run of the whole bag from
its first pose (localization D-004); the chosen lap is not in that run's map (fold 2: L2, L7 held out).

- **Track view:** borders (grey), the chunk's true path (faint grey), the truth so far (thin black) and the
  estimate (blue). The blue gets darker with the size of the lateral error, saturating at 0.5 m.
- **Chase view:** heading up, centred on the **true** car (black outline), ±22 m. The lines are:
  - borders (grey; pit lane lighter);
  - the truth so far (thin black);
  - the estimate over the last 5 s (wide blue) and its car (blue outline).

  At this scale a 0.1 m error is invisible, so the black and blue lines coincide.
- **Error scope:** the estimate relative to the true car (+), in the true car's frame. Left of truth shows on the
  left, ahead shows up. It keeps the last 3 s; rings mark 0.25, 0.5, 1 and 2 m.
- **Lateral and heading error:** estimate minus the ground truth interpolated at the image stamp, split across and
  along the true heading. This is the official math (`camloc.metrics.errors` = `tools/evaluate.py`); + means the
  estimate is left of the truth.
- **Running metrics:** the official metrics over the frames so far. At the end they are checked against
  `camloc.metrics.score` and `tools/evaluate.py` on the chunk.

**Test mode (`HealthDashboard`): the test-format daytime sample, no ground truth, so no errors.**

- **Track and chase views:** the organizers' **racing line** (faint grey; update `map.mcap`) and the estimate
  (blue). In the track view the blue gets darker with the filter's σ, saturating at 3 m. The chase view is
  centred on the **estimate**. The line through or near the car is the racing line: the car drives close to it.
- **Camera matching, this frame:** per camera, the matches left after the pose filter. PnP needs ≥ 20 of them
  (the panel says so) (`reloc.query.min_matches`), then ≥ 15 PnP inliers, an inlier ratio ≥ 0.2, height / roll / pitch
  checks (D-013) and the filter's χ² gate (D-014). Bar colour: grey no fix, aqua accepted, orange rejected at the gate.
- **Offset from the racing line:** the organizers' track-frame `n` (`track_frame.Line.xy_to_sn`), + = left of the
  line. Always shown at ±5 m (values beyond run off the panel). This is context, not an error: the car need not drive exactly on the line.
- **Filter position σ:** how sure the filter is; it grows while no fix is accepted. The y range follows the last
  6 s; the final frame shows the whole chunk.
- **Health metrics:** fix rate, fixes accepted / produced / tried, median matches, σ, track-constraint share,
  runtime, coverage.

**In both views:**
- **Blue band:** the filter's own ±1σ per axis, √((Pxx + Pyy) / 2) from the EKF covariance. It is not an
  error. It grows with the car-state process noise between fixes and shrinks with each accepted fix, whose σ is
  0.15 m · √(100 / inliers) (D-014, D-015). At night, the error leaving the band shows the filter's known
  overconfidence (~2×, D-023).
- **Front-camera thumbnail** and **camera fixes per frame:** accepted fixes, and fixes rejected at the gate.
- **Header:** chunk time, speed (car state) and frame. Two kinds of note can appear:
  - `display N frames behind`: the window skipped drawing frames. They were computed anyway.
  - `processing X s behind the 10 Hz camera clock (no pose dropped)`: day only. Frames took more than 100 ms, so the 10 Hz camera clock got ahead
    of the processing. No pose is dropped and every pose keeps its image stamp, exactly as in the submission run.

## Real time, as the hackathon scores it

- **Each pose is scored at its image stamp.** The brief and INTRO slide 13 ask for one pose per image set,
  `timestamp_ns` = the image header stamp, causal. The ground truth is interpolated at that stamp (INTRO slide 14;
  "gaps over 50 ms are skipped" refers to gaps in the ground truth, not to processing time).
- **No pose is dropped for being slow.** Slide 18 says "never skip a frame". Runtime only enters as the
  self-reported average processing time per frame, 5 % of the score, full points at or below the 100 ms frame
  period.
- **"A late pose is a wrong pose"** (INTRO slide 7: 4 m at 40 m/s in 0.1 s) is the reason for that metric: it is
  what latency costs on the car. It is not a scoring rule.
- So the day window shows the lag honestly, but the scored result is the same as the unpaced run. The measured
  runtime (`pipeline.run`'s own timer, the 'Runtime per frame' row) is what matters: day 95 ms mean / 121 ms p95
  unpaced, 102 / 129 ms while the window draws (49 % of frames over 100 ms). Night: 64 / 81 ms.

## How it works

| file | role |
|---|---|
| `run_viz.py` | command line: the `night` and `day` modes, outputs, cross-checks |
| `dashboard.py` | the figure: `Dashboard` base (layout, track view, thumbnail, chase view, fixes, blitting), `TruthDashboard` (night), `HealthDashboard` (day). `render(k)` draws frame k from per-frame arrays; static parts are drawn once (blitting, ~25 ms per 1080p frame) |
| `player.py` | `live()`: window paced at the image stamps, skips drawing when behind. `mp4()`: every frame piped to ffmpeg, final frame held 3 s |
| `replay.py` | night source: a stored run's CSV + `_diag.npz` cut to the chunk; `fr` thumbnails from the bag |
| `live_metrics.py` | per-frame errors and running official metrics; the final values must equal `metrics.score` |
| `reference.py` | paths, track borders, laps, ground truth (`artifacts/gt_laps/full.csv`), the racing line |
| `daybag.py` | day inputs: car state from the twist topics, first image stamp, the stand-in first pose, the bag-calibration swap |
| `live_day.py` | day live run: `worker()` runs `camloc.pipeline.run` in a separate process; `Log` collects the frames; `health_at()` |
| `smoke_matching.py` | the smoke test (below) |

**Day live run.**
- **Inputs:** the daytime bag has its car state on `/constructor0/validation/twist` + `/accel` (the motion fields
  of the car's state estimation), no track borders and no first pose. `daybag.py` reads the twist, uses the
  training bag's borders, and makes a **stand-in first pose**. That is the lap's motion data placed on the track by
  matching its curvature over distance against L7. It is non-causal, good to a few metres, and the run starts with
  σ 3 m / 1°. The final test gives the real first pose.
- **Worker:** `camloc.pipeline.run` (`configs/final.yaml`, `map_final`, cameras cr cl rr fr rl) runs unchanged in
  a worker process. The worker wraps, at run time and in that process only, the image-set reader and the
  relocalizer's `localize` / `feedback` / `localize_camera`. These observe each frame (final pose, P, fixes,
  per-camera matches) and pace the frames at 10 Hz. The wrappers run outside `pipeline.run`'s per-frame timer.
  `guard` in `metrics.json` confirms every shown pose equals what `pipeline.run` returned.
- **Window:** draws in the main process (macOS needs the GUI there; ~80 ms per frame on a Retina screen).

## Results (2026-10-18)

| run | result |
|---|---|
| night, L7 0–30 s (782 m, 0–144 km/h) | 0.136 m position, 0.054 m lateral, 0.069° heading, 100 % < 1 m, 100 % coverage (= `tools/evaluate.py`); whole lap L7 0.132 / 0.058 m / 0.119°, as in the localization README |
| day, 0–30 s (up to 245 km/h) | fixes in 50 % of frames (248 accepted of 261 produced, all 5 cameras). None in the first 8 s, because of the stand-in start. After that σ is 0.1–0.4 m and the estimate stays within ±0.5 m of the racing line. Runtime 95 ms mean / 121 ms p95 unpaced, just under the 100 ms budget (night: 64 ms); 102 / 129 ms while the window draws |

## Smoke test: day frames against night frames (`smoke_matching.py`)

**Setup.**
- **Places:** 12 points across the day chunk, every 2.5 s, found by distance along the track from the start line.
- **Pairs:** at each place, each of the 7 cameras: the day frame, and the L7 night frame at the same place as the
  reference. Each is matched against night frames of L3 and L6 (no lap in common).
- **Matching:** no pose guess. An essential-matrix RANSAC (2 px) then keeps the geometrically consistent matches.
- **Matchers:**
  - SuperPoint + LightGlue, with the settings of the alternate approach on `origin/main` (`odo/`): 2048
    keypoints, long side 1024 px, car body cropped.
  - SuperPoint mutual nearest neighbours with similarity > 0.75: its tracking matcher, without the search window.
  - camloc's SIFT + ratio test, on the same image region.
- **What the counts mean:** these are two-view inliers. They count correct correspondences, not 2D–3D fixes against
  a map.

| query / matcher | median verified inliers | p10 | pairs ≥ 30 | pairs ≥ 15 |
|---|---|---|---|---|
| night L7 / SuperPoint + LightGlue | 324 | 183 | 100 % | 100 % |
| night L7 / SuperPoint mutual NN (sim > 0.75) | 287 | 140 | 99 % | 100 % |
| night L7 / SIFT + ratio (camloc) | 182 | 75 | 99 % | 100 % |
| **day / SuperPoint + LightGlue** | **162** | 62 | **94 %** | 98 % |
| day / SuperPoint mutual NN (sim > 0.75) | 21 | 8 | 35 % | 72 % |
| day / SIFT + ratio (camloc) | 13 | 7 | 22 % | 46 % |

On the M4 (MPS), SuperPoint takes ~56 ms per image and LightGlue ~59 ms per pair.

## Known limits

- **No ground truth for the day run.** Its offsets are against the racing line, which the car need not follow
  exactly.
- **The stand-in first pose is a few metres off.** The first ~8 s without fixes come from it, not from daylight.
- **The band is not the lateral covariance.** It is the isotropic per-axis σ; the night replay only stores √(Pxx + Pyy).
