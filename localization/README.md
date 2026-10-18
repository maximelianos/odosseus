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

A camera fix reaches the filter only if it passes this chain (D-013):
1. at least 20 matches left after the guided filter (set by hand, not tuned);
2. PnP RANSAC at 4 px, leaving at least 15 inliers with an inlier ratio of at least 0.2;
3. height, roll and pitch checks;
4. the filter's χ² gate at 99.9 % (D-014).

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

## Experiments (candidates for iteration 2)

Changes to the method are tried here as experiments E-1 … E-4; what helps becomes iteration 2.
All runs so far use the iteration-1 method and parameters (`configs/iter1.yaml`), which is why
everything above is labelled "iteration 1".

**Protocol.** Fold 2 only (map L0 L1 L3 L4 L6 L8; validation L2 + L7; tuning lap L5), to keep the
compute low. L2 is the lap that drives off the mapped line (59 % of it more than 1 m away), so fold 2
is the informative fold for the lateral-error experiments. Development and every choice happen on
L5 (in no map; under this map 30 % of it is off the mapped line); the validation laps are run once
per finished variant, as a report. Report for every run: `scripts/exp_report.py` (off-line = more than
1 m from the GT paths of the map laps; overshoot = error component pointing away from the mapped line).

**Baseline (iteration 1, fold 2)** — `results/cv/f2/fullrun/`, `results/experiments/baseline_f2_report.json`:

| laps | pos RMSE | lateral | heading | < 1 m | sets with a fix | off the line | on-line pos | off-line pos | off-line lateral | overshoot share / slope |
|---|---|---|---|---|---|---|---|---|---|---|
| L2 + L7 (validation) | 0.220 m | 0.180 m | 0.215° | 100 % | 93.3 % | 39 % | 0.127 m | 0.316 m | 0.281 m | 91 % / 4.1 % of the offset |
| L2 | 0.265 m | 0.230 m | 0.262° | 100 % | 88.5 % | 59 % | 0.127 m | 0.330 m | 0.296 m | 92 % / 4.2 % |
| L7 | 0.132 m | 0.058 m | 0.119° | 100 % | 100 % | 10 % | 0.126 m | 0.175 m | 0.097 m | 82 % / 3.0 % |
| L5 (tuning) | 0.204 m | 0.176 m | 0.209° | 100 % | 97.4 % | 30 % | 0.103 m | 0.338 m | 0.310 m | 93 % / 5.4 % |

Runtime 58.0 ms mean / 75.8 ms p95 per image set.

**Status**

| ID | experiment | state |
|---|---|---|
| E-1 | lateral position from the track borders (idea 1) | done: not usable in this form (component test 15 % availability, 0.7 m error); not put in the filter |
| E-2 | map points triangulated across laps (idea 2) | E-2a done: depth is not the cause; E-2b (the cross-lap map) not built |
| E-3 | visual odometry to bridge fix gaps (idea 5) | done: VO heading is biased (E-3b) and makes gaps worse in the filter (E-3c); not extended |
| E-4 | fewer camera fixes, 100 / 50 / 10 % (stand-in 1 for daylight) | done on all three folds (the only experiment with a clear result, so the only one extended) |
| E-5 | forward-speed scale correction (lead from E-3a) | fold 2 done: along-track and gaps better, L2 lateral worse (+0.009 m on L2 + L7); extended as part of E-5 + E-6b |
| E-6 | fix σ growing with the lateral distance from the nearest keyframe (lead from E-2a) | fold 2 done: lateral −13 %, overshoot halved, but along-track and heading worse; E-6b (across the line only): position unchanged, lateral −7 % |
| E-5 + E-6b | the two together | done on all three folds: pooled 0.162 → 0.159 m, lateral 0.114 → 0.111 m, better on 5 of 6 laps; heading +7 % — recommended for iteration 2 |

**Summary of findings** (details in the sections below):

- **Two leads from these experiments do help, a little: the speed-scale correction (E-5) with a fix σ
  that grows across the mapped line (E-6b)** — pooled over the three folds 0.162 → 0.159 m position,
  0.114 → 0.111 m lateral, better on 5 of 6 validation laps, heading +7 %. Recommended for iteration 2.
- **None of the three requested method changes (E-1, E-2, E-3) improved the off-line error**; no
  validation run of them was warranted. Each failed for a measured reason:
  the off-line overshoot is camera-specific and not a depth effect (E-2a, so the cross-lap map was
  not built); the borders are not a clean edge and need pitch to ~0.1–0.2° from 0.34 m height (E-1);
  one camera's frame-to-frame rotation is biased and accumulates in gaps (E-3b, E-3c).
- **Daylight stand-in (E-4, three folds)**: thinning fixes at random costs little; 5–9 s without a fix
  add 0.3–0.5 m on every map. What decides a daylight score is how long matching fails in a row.
- **Leads tested** (E-5, E-6): the speed scale (the car's speed is 0.37 % low on every lap) cuts the
  along-track error in gaps by a third, and the fix σ that grows with the distance from the nearest
  keyframe halves the off-line overshoot; alone, each trades lateral against along-track error,
  together they gain a little on every metric but heading. **Still untested**: per-camera calibration
  from views on different lines (E-2a), pitch ≈ 0.9 × road slope from the 3D borders (E-1).
- Cost: about 21 MB of results and 1 MB of artifacts (no new maps); every experiment on fold 2 first;
  only E-4 and E-5 + E-6b extended to folds 1 and 3 afterwards.

### E-5 Forward-speed scale correction

`filter.vx_scale` (`camloc/odometry.py`), `scripts/exp_e5_speed_scale.py` / `.sh`,
`results/experiments/e5_speed_scale/`. The car-state forward speed reads low against the GT speed on
every lap (above 5 m/s): L0 +0.33 %, L1 +0.37 %, L3 +0.39 %, L4 +0.37 %, L5 +0.37 %, L6 +0.41 %,
L8 +0.38 % (GT speed / car speed − 1). The correction is the median over the fold's training laps:
1.00372 for fold 2 (fold 1: 1.00370, fold 3: 1.00373).

| fold 2 | pos RMSE | lateral | longitudinal | heading | < 1 m | max |
|---|---|---|---|---|---|---|
| L5, all fixes (lap restart): before → after | 0.216 → **0.207 m** | 0.185 → 0.180 m | 0.112 → **0.101 m** | 0.214 → 0.213° | 100 → 100 % | 0.88 → 0.77 m |
| L5, 5 s gaps (E-4 blocks 50 %) | 0.560 → **0.501 m** | 0.432 → 0.454 m | 0.356 → **0.211 m** | 0.256 → 0.270° | 88.9 → 90.3 % | 1.98 → 1.84 m |
| L5, 9 s gaps (E-4 blocks 10 %) | 0.725 → **0.672 m** | 0.525 → 0.585 m | 0.499 → **0.331 m** | 0.242 → 0.259° | 80.1 → 81.9 % | 2.16 → 1.84 m |
| **validation L2 + L7 (continuous)** | 0.220 → 0.229 m | 0.180 → 0.196 m | 0.127 → **0.118 m** | 0.215 → 0.209° | 100 → 100 % | 0.82 → 0.84 m |
| L2 | 0.265 → 0.280 m | 0.230 → 0.253 m | 0.132 → 0.121 m | 0.262 → 0.254° | 100 → 100 % | |
| L7 | 0.132 → **0.127 m** | 0.058 → 0.057 m | 0.119 → 0.114 m | 0.119 → 0.122° | 100 → 100 % | |
| L5 (continuous) | 0.204 → 0.205 m | 0.176 → 0.179 m | 0.104 → 0.100 m | 0.209 → 0.209° | 100 → 100 % | |

(Runtime of these runs is not meaningful: E-6 ran at the same time. The correction costs nothing.)

- **The longitudinal error falls everywhere** (−34 to −41 % in 5–9 s gaps, −7 to −10 % with all
  fixes) — the expected effect of removing a constant 0.37 % speed error.
- **But the lateral error rises on L2** (+0.023 m) and in the gaps, so fold 2's validation pair gets
  slightly worse overall (+0.009 m). A 0.37 % speed change cannot move the car sideways by itself; the
  likely path is through the fixes: a different along-track state changes which (biased, off-line)
  fixes are matched and accepted. Whether this is systematic or noise needs more laps: **run on
  folds 1 and 3** (promising for gaps and along-track, mixed on the validation pair).

### E-6 Fix σ growing with the lateral distance from the nearest keyframe

`reloc.sigma.offline_gain` (`camloc/reloc/query.py`), `scripts/exp_e6_offline_sigma.sh`,
`results/experiments/e6_offline_sigma/`. Each fix records its lateral distance d from the nearest of
the keyframes it was matched against (known at run time, no GT); its σ_xy becomes
√(σ_inliers² + (gain · d)²). E-2a: fix errors grow ~0.08–0.15 m per metre off the line.

Gain chosen on the tuning lap L5 (lap restart, fold-2 map): 0 → 0.216 m, 0.05 → 0.212 m, 0.10 →
0.206 m, **0.20 → 0.205 m** (lateral 0.185 → 0.153 m, longitudinal 0.112 → 0.136 m, heading 0.214 →
0.233°; rejected fixes 313 → 168).

| fold 2, gain 0.20 | pos RMSE | lateral | longitudinal | heading | < 1 m | max | sets with a fix | honesty | off-line overshoot share / slope |
|---|---|---|---|---|---|---|---|---|---|
| **validation L2 + L7**: baseline → E-6 | 0.220 → 0.230 m | 0.180 → **0.157 m** | 0.127 → 0.168 m | 0.215 → 0.262° | 100 → 99.8 % | 0.82 → 1.06 m | 93.3 → 95.9 % | 2.26 → **1.71** | 91 → **66 %** / 4.1 → **2.2 %** |
| L2 | 0.265 → 0.278 m | 0.230 → **0.199 m** | 0.132 → 0.195 m | 0.262 → 0.325° | 100 → 99.7 % | | 88.5 → 92.9 % | 2.31 → 1.69 | 92 → 70 % / 4.2 → 2.3 % |
| L7 | 0.132 → 0.137 m | 0.058 → 0.061 m | 0.119 → 0.122 m | 0.119 → 0.128° | 100 → 100 % | | 100 → 100 % | 2.01 → 1.87 | |
| L5 (continuous) | 0.204 → 0.206 m | 0.176 → 0.154 m | 0.104 → 0.137 m | 0.209 → 0.240° | 100 → 100 % | | 97.4 → 97.6 % | 2.45 → 1.83 | 93 → 68 % / 5.4 → 4.0 % |

- **It does what it targets**: lateral error −13 % on the validation pair (−0.031 m on L2), the
  overshoot away from the mapped line halved, the filter more honest (actual error 1.7× its σ
  instead of 2.3×).
- **But along-track (+32 %) and heading (+22 %) get worse**, so the position RMSE rises by 0.010 m.
  The inflation is isotropic: it also throws away the off-line fixes' along-track information (not
  biased) and the position fixes' pull on the heading.
- Next: **E-6b**, inflate σ only across the mapped line, and the combination with E-5 (which removes
  the along-track drift).

**E-6b: only across the mapped line** (`reloc.sigma.offline_mode: lateral`; along the line the fix
keeps its inlier-based σ, `results/experiments/e6_offline_sigma/f2_lateral/`). Gain on L5 (lap
restart): **0.10 → 0.201 m**, 0.20 → 0.217 m, 0.30 → 0.231 m (gain 0: 0.216 m).

| fold 2, gain 0.10 across the line | pos RMSE | lateral | longitudinal | heading | < 1 m | max | sets with a fix | honesty | off-line overshoot share / slope |
|---|---|---|---|---|---|---|---|---|---|
| **validation L2 + L7**: baseline → E-6b | 0.220 → **0.220 m** | 0.180 → **0.168 m** | 0.127 → 0.142 m | 0.215 → 0.226° | 100 → 100 % | 0.82 → 0.80 m | 93.3 → 94.2 % | 2.26 → 2.08 | 91 → 81 % / 4.1 → 3.5 % |
| L2 | 0.265 → 0.265 m | 0.230 → 0.215 m | 0.132 → 0.156 m | 0.262 → 0.277° | 100 → 100 % | | 88.5 → 90.1 % | 2.31 → 2.09 | 92 → 84 % / 4.2 → 3.5 % |
| L7 | 0.132 → 0.135 m | 0.058 → 0.059 m | 0.119 → 0.121 m | 0.119 → 0.124° | 100 → 100 % | | 100 → 100 % | 2.01 → 2.02 | |
| L5 (continuous) | 0.204 → 0.202 m | 0.176 → 0.169 m | 0.104 → 0.112 m | 0.209 → 0.214° | 100 → 100 % | | 97.4 → 97.3 % | 2.45 → 2.22 | 93 → 81 % / 5.4 → 5.0 % |

Runtime 58.6 / 75.4 ms (mean / p95; this run had the machine to itself — the same as the baseline).

- Position unchanged, lateral −7 %, along-track +12 %, heading +5 %: a smaller trade than E-6.
- The along-track loss remains because off-line fixes are noisier in every direction (E-2a), and
  with the larger across-line σ more of them pass the gate (93.3 → 94.2 % of sets) with their
  along-track component fully trusted.

**E-5 + E-6b together** (speed scale 1.00372, gain 0.10 across the line;
`results/experiments/e6_offline_sigma/f2_lateral_e5/`), next to every fold-2 variant
(`results/experiments/f2_variants_report.json`):

| fold 2, validation L2 + L7 | pos RMSE | lateral | longitudinal | heading | < 1 m | max | honesty | L2 / L7 pos RMSE |
|---|---|---|---|---|---|---|---|---|
| baseline (iteration 1) | 0.220 m | 0.180 m | 0.127 m | 0.215° | 100 % | 0.82 m | 2.26 | 0.265 / 0.132 m |
| E-5 speed scale | 0.229 m | 0.196 m | **0.118 m** | **0.209°** | 100 % | 0.84 m | 2.31 | 0.280 / 0.127 m |
| E-6 σ by distance, isotropic (gain 0.20) | 0.230 m | **0.157 m** | 0.168 m | 0.262° | 99.8 % | 1.06 m | **1.71** | 0.278 / 0.137 m |
| E-6b σ by distance, across the line (gain 0.10) | 0.220 m | 0.168 m | 0.142 m | 0.226° | 100 % | **0.80 m** | 2.08 | 0.265 / 0.135 m |
| **E-5 + E-6b** | **0.215 m** | 0.174 m | 0.127 m | 0.232° | 100 % | 0.91 m | 2.04 | **0.260 / 0.129 m** |

Runtime of the combination 58.5 / 75.7 ms (alone on the machine; unchanged).

- **The combination is the best fold-2 variant on position** (−0.005 m) and better on lateral
  (−0.006 m); heading is 0.017° worse. The gains are small — of the size that a different choice of
  accepted fixes produces — so it was **run on folds 1 and 3** (each with its own speed scale) before
  drawing a conclusion. E-6 alone has the best lateral error and the most honest filter, but loses on
  position and heading; it is not extended.

**E-5 + E-6b on all three folds** (`configs/exp/cv_f{1,2,3}_e5_e6b.yaml`, pooled
`results/experiments/cv_score_e5_e6b.json`; speed scale 1.00370 / 1.00372 / 1.00373 from each fold's
training laps, gain 0.10 chosen on fold 2's L5):

| | pos RMSE (25 %) | lateral (20 %) | heading (10 %) | < 1 m (10 %) | longitudinal | median | p95 | max | runtime mean / p95 |
|---|---|---|---|---|---|---|---|---|---|
| **pooled, 6 laps**: iteration 1 → E-5 + E-6b | 0.162 → **0.159 m** | 0.114 → **0.111 m** | 0.148 → 0.158° | 100 → 100 % | 0.114 → 0.113 m | 0.093 → **0.088 m** | 0.330 → **0.317 m** | 0.816 → 0.910 m | 58–62 → 58.5–59.3 ms |
| F1: L1 + L6 | 0.119 → 0.120 m | 0.044 → 0.043 m | 0.079 → 0.079° | 100 → 100 % | 0.111 → 0.112 m | | | 0.378 → 0.385 m | 59.1 / 75.6 ms |
| F2: L2 + L7 | 0.220 → **0.215 m** | 0.180 → **0.174 m** | 0.215 → 0.232° | 100 → 100 % | 0.127 → 0.127 m | | | 0.816 → 0.910 m | 58.5 / 75.7 ms |
| F3: L3 + L4 | 0.123 → **0.118 m** | 0.066 → 0.069 m | 0.114 → 0.122° | 100 → 100 % | 0.103 → **0.096 m** | | | 0.392 → 0.467 m | 59.3 / 75.8 ms |

Per lap (pos RMSE): L1 0.127 → 0.131, L2 0.265 → 0.260, L3 0.106 → 0.100, L4 0.136 → 0.133,
L6 0.104 → 0.100, L7 0.132 → 0.129 m. Tuning lap L5 (continuous): 0.123 → 0.124 (F1), 0.204 →
0.202 (F2, E-6b alone), 0.116 → 0.117 m (F3).

- **A small but consistent gain**: better on 5 of the 6 validation laps (L1, the slowest warm-up lap,
  0.004 m worse), pooled position −2 %, lateral −3 %, median −5 %, p95 −4 %; on the common-line folds
  1 and 3 it comes from the speed scale (E-6b rarely acts there), on fold 2 from both.
- **Costs**: heading +7 % (0.010°) and a larger worst pose on L2 (0.91 m); runtime unchanged.
- **Recommendation for iteration 2: adopt both** (`filter.vx_scale` from the training laps,
  `reloc.sigma.offline_gain: 0.10`, `offline_mode: lateral`), with two caveats for the test run: the
  speed scale is a property of this car's speed signal (stable at 0.33–0.42 % on all nine laps of this
  session, but a different day, tyres or car could change it — an online scale state in the filter
  would be the robust version), and E-6b only acts where the test run leaves the lines of the map.

### E-4 Fewer camera fixes (stand-in 1 for daylight against the night map)

`scripts/exp_e4_fewer_fixes.sh`, `results/experiments/e4_fewer_fixes/f2/` (`e4_summary.json`). The
tuning lap L5 against the fold-2 map (lap restart from GT, as the robustness probes of D-017), with
100 / 50 / 10 % of the fixes reaching the filter (`reloc.stress.keep_fraction`): dropped either for
random image sets (`keep_mode: random`, seed 0), or in blocks — fixes only in the first 50 % / 10 % of
every 10 s, i.e. gaps of 5 s / 9 s (`keep_mode: blocks`), like stretches where nothing matches. Dropped
fixes never reach the filter or the lock-out guard.

| fixes kept | how | pos RMSE | lateral | longitudinal | heading | < 1 m | median | p95 | max | sets with a fix |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 % | — | 0.216 m | 0.185 m | 0.112 m | 0.214° | 100 % | 0.102 m | 0.516 m | 0.88 m | 97.0 % |
| 50 % | random sets | 0.228 m | 0.200 m | 0.109 m | 0.195° | 100 % | 0.112 m | 0.540 m | 0.94 m | 47.3 % |
| 50 % | blocks (5 s gaps) | **0.560 m** | 0.432 m | 0.356 m | 0.256° | 88.9 % | 0.193 m | 1.407 m | 1.98 m | 48.7 % |
| 10 % | random sets | 0.393 m | 0.342 m | 0.193 m | 0.183° | 94.8 % | 0.167 m | 1.061 m | 1.82 m | 9.7 % |
| 10 % | blocks (9 s gaps) | **0.725 m** | 0.525 m | 0.499 m | 0.242° | 80.1 % | 0.336 m | 1.555 m | 2.16 m | 9.5 % |

(The runtimes of these runs, 111–140 ms per set, are not meaningful: other experiments ran at the
same time.)

- **What matters is the length of the gaps, not the number of fixes.** Half the fixes at random cost
  only 0.012 m; the same half in blocks costs 0.34 m. Even 10 % at random (a fix about once per
  second) keeps 95 % of the poses within 1 m; 9 s gaps leave one pose in five more than 1 m off.
- **In daylight, then, the question is how long the stretches are in which nothing matches the night
  map.** If matching fails in places (shadows, glare), the score degrades as in the block rows; if it
  only thins out evenly, as in the random rows.
- The error in the gaps is longitudinal as much as lateral (0.36–0.50 m), matching E-3a: the speed
  scale (−0.37 %) and the yaw rate drive it. Measurements that do not depend on appearance (E-1) and
  better dead reckoning (E-3, speed scale) are what would help here.

**All three folds** (`results/experiments/e4_fewer_fixes/f{1,2,3}/e4_summary.json`; the fold-1 runs at
100 % and 50 % random were made before the batch was interrupted, checked and reused — the runs are
deterministic). L5 is in no map; in the maps of folds 1 and 3 L2 covers its off-line stretch, so only
11 % of L5 is off the mapped line there (30 % in fold 2):

| fixes kept | how | fold 2: pos RMSE / < 1 m | fold 1: pos RMSE / < 1 m | fold 3: pos RMSE / < 1 m |
|---|---|---|---|---|
| 100 % | — | 0.216 m / 100 % | 0.123 m / 100 % | 0.116 m / 100 % |
| 50 % | random sets | 0.228 m / 100 % | 0.136 m / 100 % | 0.127 m / 100 % |
| 50 % | blocks (5 s gaps) | 0.560 m / 88.9 % | 0.399 m / 95.4 % | 0.418 m / 95.0 % |
| 10 % | random sets | 0.393 m / 94.8 % | 0.236 m / 99.4 % | 0.241 m / 99.3 % |
| 10 % | blocks (9 s gaps) | 0.725 m / 80.1 % | 0.615 m / 88.6 % | 0.629 m / 88.2 % |

Lateral / longitudinal / heading for folds 1 and 3: 100 % 0.083 / 0.091 m / 0.125° and 0.081 / 0.082 m
/ 0.124°; 5 s gaps 0.293 / 0.271 m / 0.205° and 0.311 / 0.279 m / 0.206°; 9 s gaps 0.380 / 0.483 m /
0.244° and 0.377 / 0.504 m / 0.250°; maxima 2.07–2.51 m with gaps.

- **The same pattern on every fold**: thinning at random costs little, gaps cost a lot.
- **Gaps add about the same absolute error with or without map coverage of the line** (5 s gaps:
  +0.28 to +0.34 m; 9 s gaps: +0.49 to +0.51 m): the gap penalty is dead-reckoning drift, not map
  coverage. So for daylight, the length of the stretches without a fix decides the score, and
  better dead reckoning helps on every map.

### E-3a Drift budget in a fix gap: how much could visual odometry win?

`scripts/exp_e3_drift_budget.py`, `results/experiments/e3_vo/drift_budget.json`. Dead reckoning from
the GT pose over 2 / 5 / 10 s windows (one start every 0.5 s, speed > 5 m/s) on L5 and the fold-2
training laps L1 L3 L4 L6, with the car-state signals (yaw-rate bias removed with its lap mean), then
with one signal replaced by its GT value — the most a visual-odometry estimate of it could win.

| gap | variant | position median | p90 | lateral RMS | longitudinal RMS |
|---|---|---|---|---|---|
| 2 s | car signals | 0.212 m | 0.478 m | 0.186 m | 0.232 m |
| 2 s | GT yaw rate | 0.201 m | 0.421 m | 0.130 m | 0.227 m |
| 2 s | GT vy | 0.156 m | 0.351 m | **0.050 m** | 0.214 m |
| 2 s | GT vx | 0.076 m | 0.369 m | 0.204 m | **0.046 m** |
| 5 s | car signals | 0.486 m | 1.176 m | 0.430 m | 0.579 m |
| 5 s | GT yaw rate | 0.432 m | 0.903 m | **0.196 m** | 0.528 m |
| 5 s | GT vy | 0.392 m | 0.874 m | 0.253 m | 0.479 m |
| 5 s | GT vx | 0.210 m | 0.939 m | 0.505 m | **0.230 m** |
| 10 s | car signals | 0.864 m | 1.997 m | 0.744 m | 0.975 m |
| 10 s | GT yaw rate | 0.805 m | 1.480 m | **0.335 m** | 0.899 m |
| 10 s | GT vy | 0.770 m | 1.641 m | 0.665 m | 0.790 m |
| 10 s | GT vx | 0.484 m | 1.650 m | 0.882 m | **0.394 m** |

Car-state signals against GT (speed > 5 m/s): gyro bias −0.00024 to −0.00035 rad/s, yaw-rate error
after the bias 0.0035–0.0054 rad/s RMS, lateral-velocity error −0.012 to −0.026 m/s mean and
0.08–0.13 m/s RMS, **forward speed 0.36–0.40 % too low on every lap**.

- **Lateral drift has two comparable sources**: the lateral-velocity error dominates short gaps (a
  perfect vy cuts 2 s drift by 73 %), the yaw-rate error long ones (a perfect yaw rate halves 5–10 s
  drift). Visual odometry can win at most this much, and only if it is more accurate than the car
  signals; its rotation (well determined from distant points) is the promising part.
- **Side finding — speed scale**: the constant −0.37 % speed error causes most of the longitudinal
  drift (5 s: 0.58 m, with the GT speed 0.23 m). A speed-scale correction (a constant from the
  training laps, or a scale state in the filter) is a cheap fix outside E-1…E-4; not tested yet.
- Decision: build E-3 for the **rotation** (relative heading between frames); slip only if time allows.

### E-3b Accuracy of the visual-odometry heading change

`camloc/vo.py`, `scripts/exp_e3_vo_accuracy.py`, `results/experiments/e3_vo/vo_accuracy.json`. For every
pair of consecutive frames of L5 (10 Hz): the pipeline's SIFT features (car-body masked) of the two
frames are matched, an essential matrix is fitted with RANSAC (1 px), and the rotation's yaw in the
base frame is compared with the GT heading change; next to it the car's yaw rate (bias removed)
integrated over the same 0.1 s. Over 5 s the per-frame increments are summed, as in a fix gap.

| camera | pairs with a VO rotation | inliers | median error per frame | frames > 0.15° off | mean error per frame | heading error over 5 s: VO / VO gated by the car signal / blend / car signal |
|---|---|---|---|---|---|---|
| `fr` (forward) | 93.3 % | 87 | 0.048° | 19.7 % | −0.018° | 7.4° / 0.87° / 0.244° / **0.226°** |
| `cl` | 81.8 % | 80 | 0.308° | 68.6 % | +0.029° | 12.7° / 0.45° / 0.231° / 0.226° |
| `cr` | 95.3 % | 78 | 0.125° | 44.7 % | −0.130° | 14.6° / 0.62° / 0.230° / 0.226° |

The car's yaw rate is off by 0.009° per frame (RMS) but drifts 0.23° over 5 s (a slowly varying error,
E-3a); "gated" = VO where it agrees with the car signal within 0.15°, else the car signal; "blend" =
5 % VO, 95 % car signal per frame.

- **One camera's frame-to-frame rotation is biased**: besides 20–70 % gross outliers, the good frames
  carry a constant error of 0.02–0.13° per frame (the rotation–translation ambiguity of forward motion
  seen by one camera with most structure to one side), which sums to 0.9° over 5 s — four times the
  car signal's drift. No mix of the two beats the car signal alone in this test.
- To beat the gyro, the rotation would need far more inliers spread over the whole view (several
  cameras together, longer feature tracks over many frames, or rotation against distant map points).

### E-3c Visual-odometry heading changes in the filter

`vo.enabled`, `camloc/ekf.py` (`clone_yaw`, `update_relative_yaw`: the yaw at the previous frame is
kept with its cross-covariance, so the relative measurement is fused with the right correlation),
`scripts/exp_e3c_vo_filter.sh`, `results/experiments/e3_vo/f2/` (`vo_e4_summary.json`). Camera `fr`,
σ = 0.06° per frame pair (the RMS of its good frames in E-3b), gate χ²(1) 99.9 %. L5 against the
fold-2 map, all fixes and the E-4 block patterns, next to the same runs without VO (E-4):

| fixes | VO | pos RMSE | lateral | longitudinal | heading | < 1 m | p95 | max | VO pairs accepted |
|---|---|---|---|---|---|---|---|---|---|
| all | no | 0.216 m | 0.185 m | 0.112 m | 0.214° | 100 % | 0.516 m | 0.88 m | — |
| all | yes | 0.226 m | 0.201 m | 0.102 m | 0.239° | 100 % | 0.539 m | 0.93 m | 87 % |
| 50 % in blocks (5 s gaps) | no | 0.560 m | 0.432 m | 0.356 m | 0.256° | 88.9 % | 1.407 m | 1.98 m | — |
| 50 % in blocks (5 s gaps) | yes | **1.107 m** | 0.807 m | 0.757 m | 0.700° | 81.7 % | 2.532 m | 5.63 m | 87 % |
| 10 % in blocks (9 s gaps) | no | 0.725 m | 0.525 m | 0.499 m | 0.242° | 80.1 % | 1.555 m | 2.16 m | — |
| 10 % in blocks (9 s gaps) | yes | **1.404 m** | 0.983 m | 1.003 m | 0.679° | 72.6 % | 3.010 m | 5.81 m | 87 % |

- **VO makes it worse, most of all in the gaps it should bridge** (heading error ×2.7–2.8). The
  per-frame bias of E-3b passes the gate (each increment is only slightly off) and accumulates.
- Runtime with VO: 122–125 ms per image set (L5 alone on the machine), over the 100 ms budget —
  the per-frame matching for VO costs ~50 ms on top.
- **Decision: E-3 does not show promise in this form**: no fold-2 continuous run, not run on the
  other folds. What could still work: a rotation estimate from many cameras at once or over longer
  feature tracks — or simply a better yaw-rate signal, which is what the E-3a ceiling is about.

### E-1 Lateral position from the track borders

`camloc/borders.py`, `scripts/exp_e1_border_overlay.py`, `scripts/exp_e1_learn_borders.py`,
`scripts/exp_e1_border_accuracy.py`, `results/experiments/e1_border/`. The 3D track borders
(`track_bounds`, an allowed input) are projected into the full-resolution images of `fr`, `cl`, `cr`
at the predicted pose; along each projected point's image normal the outward dark-to-bright edge
(asphalt → white line or wall) is searched, and a robust fit of the lateral offset and heading —
with pitch, roll and height as nuisance parameters with priors — gives one measurement per camera.
Component test: L5 frames with the GT pose shifted sideways by 0 / ±0.3 / ±0.6 m and turned by
N(0, 0.15°) as prior; the measurement should undo the shift.

| step | what was found / changed | availability | lateral error after the correction |
|---|---|---|---|
| 1 | overlay at the GT pose (`overlay/`): the borders run along visible edges (white line, wall base) | — | — |
| 2 | first fit (lateral, heading only; strongest edge) | 11 % | 0.96 m RMS |
| 3 | + pitch, roll, height as nuisance parameters | 16 % | 0.59 m |
| 4 | debug: on the right of the main straight `track_bounds` lies 0.5–1 m beyond the white line, in the run-off → **visible edge learned from the training laps** (`visual_borders.npz`: 52 % of 16,531 border points; offset to `track_bounds` p10 / median / p90 −1.86 / −0.20 / +1.56 m) | 18 % | 0.62 m |
| 5 | the camera is 0.34 m above the road, so the borders lie near the horizon, where a sideways shift looks like a pitch change (a 1.5° pitch error moves a road point 4 m away by 1.2 m) → pitch from the **road slope** (fold-2 training laps: GT pitch 0.76° → residual 0.25°; gain 0.90), roll from the lateral acceleration (0.84° → 0.70°) | 13 % | 0.64 m |
| 6 | edge ambiguity (white line, kerb stripes, barrier base, run-off edge, shadows): up to 3 candidate edges per point and RANSAC for the consensus | 15 % | 0.70 m (55 % of them > 0.3 m off) |

At the GT pose (no shift) the detected edges have no constant per-camera offset (medians within
±2 px), but they scatter by 2–4 px within a frame and only 21–59 % of the points lie within 3 px of
the projected edge.

- **Decision: not put in the filter, no fold-2 run.** A measurement that is available 15 % of the time
  and 0.7 m off is worse than the off-line fixes it should correct (0.3–0.7 m).
- **Why it fails here**: the borders are not a clean, unique edge (painted lines, kerbs, walls and
  run-off edges compete, and `track_bounds` is not on the paint everywhere), and from 0.34 m height a
  border near the horizon constrains the lateral position only if pitch is known to ~0.1–0.2°.
- **What it would take**: a dedicated line detector (white-line ridge with its width, or learned
  segmentation), tracking each border over consecutive frames instead of detecting it per frame, and
  fitting left and right borders of several cameras jointly so that pitch, roll and height are
  determined by the geometry instead of priors. Roughly 1–2 days; still the only candidate here
  that would not depend on appearance at all (daylight).
- Useful by-products: the learned visible edge, and pitch ≈ 0.9 × road slope from `track_bounds`.

### E-2a Is the off-line overshoot caused by the depth of the map points?

`scripts/exp_e2a_depth_check.sh`, `scripts/exp_offline_bias.py`, `results/experiments/e2a_depth/`.
Cross-lap triangulation (idea 2) only helps if the overshoot comes from badly determined point depth,
so that was tested first, without a rebuild: every camera of every second image set of L5 against the
fold-2 map, with the **GT pose as prior** (`scripts/reloc_diag.py`), once with all map points and once
with only points triangulated with ≥ 2° parallax (`reloc.query.min_point_parallax_deg`; 31 % of the
`fr` points and 63–75 % of the side cameras' points pass). Overshoot = fix error projected on the
direction away from the mapped line.

| distance off the mapped line | attempts | fix rate (all / ≥ 2°) | median fix error | mean overshoot | share pointing away |
|---|---|---|---|---|---|
| 0–1 m | 2,505 | 93.5 / 82.8 % | 0.144 / 0.152 m | +0.009 / +0.008 m | 52 / 51 % |
| 1–3 m | 530 | 83.4 / 72.5 % | 0.339 / 0.342 m | +0.086 / +0.088 m | 69 / 66 % |
| 3–6 m | 325 | 71.7 / 63.4 % | 0.528 / 0.497 m | +0.067 / +0.168 m | 71 / 73 % |
| 6–15 m | 235 | 40.0 / 31.1 % | 0.736 / 0.687 m | +0.307 / +0.442 m | 74 / 86 % |

Overshoot as a share of the offset, per camera (all points / ≥ 2°): `cl` +10.6 / +9.1 %, `rr` +10.1 /
+12.0 %, `rl` +3.1 / +4.7 %, `cr` −1.2 / +0.2 %, `fr` −14.7 / −11.5 %; all cameras +3.0 / +4.9 %.
`fr` also has 62.5 % of its fixes wrong by > 3 m at 6–15 m off the line (0 % with ≥ 2° points, but
then only 2 % of its attempts give a fix there).

- **The overshoot is in the raw fixes**, with a perfect prior: it comes from the map or the pose
  solution, not from the filter.
- **It does not shrink with high-parallax points**, so it is not caused by imprecise point depth.
  Triangulating across laps would mainly make depth more precise. **Decision: E-2b (the cross-lap
  map) is not built**; it would cost ~2 h and ~0.5 GB without addressing the cause.
- **It is camera-specific** (−15 % to +11 % of the offset, opposite signs). That fits per-camera
  geometry errors (intrinsics / extrinsics) that cancel on the mapped line — map and query share the
  viewpoint — and show when the viewpoint moves. The filter partly averages them (+3 % net, +4 % in
  the baseline run).
- **Off the line the fixes also get worse and rarer** (median error 0.14 → 0.74 m, fix rate 94 →
  40 %), while their stated σ depends only on the inlier count.
- Better-aimed follow-ups (not run): per-camera calibration refinement from views on different lines,
  and a fix σ that grows with the distance from the nearest keyframe (known at run time, no GT).

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
| `camloc/borders.py`, `camloc/vo.py` | experiments E-1 (border measurement) and E-3 (visual odometry); off by default |
| `configs/iter1.yaml` | all parameters of iteration 1, split A |
| `configs/iter1b.yaml` | split B: same parameters, L0 held out (pit-lane stress test) |
| `configs/cv_f1.yaml`–`cv_f3.yaml` | the cross-validation folds (D-021); `cv_f1_swapped`, `cv_f2_swapped` (D-022); `cv_f*_untuned` (D-023) |
| `configs/final.yaml` | the hidden test run: map from all laps |
| `scripts/` | entry points (see above); `scripts/exp_*` run and report the experiments (Experiments section) |
| `results/laps.csv` | every lap: ns / UTC / local timestamps, kind, split role (B), image sets, speed metrics |
| `results/cv/` | cross-validation: per-fold runs, scores and plots, pooled `cv_score.json` |
| `results/experiments/` | experiments E-1 … E-4 (fold 2; E-4 on all folds); `_aborted/` keeps the log of the stopped all-folds batch |
| `results/iter1b/` | split B: continuous runs (with cameras, odometry-only ablations), scores, plots |
| `results/iter1/` | split A: continuous runs, lap-restart tuning runs, robustness probes, s/n analysis, diagnostics |
| `artifacts/` | derived data: caches, masks, maps (rebuilt by `run.sh`) |
| `RUN_INSTRUCTIONS.md` | setup, the test run, validation |
| `docs/approach_comparison.md` | this pipeline vs the `main` branch (filters, matching, tuning), a SIFT fallback for `main`, and `main`'s ground-truth use at inference |
