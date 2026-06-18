# Phase 0-6 Progress Check

Check date: 2026-06-12  
Scope: compare the Phase 0-6 plan in `CARLA_AutoDriving_ProjectPlan.md` with the current `carla_autodrive/` implementation.

## Summary

| Phase | Planned goal | Current status | Estimated progress |
|---|---|---:|---:|
| Phase 0 | CARLA setup, vehicle spawn, basic sensors/control | Connection, sensors, and manual-control smoke tests passed | 100% |
| Phase 1 | Build the competition track environment | Blueprint-derived track fitting, OpenDRIVE generation, and CARLA load validation are in place | 100% |
| Phase 2 | Build the sensor stack | RGB, RPLIDAR A1M8 approximation, 12 ultrasonic approximations, transforms, standard inputs, and recording scripts are implemented | 100% |
| Phase 3 | Build perception | Minimal pipeline, synthetic dataset recorder, auto labeler, and baseline training smoke tests are implemented | 45% |
| Phase 4 | Build control and mission logic | Pure Pursuit + PID, long-run tuning, obstacle routes, parking route, and hardware clamps are implemented | 68% |
| Phase 5 | Build mission FSM | Basic FSM and runner are implemented; time trial, obstacle/signal, and parking validations have run | 55% |
| Phase 6 | Build scoring and lap-time optimization tools | Event-based scorer, JSON/CSV reports, parameter runner, and best-run selection are implemented | 45% |

The rough average progress is about 73%. The main remaining work is speed optimization for the time trial, tighter mission completion behavior, and broader Phase 6 server-side sweeps.

## Scoring Strategy

Avoiding penalties is the baseline. To rank well, the car needs a faster two-lap time trial. A full end-to-end driving model is probably not the best use of time here. The more practical route is to keep the modular stack and add a better racing line, curvature-based velocity planning, lookahead/gain scheduling, and automated repeated evaluation.

Phase 6 should be treated as an evaluation runner, not just a penalty calculator. It should keep logging lap time, cross-track error, lane margin, speed profile, control reason, collisions, stop behavior, parking holds, and completion state. Those outputs can then drive sweeps over target speed, lookahead, curve speed caps, braking distance, and steering gain.

The 2026-06-05 parts-list photo is reflected in `carla_autodrive/config/vehicle_hardware.yaml`: two kids electric cars, two Arduino Megas, six motor drivers, two batteries, four cameras, twelve ultrasonic sensors, two RPLIDAR A1M8 units, and two SMPS units. The competition rules fix the SMPS at 12.0 V and disallow voltage boosting. The actual motor-driver PWM range, Arduino command protocol, current limits, and protection thresholds are still not specified, so the code keeps conservative final output clamps until bench measurements confirm the real numbers.

## Evidence

- CARLA connection validation passed with server/client 0.9.16, map `Carla/Maps/Town10HD_Opt`, and 155 spawn points.
- Phase 0 sensor/control smoke tests received RGB, LiDAR, and radar frames and exited cleanly.
- Phase 2 ran with one vehicle plus RGB, LiDAR, and 12 radar-based ultrasonic approximations.
- Phase 5 produced JSON/CSV Phase 6 reports with collision, lane, stop, and parking event fields.
- The workspace is not a Git repository, so progress is estimated from code, config, scripts, outputs, and CARLA runs rather than commit history.

## Phase Notes

Phase 0 is complete. `CarlaSession` handles client/world setup, synchronous mode, actor cleanup, and vehicle spawn fallback. `phase0_spawn_sensors.py` covers vehicle spawn, sensor attachment, fixed throttle, manual control, and Traffic Manager autopilot.

Phase 1 is complete enough for the current simulator workflow. `track.yaml`, `TrackSpec`, `opendrive_gen.py`, and `build_track.py` generate and validate the OpenDRIVE track. Runtime helpers can draw mission elements, spawn obstacle actors, place prop actors, monitor trigger zones, and control the OpenDRIVE traffic light actor. The fitted track loads in CARLA, although its drawing-match accuracy is still only approximate.

Phase 2 is complete for the current stack. `SensorStack` expands the parts-list configuration into RGB, RPLIDAR-like LiDAR, and 12 ultrasonic approximations. `PerceptionInput` standardizes camera, LiDAR, radar, and per-sensor metadata for Phase 3.

Phase 3 has the minimum perception path and training scaffolding. Lane detection, obstacle clustering, synthetic labeling, dataset recording, and baseline training scripts exist. The weak area is real positive data for traffic lights and obstacles, plus a final inference bridge from trained models back into `PerceptionPipeline`.

Phase 4 drives the custom track with Pure Pursuit, PID speed control, curve speed caps, hardware clamps, obstacle-avoidance presets, and parking maneuvers. Long-run validation is working, but faster racing-line behavior and real perception-driven obstacle replanning are still open.

Phase 5 has the core mission state machine. Time trial completes two laps, but the known 302.1 s run is over the 4-minute limit. Obstacle/signal and parking validation runs complete under the current simplified assumptions. Traffic-light color perception and the full parking exit route still need work.

Phase 6 now has event monitors, reports, scoring, and sweep tooling. The next step is to run longer parameter sweeps on the CARLA server and promote the best settings back into the defaults.

## Immediate Next Work

1. Run the time-trial sweep with perception off and map already loaded.
2. Compare `summary.csv` and `best_run.json` for fastest complete runs with acceptable CTE and no event penalties.
3. Test the best speed/lookahead/gain set for at least one full two-lap run.
4. Add a more aggressive racing line or route smoothing if the current centerline route cannot get under four minutes.
5. Recheck hardware clamps after motor-driver and Arduino bench measurements.
