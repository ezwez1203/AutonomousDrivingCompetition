# carla_autodrive

CARLA simulation project for the 2026 National University AI Autonomous Driving Competition.

The project is still simulator-first. A few hardware assumptions come from the 2026-06-05 purchase-list photo, so treat them as a practical baseline rather than final electrical documentation.

## Project Layout

```text
carla_autodrive/
├── config/         # YAML parameters: sim, sensors, track, hardware
├── sensors/        # RGB camera, LiDAR, and radar wrappers
├── perception/     # Perception modules
├── control/        # Pure Pursuit, PID, route following, hardware clamps
├── missions/       # Obstacle and parking route builders
├── state_machine/  # Mission FSM
├── simulator/      # Scoring, events, and Phase 6 reporting
├── utils/          # Logging, config loading, CARLA sessions
├── maps/           # Custom OpenDRIVE track generation
└── scripts/        # CLI entry points
```

## Environment

- conda env: `carla` on Python 3.12
- CARLA Python API 0.9.16 plus numpy and pyyaml

```bash
conda activate carla
pip install -r requirements.txt
```

## CARLA Server

This code connects to a running CARLA server through the Python client. If your local CARLA folder only has `Maps` and no `CarlaUE4.sh`, download and extract the CARLA 0.9.16 package release first.

```bash
./CarlaUE4.sh -quality-level=Low -RenderOffScreen
```

## Phase 0: Basic Connection And Sensors

```bash
conda activate carla
cd "$PROJECT_ROOT"

python -m carla_autodrive.scripts.check_connection

python -m carla_autodrive.scripts.phase0_spawn_sensors
python -m carla_autodrive.scripts.phase0_spawn_sensors --map Town01 --duration 30 --autopilot
python -m carla_autodrive.scripts.phase0_spawn_sensors --duration 10 --throttle 0.2 --steer 0.1 --brake 0.0
```

Example log shape:

```text
[12:00:01] INFO carla_autodrive: vehicle spawned: vehicle.tesla.model3 (id=42) @ spawn[0]
[12:00:01] INFO carla_autodrive: sensor attached: rgb_camera (sensor.camera.rgb, id=43)
[12:00:01] INFO carla_autodrive: sensor attached: lidar (sensor.lidar.ray_cast, id=44)
```

## Phase 1: Custom Track

The competition track is generated as OpenDRIVE from `config/track.yaml`.

```bash
python -m carla_autodrive.scripts.build_track
python -m carla_autodrive.scripts.build_track --load --timeout 120

python -m carla_autodrive.scripts.phase0_spawn_sensors --duration 8 --throttle 0.05
python -m carla_autodrive.scripts.phase1_draw_elements --duration 60
python -m carla_autodrive.scripts.phase1_draw_elements --duration 30 --spawn-obstacles --obstacle2 0 --obstacle3 0
python -m carla_autodrive.scripts.phase1_place_actors --duration 60
python -m carla_autodrive.scripts.phase1_place_actors --duration 2 --tick 0.5 --spawn-test-vehicle-zone start_line
python -m carla_autodrive.scripts.phase1_traffic_light_control --duration 10 --state red
python -m carla_autodrive.scripts.phase1_traffic_light_control --duration 20 --cycle --red-time 5 --green-time 5 --yellow-time 2
```

Track utilities:

```bash
python -m carla_autodrive.scripts.measure_track_accuracy --mask nonwhite --component largest
python -m carla_autodrive.scripts.fit_track_from_blueprint --iterations 120 --samples 360 --control-points 24 --learning-rate 0.18 --smooth 0.35 --coverage-weight 0.05 --max-step-px 8
python -m carla_autodrive.scripts.fit_track_from_blueprint --iterations 120 --samples 360 --control-points 24 --learning-rate 0.18 --smooth 0.35 --coverage-weight 0.05 --max-step-px 8 --apply
python -m carla_autodrive.scripts.export_unreal_bake
```

Current track assumptions:

- Track dimensions live in `config/track.yaml`.
- There are two lanes: lane 1 is inner, lane 2 is outer.
- CARLA uses a scaled version of the real millimeter layout, currently around 10x, which makes physics less fragile.
- The fitted track is image-derived, so it is good enough for control work but should still be treated as an approximation until CAD/DXF data is available.

## Real-Car Hardware Baseline

`config/vehicle_hardware.yaml` records the parts visible in the 2026-06-05 purchase-list photo. The list appears to cover two cars: two kids electric cars, two Arduino Mega boards, six motor drivers, two batteries, four cameras, twelve ultrasonic sensors, two RPLIDAR A1M8 units, and two 12V SMPS units.

The current simulator model is a little simplified:

- one front RGB camera is used by the active perception pipeline,
- one RPLIDAR A1M8 is approximated as a 2D 360-degree LiDAR,
- twelve ultrasonic sensors are approximated with short-range CARLA radar sensors,
- motor-driver PWM/current limits still need a bench check before real-car driving.

## Phase 2: Sensor Stack

```bash
python -m carla_autodrive.scripts.phase2_sensor_stack --duration 10 --no-save
python -m carla_autodrive.scripts.phase2_sensor_stack --duration 20 --save-every 20
python -m carla_autodrive.scripts.phase2_sensor_stack --duration 20 --save-dir carla_autodrive/reports/my_sensor_run
```

The `SensorStack` returns a `PerceptionInput` snapshot with:

- `camera_bgra`: RGB camera data in BGRA array form,
- `lidar_points`: vehicle-frame `(x, y, z, intensity)` points,
- `radar_points`: fused vehicle-frame radar detections,
- `radar_by_name`: per-sensor radar arrays for the twelve ultrasonic stand-ins.

## Phase 3: Perception

```bash
conda run -n carla python -m carla_autodrive.scripts.phase3_perception_demo --duration 10
conda run -n carla python -m carla_autodrive.scripts.phase3_perception_demo --duration 5 --no-radar
```

`PerceptionPipeline.process(snapshot)` returns lane, traffic-light, obstacle, and parking-line observations. The current lane detector is a lightweight threshold-based baseline. It is useful for smoke tests, but the real track camera feed will probably need tuning or a trained model.

## Phase 4: Control

```bash
conda run -n carla python -m carla_autodrive.scripts.phase4_control_demo \
  --ticks 1800 --target-speed 2.0 --no-perception --route-source track

conda run -n carla python -m carla_autodrive.scripts.phase4_control_demo \
  --ticks 1800 --target-speed 2.0 --no-perception --route-source track \
  --curve-max-lat-acc 0.45 --curve-min-speed 1.2

conda run -n carla python -m carla_autodrive.scripts.phase4_control_demo \
  --ticks 1200 --target-speed 2.0 --no-perception --route-source track \
  --avoid-obstacles --obstacle2 1 --obstacle3 2 --spawn-preset-obstacles

conda run -n carla python -m carla_autodrive.scripts.phase4_control_demo \
  --ticks 2200 --target-speed 2.0 --no-perception --route-source track \
  --parking-maneuver --parking-zone 2 --reverse-parking
```

Useful knobs:

- `--target-speed`: base route speed,
- `--curve-max-lat-acc`: curve speed cap aggressiveness,
- `--curve-lookahead`: how early the controller sees upcoming curves,
- `--steer-speed-gain`: how much steering demand cuts speed,
- `--brake-overspeed-margin`: how far above target speed the controller waits before braking.

## Phase 5: Mission Runner

```bash
conda run -n carla python -m carla_autodrive.scripts.phase5_mission_runner \
  --mission time_trial --ticks 7000 --target-speed 4.8 --curve-max-lat-acc 0.80 \
  --curve-lookahead 4.0 --steer-speed-gain 1.6 --lane-corridor-scoring --no-perception

conda run -n carla python -m carla_autodrive.scripts.phase5_mission_runner \
  --mission obstacle_signal --ticks 3000 --target-speed 2.0 \
  --obstacle2 1 --obstacle3 2 --spawn-preset-obstacles --green-after-sec 3.0

conda run -n carla python -m carla_autodrive.scripts.phase5_mission_runner \
  --mission parking --ticks 2200 --target-speed 2.0 --parking-zone 2 --reverse-parking
```

Time-trial lane scoring now uses the lane corridor by default rather than raw route CTE. The route itself is still the lane-2 virtual line between the center dashed line and the outside solid line, so the car can move within that right-hand lane without collecting false lane-intrusion penalties. The raw CTE is still reported because it remains useful for tuning.

For obstacle/signal missions, lane penalties are suppressed while the FSM is in `OBSTACLE_AVOID`, since that section is allowed to use the adjacent lane for avoidance. Use `--no-lane-corridor-scoring` only when comparing against older reports that used the legacy raw-CTE proxy.

## Phase 6: Scoring, Reports, And Sweeps

Phase 5 can write one JSON summary and one per-tick CSV. These reports include timing, speed, CTE, heading, FSM state, control reason, collision count, lane-corridor intrusion/departure events, stop-violation proxy, and parking-hold ticks.

```bash
conda run -n carla python -m carla_autodrive.scripts.phase5_mission_runner \
  --mission time_trial --ticks 7000 --target-speed 4.8 --curve-max-lat-acc 0.80 \
  --curve-lookahead 4.0 --steer-speed-gain 1.6 --lane-corridor-scoring --no-perception \
  --report-path carla_autodrive/reports/time_trial.json \
  --csv-path carla_autodrive/reports/time_trial.ticks.csv
```

Parameter sweep:

```bash
conda run -n carla python -m carla_autodrive.scripts.phase6_test_runner \
  --mission time_trial \
  --ticks 7000 \
  --target-speeds 4.8,5.0,5.2,5.4 \
  --curve-max-lat-accs 0.80,0.85,0.90 \
  --curve-lookaheads 3.0,3.5,4.0 \
  --steer-speed-gains 1.4,1.6,1.8 \
  --rank-objective time \
  --lane-corridor-scoring \
  --no-perception \
  --no-auto-load-track-map \
  --out-dir carla_autodrive/reports/phase6_time_trial_corridor_150_attack
```

Outputs:

- `summary.csv`: all run metrics in one table,
- `best_run.json`: best run ranked by completion, penalty score, sim time, and mean CTE.

Current time-trial baseline from `phase6_time_trial_corridor_sub170`, confirmed with repeated single-run checks:

```text
target_speed_mps: 4.8
curve_max_lat_acc: 0.80
curve_lookahead: 4.0
steer_speed_gain: 1.6
sim_time_s: 153.20
lane_intrusion_ticks: 0
lane_departure_ticks: 0
collision_count: 0
score: 0.11
```

This is the current time-trial race baseline. It has been reproduced in repeated single-run checks, but any faster setup still needs the same confirmation rule: prefer the fastest run with `collision_count=0`, `lane_departure_ticks=0`, and `lane_intrusion_ticks=0`.

## Config Files

- `config/sim.yaml`: CARLA host/port, map, sync mode, vehicle defaults,
- `config/sensors.yaml`: camera, LiDAR, and radar settings,
- `config/track.yaml`: track shape and mission elements,
- `config/hardware_limits.yaml`: final throttle/brake/reverse clamps,
- `config/vehicle_hardware.yaml`: photo-derived hardware BOM, with a few assumptions called out.
