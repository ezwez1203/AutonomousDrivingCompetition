# CARLA Autonomous Driving Project Plan

This plan is tuned for the 2026 university autonomous-driving competition and the current CARLA 0.9.16 workspace. It is intentionally pragmatic: build the smallest reliable simulator stack first, then use repeated CARLA runs to tune lap time and mission behavior.

## Goal

Build a CARLA-based development environment for a 1/5-scale autonomous electric car. The simulator should cover the competition track, RGB camera, RPLIDAR-like LiDAR, ultrasonic-style near-field sensing, mission logic, scoring, and transfer constraints for the real vehicle.

## Architecture

```text
CARLA world
  -> sensor stack
  -> perception
  -> mission FSM
  -> controller
  -> hardware limiter
  -> CARLA VehicleControl or real Arduino command
```

The simulator and real car should share the same high-level interfaces wherever possible. CARLA is used for fast iteration and validation; the real car still needs bench calibration for motor response, steering response, current limits, sensor placement, and lighting.

## Phase 0: Environment And Smoke Tests

Objectives:

- Run CARLA 0.9.16 and connect with the Python client.
- Spawn a vehicle.
- Attach RGB, LiDAR, and radar sensors.
- Apply basic throttle, steering, braking, and optional Traffic Manager autopilot.
- Confirm frame reception and clean actor teardown.

Implemented scripts:

- `carla_autodrive.scripts.check_connection`
- `carla_autodrive.scripts.phase0_spawn_sensors`

Completion criterion: a short run prints sensor summaries and exits without orphaned actors.

## Phase 1: Track Environment

Objectives:

- Convert the competition drawing into a usable closed track.
- Generate OpenDRIVE from `config/track.yaml`.
- Load the OpenDRIVE world into CARLA.
- Place start lines, lap timers, crosswalk, parking zones, traffic light, and obstacle candidates.
- Spawn smoothed runtime mesh lane/crosswalk markings and provide optional actor placement.

Implemented components:

- `maps/track_spec.py`
- `maps/opendrive_gen.py`
- `scripts/build_track.py`
- `scripts/runtime_mesh_markings.py`
- `scripts/phase1_draw_elements.py`
- `scripts/phase1_place_actors.py`
- `scripts/phase1_traffic_light_control.py`
- `scripts/fit_track_from_blueprint.py`
- `scripts/measure_track_accuracy.py`

The current track is derived from the available raster blueprint. It is good enough for control development, but it is still an approximation, not a perfect CAD import.

## Phase 2: Sensor Stack

Real parts reflected in the simulator:

- RGB cameras
- RPLIDAR A1M8 approximated by a low-channel 360-degree CARLA LiDAR
- Twelve ultrasonic sensors approximated by short-range CARLA radar sensors
- Sensor transforms and frame metadata

Implemented components:

- `sensors/base.py`
- `sensors/camera.py`
- `sensors/lidar.py`
- `sensors/radar.py`
- `sensors/stack.py`
- `sensors/calibration.py`
- `sensors/frames.py`
- `sensors/recording.py`
- `scripts/phase2_sensor_stack.py`

Completion criterion: one vehicle spawns with RGB, LiDAR, and 12 ultrasonic approximations, then produces usable frame summaries.

## Phase 3: Perception

Initial approach:

- Use simple lane-color thresholding and geometry as the first lane signal.
- Use LiDAR ROI filtering and clustering for obstacles.
- Use radar as near-field support, especially for ultrasonic-like behavior.
- Record synthetic datasets from CARLA with labels for lane offset, signal state, and obstacle pose.
- Train small baseline models only after positive samples are available.

Implemented components:

- `perception/types.py`
- `perception/lane.py`
- `perception/obstacles.py`
- `perception/parking.py`
- `perception/pipeline.py`
- `perception/dataset.py`
- `learning/phase3_baselines.py`
- `scripts/phase3_perception_demo.py`
- `scripts/phase3_dataset_recorder.py`
- `scripts/train_phase3_baselines.py`

Remaining work:

- Collect enough real positive samples for traffic lights and obstacles.
- Connect trained traffic-light and obstacle models into runtime inference.
- Tune parking-line perception on the actual visible parking assets.

## Phase 4: Control

Current controller:

- Route-based Pure Pursuit steering
- PID speed control
- Curvature speed caps
- Endpoint slowdown/stop behavior for finite routes
- Reverse gear support for parking
- Final command limiting from `hardware_limits.yaml`

Implemented components:

- `control/types.py`
- `control/pid.py`
- `control/pure_pursuit.py`
- `control/route_following.py`
- `control/vehicle_controller.py`
- `control/hardware_limits.py`
- `missions/obstacle_avoidance.py`
- `missions/parking.py`
- `scripts/phase4_control_demo.py`

Known gaps:

- The time-trial path still follows a conservative route.
- Obstacle avoidance is mostly preset-route based.
- Parking works as a route/control structure, but final visual alignment still needs real validation.

## Phase 5: Mission State Machine

Mission modes:

- `time_trial`
- `obstacle_signal`
- `parking`

FSM responsibilities:

- Count time-trial laps.
- Switch through obstacle avoidance, signal stop/wait/go, and finish states.
- Handle parking approach, reverse, hold, and completion states.
- Feed the controller with the correct route and target speed.

Implemented components:

- `state_machine/types.py`
- `state_machine/fsm.py`
- `scripts/phase5_mission_runner.py`

Known status:

- Time trial can complete two laps, but the known run is still too slow for the four-minute limit.
- Obstacle/signal flow works under simplified signal assumptions.
- Parking hold works, while full competition-style exit behavior still needs tightening.

## Phase 6: Scoring And Optimization

The Phase 6 runner should record enough information to explain every run, not just print a final score.

Implemented components:

- `simulator/events.py`
- `simulator/scoring.py`
- Phase 6 JSON and CSV output from `phase5_mission_runner.py`
- `scripts/phase6_test_runner.py`

Tracked events include:

- Collision count and collision impulse
- Lane intrusion/departure ticks
- Stop-required and stop-violation ticks
- Parking hold ticks
- Minimum stop speed
- Completion state and elapsed simulated time

Optimization parameters:

- Target speed
- Curve max lateral acceleration
- Curve lookahead
- Steering speed gain
- Perception on/off
- Mission-specific route settings

The first practical goal is a complete two-lap time trial under four minutes with no major penalty events.

## Phase 7: Scenario Coverage

Once the core stack is stable, run scenario sweeps across:

- Nine obstacle placements
- Two parking zones
- Four parking start positions
- Several target speeds and curve caps
- Several steering/lookahead settings

Outputs should include success rate, average score, best score, worst failure mode, and the exact config for each run.

## Phase 8: Real-Car Transfer

Real-car work should use the same logical interfaces as the simulator but must calibrate the physical layer.

Items to measure:

- Arduino serial protocol and command rate
- Motor-driver PWM range
- Throttle/reverse/brake response
- Steering response and deadband
- Current limits and safe thermal behavior
- Camera intrinsics/extrinsics and distortion
- LiDAR pose and scan timing
- Ultrasonic sensor noise and field of view

Expected differences:

| Area | CARLA | Real car |
|---|---|---|
| Camera | Clean image, no distortion | Calibration and lighting compensation required |
| LiDAR | Low noise | Filtering and mounting calibration required |
| Steering | Immediate response | Servo/motor delay and deadband likely |
| Speed | Direct vehicle state | Encoder or estimated speed required |
| Lighting | Stable | Indoor lighting changes matter |

## Timeline

| Period | Work |
|---|---|
| Week 1 | Phase 0 environment and smoke tests |
| Weeks 2-3 | Phase 1 track and workshop preparation |
| Week 4 | Phase 2 sensor stack |
| Weeks 5-6 | Phase 3 perception data and baselines |
| Weeks 7-8 | Phase 4 control and Phase 5 FSM |
| Week 8 | Phase 6 scoring and Phase 7 scenario sweeps |
| August 3-5 | Practice driving and real-car tuning |
| August 10 | Competition day |

## Priority

1. Make the time trial finish reliably under four minutes.
2. Keep penalties near zero before chasing small lap-time gains.
3. Sweep speed, curve caps, lookahead, and steering gain on the loaded CARLA server.
4. Validate all obstacle placement cases.
5. Finish parking exit behavior and real parking-line tuning.
6. Lock real-car hardware clamps only after bench measurements.
