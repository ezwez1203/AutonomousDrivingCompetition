# 2026 National University AI Autonomous Driving Competition Rules

This file is an English working summary of the competition information used by the project. It keeps the engineering-relevant rules close to the code, but the official Korean notice should still be treated as the source of truth if a dispute comes up.

## Overview

- Event: 2026 National University AI Autonomous Driving Competition
- Purpose: design and implement an autonomous 1/5-scale electric vehicle using cameras, LiDAR, ultrasonic sensors, and the provided hardware platform.
- Date: August 10, 2026, 10:00-18:00
- Venue: Semiconductor Building, Sungkyunkwan University Natural Sciences Campus, Suwon
- Participants: 24 teams from 12 universities, roughly 200 people
- Organizer group: Ministry of Education, KIAT, KEA, and the Sungkyunkwan University Future Mobility Bootcamp group
- Main tasks: track driving, obstacle avoidance, traffic-signal response, and perpendicular parking

## Schedule

| Date | Item | Venue |
|---|---|---|
| June 1-24 | Online pre-training and Q&A | Online eX-campus |
| June 26, 13:00-17:00 | Hardware control and sensor-fusion workshop | Sungkyunkwan University |
| August 3-5, 10:00-18:00 | Practice driving | Same venue |
| August 10, 10:00-11:00 | Opening ceremony | Same venue |
| August 10, 11:00-11:30 | Rule briefing and run-order draw | Same venue |
| August 10, 11:30-13:00 | Lunch and mentor networking | Same venue |
| August 10, 13:00-17:00 | Finals | Same venue |
| August 10, 17:00-18:00 | Scoring, awards, closing, photos | Same venue |

The schedule can change.

## Team And Vehicle Rules

- Undergraduate students may participate. Graduate students are not eligible.
- A team has 1-5 students and must include one faculty advisor.
- The vehicle must drive autonomously using onboard software. No human or assistive-device help is allowed during a run.
- Hardware provided by the organizers, including wheels, motor drivers, Arduino, and SMPS, may not be modified, replaced, disassembled, or rewired.
- Electrical and mechanical parts outside the permitted set may not be used.
- The vehicle configuration inspected before the event must be kept for all events. Per-mission reinstalling or reconfiguration is not allowed.
- The SMPS voltage is fixed at 12.0 V. Boosting above the announced voltage is not allowed and can lead to disqualification.
- Sensors must stay within the allowed envelope: 110 cm front/back, 60 cm left/right, and 75 cm height.
- Jigs for sensor attachment are allowed inside that envelope, but vehicle modification beyond sensor mounting and appearance is not allowed.
- Only one participant may be on the track area.

## Competition Format

- The time-trial event and mission event are scored separately, then combined for final ranking.
- Only the top seven teams that finish the two-lap time trial within four minutes advance to the mission event.
- Time trial limit: 4 minutes.
- Obstacle/traffic-signal mission limit: 4 minutes.
- Perpendicular parking mission limit: 4 minutes.
- Teams must be ready to start within two minutes after placing the vehicle on the start line. After that, the judge may start the run.
- If the vehicle does not start within 10 seconds after the start signal, one retry is allowed.
- Each event allows at most one retry caused by failure to start.
- Repositioning is allowed up to three times when the vehicle cannot return to its route after leaving the lane.

## Track Definitions

- The track has two lanes and is driven counterclockwise.
- Lane 1 is the inner lane; lane 2 is the outer lane.
- The inner white solid line is the IN line for lane 1.
- The dashed center line separates lane 1 and lane 2.
- The outer white solid line is the OUT line for lane 2.
- Road/lane width is 850 mm per lane.
- Lane-marking width is 50 mm.
- Start-line width is 100 mm.
- Parking space size is 950 mm x 1500 mm.
- Time-trial and obstacle/signal starts are on lane 2 opposite the crosswalk section.
- Parking starts from an IN-side start point near the signal area; one of four start positions is drawn on competition day.

## Penalty Terms

- Lane intrusion: at least one wheel touches the white dashed line. One contact with the same dashed line counts as one intrusion.
- Lane departure: at least one wheel crosses a white solid or dashed line.
- Retry: restart after launch failure. Power cycling or program reset is allowed, and the previous record is discarded.
- Reposition: after a lane departure where the car cannot return by itself, the judge moves or allows moving the car to a specified point.

## Time Trial

- The vehicle starts in lane 2 and must not cross the Start line before launch.
- It drives two counterclockwise laps in lane 2.
- The final time is two-lap elapsed time plus penalties.
- Any repositioning time is included in the driving time.
- If the vehicle exceeds four minutes, the run stops and penalties are assigned based on the unfinished section.
- IN-line intrusion and departure are judged by monitoring; OUT-line departure is judged by the following referee.
- After all time-trial runs, teams receive 15 minutes for maintenance, battery replacement, and mission-program adjustments.

## Mission Event

The mission event is split into obstacle/traffic-signal driving and perpendicular parking.

- Obstacle vehicles are numbered 1, 2, and 3 from nearest to the start line.
- The vehicle starts from the start line, avoids obstacles, and passes the crosswalk section in one continuous run.
- The traffic-signal mission can be run separately for 90 seconds only if the obstacle section took less than 150 seconds.
- The parking mission starts at an IN point, parks, holds for 3-5 seconds, exits, and reaches the opposite OUT point.
- If all wheels leave the parking zone during parking, the run is disqualified.
- Skipped missions receive penalties.

Mission points:

| Mission | Points |
|---|---:|
| Perpendicular parking | 200 |
| Obstacle avoidance | 200 |
| Traffic signal | 100 |

The mission event starts from a base score of 500, then adds successful mission points and subtracts penalties up to the specified limits.

## Mission Details

Obstacle avoidance:

- There are three obstacle vehicles. Vehicle type and color can vary and are announced by the organizer.
- Obstacle 1 is fixed in lane 1.
- Obstacles 2 and 3 each have three candidate positions across left/right lane choices, giving nine combined cases drawn on competition day.
- Collision with obstacle 1 causes a penalty and reposition to M1.
- Collision with obstacle 2 or 3 causes a penalty and reposition to M2.
- M1-M2 lane intrusion/departure, including the center line, causes a penalty and reposition to M1.
- M2-M3 departure causes a penalty and reposition to M2.
- Center-line intrusion in the M2-M3 avoidance section is not penalized.

Traffic signal:

- The car must stop before the crosswalk while the signal is red.
- The judge changes the signal to green after an arbitrary hold time.
- The car must restart within five seconds after green and pass the finish line with its front end.
- The traffic-signal penalty cap is 100 points.

Perpendicular parking:

- The selected start position is drawn on the competition day, and code changes after the draw are not allowed.
- Parking succeeds only if both front and rear wheels are inside the marked space.
- The car must park in reverse, hold for 3-5 seconds, exit forward, and pass the opposite OUT line with its front wheels.
- Parking must use steering. Differential-turn parking is not allowed.
- The parking penalty cap is 200 points.

## Final Ranking

- Time-trial rank and mission-event rank are calculated separately.
- The final rank is based on the sum of those ranks; lower is better.
- If mission scores are tied, the team with the shorter obstacle-avoidance completion time ranks higher.
- If the final rank sum is tied, the team with the better mission-event rank places higher.

## Inspection And Appeals

- Teams complete a self-check sheet, then judges inspect the vehicle.
- Vehicles violating hardware, sensor, or placement rules can be disqualified before the finals.
- If a violation can be fixed within 30 minutes, one reinspection may be allowed.
- The top four teams receive detailed inspection.
- Appeals are allowed only after all events, only during the official appeal window, and only for judgments directly involving the appealing team.
