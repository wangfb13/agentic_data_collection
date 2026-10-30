# Mechanical arm interface audit (read-only, 2026-10-26)

Repository on 5090: `/home/dm/DM_das/starvla-feat-sparky_x5_starvla`.
All paths below are relative to that repository. Only text was read; no robot,
gripper, or sensor module was imported or executed. No motion was commanded.
No applicable AGENTS.md was found in the repository/ancestor directories;
one unrelated datasets directory was unreadable. `rg` is unavailable on 5090,
so remote searches used find/grep.

| Interface | Source and evidence | Status |
|---|---|---|
| Pose read / write | `src/lerobot/robots/spark3_pose_gripper/config_spark3_pose_gripper.py:30`: `/wbc_state_pose`; line 31 `/wbc_goal_pose`; lines 33–34 joint command/state topics | Text confirmed, no live topic check |
| Quaternion | `.../schema.py:12` and lines 294–303: xyz followed by qx,qy,qz,qw | xyzw confirmed |
| Frame | `.../spark3_pose_gripper.py:394–403` stamps commands `base_link` | Command frame confirmed; actual controller interpretation must be verified |
| Units | `.../schema.py:19–47` contains human-scale positions and 0.12971 TCP offset; no explicit unit annotation | Values consistent with metres, but physical unit remains a required confirmation |
| TCP | `.../schema.py:322–343`: state = flange @ flange_to_tcp, command flange = TCP @ inverse(flange_to_tcp) | Confirmed transform direction |
| TCP preset | `.../config_spark3_pose_gripper.py:45–52` defaults to v2. `.../schema.py:37–47`: left translation [0,0.01035,-0.12971], right [0,-0.01035,-0.12971], with side-specific rotations | Actual installed tool and calibration unknown |
| Speed | `.../spark3_pose_gripper.py:420–448` publishes endpoint poses. `linear_velocity`/`angular_velocity` derive from **base_twist**, not arm Cartesian velocity | No verified arm speed/linear trajectory API |
| Stop | `.../spark3_pose_gripper.py:600–630` only unregisters subscribers/publishers and disconnects cameras | Disconnect is not motor stop; true stop API unknown |
| Initial pose service | `.../spark3_pose_gripper.py:332–333,365–373` may call configured Trigger service during connect; default config line 38 is None | Enabling it can cause initialization motion |
| Partial actions | `.../spark3_pose_gripper.py:213–239` seeds `_last_action` from defaults, then `515–568` merges partial actions and publishes all configured arm/gripper commands | Reusing it for single-arm small steps risks unintended commands on omitted channels |
| Gripper values | `.../schema.py:186–203` normalized command range 0..1. `scripts/single_gripper_force_control.py:19–22,846–849` documents state*1000 and command counts/1000, increasing opens | Physical opening mm calibration is not established by these normalized counts |

The suggested container path `/home/cxzk/daimon_motion_system` was absent in the
running `daimon_ros1_das` container. No executable was run to discover an API.

## First-version geometry certificate

`geometry.interface_units()` is the only mm↔m conversion boundary. All planning
uses homogeneous matrices with translations in mm. Offsets are exactly
`T0 @ Trans(delta_acl) @ Rot(axis)`; lift is along base z.

Collision geometry contains two fingers and a palm plus three finite cylinders.
Each cylinder is enclosed by a circumscribed 32-sided prism. Full separating-axis
tests check oriented boxes against that enclosure. Safety margin expands boxes
by a cube containing the same-radius Euclidean ball. Path samples are additionally
expanded by a proven bound for translation and rotation between samples; this can
reject safe points, but does not create gaps between samples for the specified
linear/shortest-rotation path. This guarantee requires a verified real backend
that follows that trajectory, not merely a pose endpoint publisher.

During controlled contact, only a configured soft-pad volume is exempt from
bottle exclusion; finger cores, pad rims, palm, workspace and table stay checked.
All unknown physical dimensions remain TODO. During lift/lower, supply
`attached_grasp_pose` to `check_segment`; bottle pose follows the actual grasp,
and only base-z translation with unchanged rotation is supported. Bottle/table
contact at the supported starting/ending position is allowed, penetration is not.

The model does **not** cover arm links, wrist housing, cables, other obstacles,
or the other arm. Existing controller collision protection must remain enabled;
physical setup must supply safe workspace and complete controller collision
coverage. This version refuses real execution until those interfaces and all
TODO measurements are supplied. No controller settings are modified.
