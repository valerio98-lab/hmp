# Motion assets

Put the converted motion clips (`.npz`) in `g1_lafan_recovery/`, the folder
referenced by
[`motion_data_cfg.yaml`](../../src/hmp/config/g1/motion_data_cfg.yaml). See
the *Motion data* section of the main [README](../../README.md) for how to
download and convert LAFAN-G1.

Phases I and II train on every clip in the folder. The fall-recovery task
also reads the `*fallAndGetUp*.npz` clips from it to start part of its
episodes from fallen poses.

## NPZ format

Quaternions are `wxyz`. Two layouts are supported:

1. Minimal: `fps`, `root_pos`, `root_quat_w`, `joint_pos` (optionally
   `joint_vel`). Body kinematics are reconstructed with forward kinematics at
   load time. This is what `scripts/convert_lafan_g1.py` writes.
2. Extended: `fps`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`,
   `body_lin_vel_w`, `body_ang_vel_w`, `body_names`.
