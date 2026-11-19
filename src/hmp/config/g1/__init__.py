"""Unitree G1 task registry for the three-stage HMP pipeline.

Phase I   motion-imitation expert:           Mjlab-HMP-EncDec-Unitree-G1
Phase II  hybrid motion prior distillation:  Mjlab-HMP-Imitation-Unitree-G1
Phase III code-selection task policies on the frozen HMP:
  velocity tracking:                         Mjlab-HMP-Locomotion-Unitree-G1
  point-goal navigation:                     Mjlab-HMP-Navigation-Unitree-G1
  fall-recovery velocity tracking:           Mjlab-HMP-Recovery-Unitree-G1
"""

from mjlab.tasks.registry import register_mjlab_task

from hmp.rl import (
  HmpImitationRunner,
  HmpLocomotionOnPolicyRunner,
  HmpOnPolicyRunner,
)

from .env_cfgs import (
  unitree_g1_hmp_env_cfg,
  unitree_g1_hmp_locomotion_env_cfg,
  unitree_g1_hmp_navigation_env_cfg,
  unitree_g1_hmp_recovery_env_cfg,
)
from .rl_cfg import (
  unitree_g1_hmp_encdec_ppo_runner_cfg,
  unitree_g1_hmp_imitation_runner_cfg,
  unitree_g1_hmp_locomotion_runner_cfg,
  unitree_g1_hmp_navigation_runner_cfg,
  unitree_g1_hmp_recovery_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-HMP-EncDec-Unitree-G1",
  env_cfg=unitree_g1_hmp_env_cfg(),
  play_env_cfg=unitree_g1_hmp_env_cfg(play=True),
  rl_cfg=unitree_g1_hmp_encdec_ppo_runner_cfg(),
  runner_cls=HmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HMP-Imitation-Unitree-G1",
  env_cfg=unitree_g1_hmp_env_cfg(),
  play_env_cfg=unitree_g1_hmp_env_cfg(play=True),
  rl_cfg=unitree_g1_hmp_imitation_runner_cfg(),
  runner_cls=HmpImitationRunner,
)

register_mjlab_task(
  task_id="Mjlab-HMP-Locomotion-Unitree-G1",
  env_cfg=unitree_g1_hmp_locomotion_env_cfg(),
  play_env_cfg=unitree_g1_hmp_locomotion_env_cfg(play=True),
  rl_cfg=unitree_g1_hmp_locomotion_runner_cfg(),
  runner_cls=HmpLocomotionOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HMP-Navigation-Unitree-G1",
  env_cfg=unitree_g1_hmp_navigation_env_cfg(),
  play_env_cfg=unitree_g1_hmp_navigation_env_cfg(play=True),
  rl_cfg=unitree_g1_hmp_navigation_runner_cfg(),
  runner_cls=HmpLocomotionOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-HMP-Recovery-Unitree-G1",
  env_cfg=unitree_g1_hmp_recovery_env_cfg(),
  play_env_cfg=unitree_g1_hmp_recovery_env_cfg(play=True),
  rl_cfg=unitree_g1_hmp_recovery_runner_cfg(),
  runner_cls=HmpLocomotionOnPolicyRunner,
)
