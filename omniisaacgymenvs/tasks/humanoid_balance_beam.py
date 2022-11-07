# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from omniisaacgymenvs.tasks.shared.locomotion import LocomotionTask
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.humanoid import Humanoid

from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot_notarget, quat_conjugate, compute_up, normalize_angle, get_basis_vector
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.objects import DynamicCylinder

import numpy as np
import torch
import math

from pxr import PhysxSchema


class HumanoidBBTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._num_observations = self._cfg['train']['num_observations']
        self._num_actions = 21
        self._humanoid_positions = torch.tensor([0, 0, 2.34])
        self._wood_position = torch.tensor([65, 0, 0.8])
        self._wood_orientation = torch.tensor([0.7071068, 0, 0.7071068, 0])

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self._task_cfg["env"]["angularVelocityScale"]
        self.contact_force_scale = self._task_cfg["env"]["contactForceScale"]
        self.power_scale = self._task_cfg["env"]["powerScale"]
        self.heading_weight = self._task_cfg["env"]["headingWeight"]
        self.up_weight = self._task_cfg["env"]["upWeight"]
        self.actions_cost_scale = self._task_cfg["env"]["actionsCost"]
        self.energy_cost_scale = self._task_cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self._task_cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self._task_cfg["env"]["deathCost"]
        self.termination_height = self._task_cfg["env"]["terminationHeight"]
        self.alive_reward_scale = self._task_cfg["env"]["alive_reward_scale"]

        RLTask.__init__(self, name, env)
        self.get_obs = get_observations
        self.calc_metrics = calculate_metrics

        if 'masa' in self._cfg['train']['params']['network']['name']:
            self.get_obs = get_observations_masa
        self.inds_neg = torch.tensor(self._cfg['train']['params']['network'].get('inds_neg', []),dtype=int).to(self._device)
        self.left_inds = torch.tensor(self._cfg['train']['params']['network'].get('left_inds', []),dtype=int).to(self._device)
        self.left_inds_neg = torch.tensor(self._cfg['train']['params']['network'].get('left_inds_neg', []),dtype=int).to(self._device)
        return

    def set_up_scene(self, scene) -> None:
        self.get_humanoid()
        RLTask.set_up_scene(self, scene)
        for i in range(64):
            translation_wood = self._wood_position
            translation_wood[1] = -157.5 + i*5
            wood = DynamicCylinder(
                    prim_path="/World/woods/wood_"+str(i),
                    translation=translation_wood,
                    name='wood_'+str(i),
                    radius = 0.05,
                    height= 450.,
                    orientation=self._wood_orientation,
                    color=torch.tensor([0.9, 0.6, 0.2]),
                )
            self._sim_config.apply_articulation_settings("wood", get_prim_at_path(wood.prim_path), self._sim_config.parse_actor_config("wood"))
            scene.add(wood)
        self._humanoids = ArticulationView(prim_paths_expr="/World/envs/.*/Humanoid/torso", name="humanoid_view", reset_xform_properties=False)
        # self._feet = [RigidPrimView(prim_paths_expr="/World/envs/.*/Humanoid/left_foot", name="left_feet_view", reset_xform_properties=False),
        #                 RigidPrimView(prim_paths_expr="/World/envs/.*/Humanoid/right_foot", name="right_feet_view", reset_xform_properties=False)]
        # self._left_feet = RigidPrimView(prim_paths_expr="/World/envs/.*/Humanoid/left_foot", name="left_feet_view", reset_xform_properties=False)
        # self._right_feet = RigidPrimView(prim_paths_expr="/World/envs/.*/Humanoid/right_foot", name="right_feet_view", reset_xform_properties=False)
        scene.add(self._humanoids)
        return

    def get_humanoid(self):
        humanoid = Humanoid(prim_path=self.default_zero_env_path + "/Humanoid", name="Humanoid", translation=self._humanoid_positions)
        self._sim_config.apply_articulation_settings("Humanoid", get_prim_at_path(humanoid.prim_path), 
            self._sim_config.parse_actor_config("Humanoid"))

    def get_robot(self):
        return self._humanoids

    def post_reset(self):
        self.joint_gears = torch.tensor(
            [
                67.5000, # lower_waist
                67.5000, # lower_waist
                67.5000, # right_upper_arm
                67.5000, # right_upper_arm
                67.5000, # left_upper_arm
                67.5000, # left_upper_arm
                67.5000, # pelvis
                45.0000, # right_lower_arm
                45.0000, # left_lower_arm
                45.0000, # right_thigh: x
                135.0000, # right_thigh: y
                45.0000, # right_thigh: z
                45.0000, # left_thigh: x
                135.0000, # left_thigh: y
                45.0000, # left_thigh: z
                90.0000, # right_knee
                90.0000, # left_knee
                22.5, # right_foot
                22.5, # right_foot
                22.5, # left_foot
                22.5, # left_foot
            ],
            device=self._device,
        )
        self.max_motor_effort = torch.max(self.joint_gears)
        self.motor_effort_ratio = self.joint_gears / self.max_motor_effort
        dof_limits = self._humanoids.get_dof_limits()
        self.dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self._device)

        self._robots = self.get_robot()
        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()
        self.initial_dof_pos = self._robots.get_joint_positions()

        # initialize some data used later on
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.target_dirs = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.dt = 1.0 / 60.0
        self.xys = self.initial_root_pos[...,:2] / self.dt # self.torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self._device).repeat(self.num_envs)
        self.prev_xys = self.xys.clone()

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self._device)

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._robots.get_world_poses(clone=False)
        velocities = self._robots.get_velocities(clone=False)
        velocity = velocities[:, 0:3]
        ang_velocity = velocities[:, 3:6]
        dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)

        # force sensors attached to the feet
        # sensor_force_torques = self._robots._physics_view.get_force_sensor_forces() # (num_envs, num_sensors, 6)

        # slow down running from 210k fps to 2100 fps
        # poses_feet = [foot.get_world_poses(clone=False) for foot in self._feet]
        # poses_feet = [torch.cat(pose_foot, dim=-1) for pose_foot in poses_feet]
        # velocities_feet = [foot.get_velocities(clone=False) for foot in self._feet]

        self.obs_buf[:], self.xys[:], self.prev_xys[:], self.up_vec[:], self.heading_proj = self.get_obs(
            torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.xys, self.dt,
            self.basis_vec0, self.basis_vec1, self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions, self.angular_velocity_scale, self.inds_neg, self.left_inds, self.left_inds_neg
        )
        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        forces = self.actions * self.joint_gears * self.power_scale

        indices = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)

        # applies joint torques
        self._robots.set_joint_efforts(forces, indices=indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions and velocities
        dof_pos = torch_rand_float(-0.2, 0.2, (num_resets, self._robots.num_dof), device=self._device)
        dof_pos[:] = tensor_clamp(
            self.initial_dof_pos[env_ids] + dof_pos, self.dof_limits_lower, self.dof_limits_upper
        )
        dof_vel = torch_rand_float(-0.1, 0.1, (num_resets, self._robots.num_dof), device=self._device)

        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        root_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        self._robots.set_joint_positions(dof_pos, indices=env_ids)
        self._robots.set_joint_velocities(dof_vel, indices=env_ids)

        self._robots.set_world_poses(root_pos, root_rot, indices=env_ids)
        self._robots.set_velocities(root_vel, indices=env_ids)

        self.prev_xys[env_ids] = self.initial_root_pos[env_ids,:2] / self.dt #-torch.norm(to_target, p=2, dim=-1) / self.dt
        self.xys[env_ids] = self.prev_xys[env_ids].clone()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        num_resets = len(env_ids)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self.calc_metrics(
            self.obs_buf, self.actions, self.up_weight, self.heading_proj, self.heading_weight, self.xys, self.prev_xys,
            self.actions_cost_scale, self.energy_cost_scale, self.termination_height,
            self.death_cost, self._robots.num_dof, self.get_dof_at_limit_cost(), self.alive_reward_scale, self.motor_effort_ratio
        )

    def is_done(self) -> None:
        if len(self.obs_buf.shape) == 2:
            height = self.obs_buf[:,0]
        elif len(self.obs_buf.shape) == 3:
            height = self.obs_buf[:,0,0]
        else:
            assert False, f"observation shape {obs_buf.shape} not exist"
        self.reset_buf[:] = is_done(
            height, self.termination_height, self.reset_buf, self.progress_buf, self._max_episode_length
        )

    def get_dof_at_limit_cost(self):
        return get_dof_at_limit_cost(self.obs_buf, self.motor_effort_ratio, self.joints_at_limit_cost_scale)


@torch.jit.script
def get_dof_at_limit_cost(obs_buf, motor_effort_ratio, joints_at_limit_cost_scale):
    # type: (Tensor, Tensor, float) -> Tensor
    if len(obs_buf.shape) == 2:
        dof_pos = obs_buf[:, 12:33]
    elif len(obs_buf.shape) == 3:
        dof_pos = obs_buf[:, 1, 12:33]
    else:
        assert False, f"observation shape {obs_buf.shape} not exist"
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(dof_pos) - 0.98) / 0.02
    dof_at_limit_cost = torch.sum(
        (torch.abs(dof_pos) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1
    )
    return dof_at_limit_cost

@torch.jit.script
def get_observations(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    xys,
    dt,
    basis_vec0,
    basis_vec1,
    dof_limits_lower,
    dof_limits_upper,
    dof_vel_scale,
    actions,
    angular_velocity_scale,
    inds_neg = torch.zeros([]),
    left_inds = torch.zeros([]),
    left_inds_neg = torch.zeros([])
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, float, Tensor, float, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    prev_xys = xys.clone()
    xys = torso_position[...,:2] / dt

    torso_quat, up_proj, up_vec = compute_up(
        torso_rotation, basis_vec1, 2
    )
    heading_vec = get_basis_vector(torso_quat, basis_vec0).view(-1,3)

    vel_loc, angvel_loc, roll, pitch, yaw = compute_rot_notarget(
        torso_quat, velocity, ang_velocity
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, num_sensors * 6, num_dofs
    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),
            vel_loc,
            angvel_loc * angular_velocity_scale,
            normalize_angle(yaw).unsqueeze(-1),
            normalize_angle(pitch).unsqueeze(-1),
            normalize_angle(roll).unsqueeze(-1),
            up_proj.unsqueeze(-1),
            torso_position[:, 1].view(-1,1),
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            actions,
        ),
        dim=-1,
    )

    return obs, xys, prev_xys, up_vec, heading_vec[:,0]

@torch.jit.script
def get_observations_masa(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    xys,
    dt,
    basis_vec0,
    basis_vec1,
    dof_limits_lower,
    dof_limits_upper,
    dof_vel_scale,
    actions,
    angular_velocity_scale,
    inds_neg = torch.zeros([]),
    left_inds = torch.zeros([]),
    left_inds_neg = torch.zeros([])
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, float, Tensor, float, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    prev_xys = xys.clone()
    xys = torso_position[...,:2] / dt

    torso_quat, up_proj, up_vec = compute_up(
        torso_rotation, basis_vec1, 2
    )
    heading_vec = get_basis_vector(torso_quat, basis_vec0).view(-1,3)

    vel_loc, angvel_loc, roll, pitch, yaw = compute_rot_notarget(
        torso_quat, velocity, ang_velocity
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, num_sensors * 6, num_dofs
    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),
            vel_loc,
            angvel_loc * angular_velocity_scale,
            normalize_angle(yaw).unsqueeze(-1),
            normalize_angle(pitch).unsqueeze(-1),
            normalize_angle(roll).unsqueeze(-1),
            up_proj.unsqueeze(-1),
            torso_position[:, 1].view(-1,1),
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            actions,
        ),
        dim=-1,
    )
    obs[:,inds_neg] = - obs[:,inds_neg]
    obs_shared_agents = torch.zeros([obs.shape[0], 2, len(left_inds)], device=obs.device, dtype=obs.dtype)
    obs_shared_agents[:, 0] = obs[:, left_inds]
    obs_shared_agents[:, 0, left_inds_neg] = - obs_shared_agents[:, 0, left_inds_neg]
    obs_shared_agents[:, 1] = obs

    return obs_shared_agents, xys, prev_xys, up_vec, heading_vec[:,0]

@torch.jit.script
def calculate_metrics(
    obs_buf_origin,
    actions,
    up_weight,
    heading_proj,
    heading_weight,
    xys,
    prev_xys,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    death_cost,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    motor_effort_ratio
):
    # type: (Tensor, Tensor, float, Tensor, float, Tensor, Tensor, float, float, float, float, int, Tensor, float, Tensor) -> Tensor

    # heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    # heading_reward = torch.where(
    #     obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8
    # )

    if len(obs_buf_origin.shape) == 3:
        obs_buf = obs_buf_origin[:,1].clone()
    elif len(obs_buf_origin.shape) == 2:
        obs_buf = obs_buf_origin.clone()
    else:
        assert False, f"observation shape {obs_buf_origin.shape} not exist"

    heading_reward = heading_proj * heading_weight

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(obs_buf[:,0])
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 12+num_dof:12+num_dof*2])* motor_effort_ratio.unsqueeze(0), dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(obs_buf[:,0]) * alive_reward_scale
    progress_reward = xys[:,0] - prev_xys[:,0]
    off_track_cost = torch.abs(xys[:,1] - prev_xys[:,1])

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
        - off_track_cost
    )

    # adjust reward for fallen agents
    total_reward = torch.where(
        obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward
    )
    return total_reward

@torch.jit.script
def is_done(
    height,
    termination_height,
    reset_buf,
    progress_buf,
    max_episode_length
):
    # type: (Tensor, float, Tensor, Tensor, float) -> Tensor
    reset = torch.where(height < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    return reset
