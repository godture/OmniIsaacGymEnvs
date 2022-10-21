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


from abc import abstractmethod

from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate, quat_rotate_inverse, get_euler_xyz, quat_mul
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch
import math

# rotate local coordinate system by local z axis for 0,90,180,270 degrees to project legs' observations to the same space
QUATS_LEGS = torch.tensor([[[1,0,0,0]],[[0.7071068, 0, 0, 0.7071068]],[[0,0,0,1]],[[-0.7071068, 0, 0, 0.7071068]]])

class LocomotionTask(RLTask):
    def __init__(
        self,
        name,
        env,
        offset=None
    ) -> None:

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
        self.quats_legs = None
        self.calc_metrics = calculate_metrics
        if self._cfg['task_name'] == "Ant":
            self.calc_metrics = calculate_metrics_ant
            if 'masa' in self._cfg['train']['params']['network']['name']:
                self.get_obs = get_observations_antmasa
                self.quats_legs = QUATS_LEGS.repeat(1,self._num_envs,1).to(self._device)
        return

    @abstractmethod
    def set_up_scene(self, scene) -> None:
        pass

    @abstractmethod
    def get_robot(self):
        pass

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._robots.get_world_poses(clone=False)
        velocities = self._robots.get_velocities(clone=False)
        velocity = velocities[:, 0:3]
        ang_velocity = velocities[:, 3:6]
        dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)

        # force sensors attached to the feet
        sensor_force_torques = self._robots._physics_view.get_force_sensor_forces() # (num_envs, num_sensors, 6)

        # slow down running from 210k fps to 2100 fps
        # poses_feet = [foot.get_world_poses(clone=False) for foot in self._feet]
        # poses_feet = [torch.cat(pose_foot, dim=-1) for pose_foot in poses_feet]
        # velocities_feet = [foot.get_velocities(clone=False) for foot in self._feet]

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = self.get_obs(
            torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.targets, self.potentials, self.dt,
            self.inv_start_rot, self.basis_vec0, self.basis_vec1, self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            sensor_force_torques, self._num_envs, self.contact_force_scale, self.actions, self.angular_velocity_scale, self.quats_legs
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

        # to_target = self.targets[env_ids] - self.initial_root_pos[env_ids]
        # to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = self.initial_root_pos[env_ids,:2] / self.dt #-torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        num_resets = len(env_ids)

    def post_reset(self):
        self._robots = self.get_robot()
        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()
        self.initial_dof_pos = self._robots.get_joint_positions()

        # initialize some data used later on
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self._device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.target_dirs = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.dt = 1.0 / 60.0
        self.potentials = self.initial_root_pos[...,:2] / self.dt # self.torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self._device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self._device)

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self.calc_metrics(
            self.obs_buf, self.actions, self.up_weight, self.heading_weight, self.potentials, self.prev_potentials,
            self.actions_cost_scale, self.energy_cost_scale, self.termination_height,
            self.death_cost, self._robots.num_dof, self.get_dof_at_limit_cost(), self.alive_reward_scale, self.motor_effort_ratio
        )

    def is_done(self) -> None:
        self.reset_buf[:] = is_done(
            self.obs_buf, self.termination_height, self.reset_buf, self.progress_buf, self._max_episode_length
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def get_observations(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    targets,
    potentials,
    dt,
    inv_start_rot,
    basis_vec0,
    basis_vec1,
    dof_limits_lower,
    dof_limits_upper,
    dof_vel_scale,
    sensor_force_torques,
    num_envs,
    contact_force_scale,
    actions,
    angular_velocity_scale,
    quats_legs
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    prev_potentials = potentials.clone()
    potentials = torso_position[...,:2] / dt # -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, num_sensors * 6, num_dofs
    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),
            vel_loc,
            angvel_loc * angular_velocity_scale,
            normalize_angle(yaw).unsqueeze(-1),
            normalize_angle(roll).unsqueeze(-1),
            normalize_angle(angle_to_target).unsqueeze(-1),
            up_proj.unsqueeze(-1),
            heading_proj.unsqueeze(-1),
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            # sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,
            actions,
        ),
        dim=-1,
    )

    return obs, potentials, prev_potentials, up_vec, heading_vec

# @torch.jit.script
def get_observations_antmasa(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    targets,
    potentials,
    dt,
    inv_start_rot,
    basis_vec0,
    basis_vec1,
    dof_limits_lower,
    dof_limits_upper,
    dof_vel_scale,
    sensor_force_torques,
    num_envs,
    contact_force_scale,
    actions,
    angular_velocity_scale,
    quats_legs
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, int, float, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    prev_potentials = potentials.clone()
    potentials = torso_position[...,:2] / dt # -torch.norm(to_target, p=2, dim=-1) / dt

    num_batch = velocity.shape[0]
    vel_locs = torch.zeros([num_batch,12], device=velocity.device, dtype=velocity.dtype)
    angvel_locs = torch.zeros([num_batch,12], device=velocity.device, dtype=velocity.dtype)
    yaws = torch.zeros([num_batch,4], device=velocity.device, dtype=velocity.dtype)
    rolls = torch.zeros([num_batch,4], device=velocity.device, dtype=velocity.dtype)
    angles_to_target = torch.zeros([num_batch,4], device=velocity.device, dtype=velocity.dtype)
    heading_projs = torch.zeros([num_batch,4], device=velocity.device, dtype=velocity.dtype)
    up_proj = None
    up_vec = None
    heading_vec = None
    for i,quat_leg in enumerate(quats_legs):
        torso_r_leg = quat_mul(torso_rotation, quat_leg)
        torso_quat, u_proj, heading_proj, u_vec, h_vec = compute_heading_and_up(
            torso_r_leg, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
        )
        vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
            torso_quat, velocity, ang_velocity, targets, torso_position
        )
        vel_locs[...,i*3:(i+1)*3] = vel_loc
        angvel_locs[...,i*3:(i+1)*3] = angvel_loc
        yaws[...,i] = yaw
        rolls[...,i] = roll
        heading_projs[...,i] = heading_proj
        angles_to_target[...,i] = angle_to_target
        if heading_vec is None: heading_vec = h_vec
        if up_vec is None: up_vec = u_vec
        if up_proj is None: up_proj = u_proj
        

    # torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
    #     torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    # )

    # vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
    #     torso_quat, velocity, ang_velocity, targets, torso_position
    # )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)
    # foot joint directions of legs 1 and 2 are different from that of 0 and 3
    # maybe reverse them in the network
    # dof_pos_scaled[...,5:7] = - dof_pos_scaled[...,5:7]
    # dof_vel[...,5:7] = - dof_vel[...,5:7]
    # actions[...,5:7] = - actions[...,5:7]

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, num_sensors * 6, num_dofs
    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),
            vel_locs,
            angvel_locs * angular_velocity_scale,
            torch.sin(yaws),
            normalize_angle(rolls),
            torch.sin(angles_to_target),
            up_proj.unsqueeze(-1),
            heading_projs,
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            # sensor_force_torques.reshape(num_envs, -1) * contact_force_scale,
            actions,
            torch.cos(yaws),
            torch.cos(angles_to_target)
        ),
        dim=-1,
    )

    return obs, potentials, prev_potentials, up_vec, heading_vec

@torch.jit.script
def is_done(
    obs_buf,
    termination_height,
    reset_buf,
    progress_buf,
    max_episode_length
):
    # type: (Tensor, float, Tensor, Tensor, float) -> Tensor
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    return reset


@torch.jit.script
def calculate_metrics(
    obs_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    death_cost,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    motor_effort_ratio
):
    # type: (Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, int, Tensor, float, Tensor) -> Tensor

    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(
        obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8
    )

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 12+num_dof:12+num_dof*2])* motor_effort_ratio.unsqueeze(0), dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(obs_buf[...,0]) * alive_reward_scale
    progress_reward = potentials[...,0] - prev_potentials[...,0]
    off_track_cost = torch.abs(potentials[...,1] - prev_potentials[...,1])

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
def calculate_metrics_ant(
    obs_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    death_cost,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    motor_effort_ratio
):
    # type: (Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, int, Tensor, float, Tensor) -> Tensor

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(obs_buf[:,0])

    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    if obs_buf.shape[-1] == 36:
        up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)
        electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 12+num_dof:12+num_dof*2])* motor_effort_ratio.unsqueeze(0), dim=-1)
    elif obs_buf.shape[-1] in [66, 74]:
        up_reward = torch.where(obs_buf[:, 37] > 0.93, up_reward + up_weight, up_reward)
        electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 42+num_dof:42+num_dof*2])* motor_effort_ratio.unsqueeze(0), dim=-1)
    else:
        assert False, f"observation shape {obs_buf.shape[-1]} not exist"

    # reward for duration of staying alive
    alive_reward = torch.ones_like(obs_buf[...,0]) * alive_reward_scale

    progress_reward = potentials[...,0] - prev_potentials[...,0]
    off_track_cost = torch.abs(potentials[...,1] - prev_potentials[...,1])

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        # + heading_reward # no heading reward for ant central symmetry
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