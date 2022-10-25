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


from omniisaacgymenvs.robots.articulations.ant import Ant
from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omni.isaac.core.utils.torch.rotations import compute_up, compute_rot_notarget, quat_mul, quat_conjugate, quat_rotate_inverse, get_euler_xyz, get_basis_vector
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.objects import DynamicCylinder
from omni.isaac.core.prims import RigidPrimView

from pxr import PhysxSchema

import numpy as np
import torch
import math

# rotate local coordinate system by local z axis for 0,90,180,270 degrees to project legs' observations to the same space
QUATS_LEGS = torch.tensor([[[1,0,0,0]],[[0.7071068, 0, 0, 0.7071068]],[[0,0,0,1]],[[-0.7071068, 0, 0, 0.7071068]]])

class AntBalanceTask(RLTask):
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
        self._num_actions = 8
        self._ant_positions = torch.tensor([0, 0, 0.5])
        self._wood_position = torch.tensor([0.0, 0.0, 2.8])

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self._task_cfg["env"]["angularVelocityScale"]
        # self.contact_force_scale = self._task_cfg["env"]["contactForceScale"]
        self.power_scale = self._task_cfg["env"]["powerScale"]
        # self.heading_weight = self._task_cfg["env"]["headingWeight"]
        self.up_weight = self._task_cfg["env"]["upWeight"]
        self.actions_cost_scale = self._task_cfg["env"]["actionsCost"]
        self.energy_cost_scale = self._task_cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self._task_cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self._task_cfg["env"]["deathCost"]
        self.termination_height = self._task_cfg["env"]["terminationHeight"]
        self.alive_reward_scale = self._task_cfg["env"]["alive_reward_scale"]
        self.termination_height_wood = self._task_cfg["env"]["terminationHeightWood"]

        RLTask.__init__(self, name, env)
        # TODO: add normal observation func
        self.get_obs = get_observations
        self.quats_legs = None
        self.calc_metrics = calculate_metrics

        if 'masa' in self._cfg['train']['params']['network']['name']:
            self.get_obs = get_observations_masa
            self.quats_legs = QUATS_LEGS.repeat(1,self._num_envs,1).to(self._device)
        return

    def set_up_scene(self, scene) -> None:
        self.get_ant()
        self.add_wood()
        RLTask.set_up_scene(self, scene)
        self._ants = ArticulationView(prim_paths_expr="/World/envs/.*/Ant/torso", name="ant_view", reset_xform_properties=False)
        scene.add(self._ants)
        self._woods = RigidPrimView(prim_paths_expr="/World/envs/.*/Wood/wood", name="wood_view", reset_xform_properties=False)
        scene.add(self._woods)
        return
    
    def add_wood(self):
        wood = DynamicCylinder(
            prim_path=self.default_zero_env_path + "/Wood/wood", 
            translation=self._wood_position, 
            name="wood_0",
            radius = 0.1,
            height= 4.0,
            color=torch.tensor([0.9, 0.6, 0.2]),
        )
        self._sim_config.apply_articulation_settings("wood", get_prim_at_path(wood.prim_path), self._sim_config.parse_actor_config("wood"))

    def get_ant(self):
        ant = Ant(prim_path=self.default_zero_env_path + "/Ant", name="Ant", translation=self._ant_positions)
        self._sim_config.apply_articulation_settings("Ant", get_prim_at_path(ant.prim_path), self._sim_config.parse_actor_config("Ant"))

    def get_robot(self):
        return self._ants

    def post_reset(self):
        self.joint_gears = torch.tensor([15, 15, 15, 15, 15, 15, 15, 15], dtype=torch.float32, device=self._device)
        dof_limits = self._ants.get_dof_limits()
        self.dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self._device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self._device)

        self._robots = self.get_robot()
        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()
        self.initial_dof_pos = self._robots.get_joint_positions()
        self.initial_wood_pos, self.initial_wood_rot = self._woods.get_world_poses()

        # initialize some data used later on
        # self.start_rotation = torch.tensor([1, 0, 0, 0], device=self._device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        # self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        # self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        # self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        # self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        # self.target_dirs = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.dt = 1.0 / 60.0
        self.xys = self.initial_root_pos[...,:2] / self.dt # self.torch.tensor([-1000.0 / self.dt], dtype=torch.float32, device=self._device).repeat(self.num_envs)
        self.prev_xys = self.xys.clone()

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self._device)

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._robots.get_world_poses(clone=False)
        torso_position = torso_position - self._env_pos
        velocities = self._robots.get_velocities(clone=False)
        velocity = velocities[:, 0:3]
        ang_velocity = velocities[:, 3:6]
        dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)

        self.wood_positions, wood_orientations = self._woods.get_world_poses(clone=False)
        self.wood_positions = self.wood_positions[:, 0:3] - self._env_pos
        self.wood_up_vec = get_basis_vector(wood_orientations, self.basis_vec1).view(self.num_envs, 3)

        wood_velocities = self._woods.get_velocities(clone=False)
        wood_linvels = wood_velocities[:, 0:3]
        wood_angvels = wood_velocities[:, 3:6]

        # slow down running from 210k fps to 2100 fps
        # poses_feet = [foot.get_world_poses(clone=False) for foot in self._feet]
        # poses_feet = [torch.cat(pose_foot, dim=-1) for pose_foot in poses_feet]
        # velocities_feet = [foot.get_velocities(clone=False) for foot in self._feet]

        # TODO: consider whether normalize wood relative position
        self.obs_buf[:], self.xys[:], self.prev_xys[:], self.up_vec[:] = self.get_obs(
            torso_position, torso_rotation, velocity, ang_velocity, dof_pos, dof_vel, self.xys, self.dt, self.basis_vec1,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale, self.actions, self.angular_velocity_scale,
            self.quats_legs, self.wood_positions, wood_orientations, wood_linvels, wood_angvels
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

        root_pos_wood, root_rot_wood = self.initial_wood_pos[env_ids], self.initial_wood_rot[env_ids]
        self._woods.set_world_poses(root_pos_wood, root_rot_wood, indices=env_ids)
        wood_vel = torch.zeros((num_resets, 6), device=self._device)
        self._woods.set_velocities(wood_vel, indices=env_ids)

        self.prev_xys[env_ids] = self.initial_root_pos[env_ids,:2] / self.dt
        self.xys[env_ids] = self.prev_xys[env_ids].clone()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        num_resets = len(env_ids)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = calculate_metrics(
            self.obs_buf, self.actions, self.up_weight, self.xys, self.prev_xys, self.actions_cost_scale, self.energy_cost_scale,
            self.termination_height, self.termination_height_wood, self.death_cost, self._robots.num_dof, self.get_dof_at_limit_cost(),
            self.alive_reward_scale, self.motor_effort_ratio, self.wood_positions[:,2], self.wood_up_vec[:,2]
        )

    def is_done(self) -> None:
        self.reset_buf[:] = is_done(
            self.obs_buf, self.termination_height, self.reset_buf, self.progress_buf, self._max_episode_length, self.wood_positions[:,2], self.wood_up_vec[:,2]
        )

    def get_dof_at_limit_cost(self):
        return get_dof_at_limit_cost(self.obs_buf, self._ants.num_dof)

@torch.jit.script
def get_dof_at_limit_cost(obs_buf, num_dof):
    # type: (Tensor, int) -> Tensor
    if obs_buf.shape[-1] == 48:
        return torch.sum(obs_buf[:, 12:12+num_dof] > 0.99, dim=-1)
    elif obs_buf.shape[-1] == 109:
        return torch.sum(obs_buf[:, 36:36+num_dof] > 0.99, dim=-1)
    else:
        assert False, f"observation shape {obs_buf.shape[-1]} not exist"

@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

# @torch.jit.script
def get_observations(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    xys,
    dt,
    basis_vec1,
    dof_limits_lower,
    dof_limits_upper,
    dof_vel_scale,
    actions,
    angular_velocity_scale,
    quats_legs,
    wood_positions,
    wood_orientations,
    wood_linvels,
    wood_angvels
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, float, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    prev_xys = xys.clone()
    xys = torso_position[:,:2] / dt

    num_batch = velocity.shape[0]
    up_proj = None
    up_vec = None


    torso_quat, up_proj, up_vec = compute_up(
        torso_rotation, basis_vec1, 2
    )
    vel_loc, angvel_loc, roll, pitch, yaw = compute_rot_notarget(
        torso_quat, velocity, ang_velocity
    )

    # wood
    wood_pos_loc = quat_rotate_inverse(torso_quat, wood_positions-torso_position)
    wood_orientations_loc = quat_mul(quat_conjugate(torso_quat), wood_orientations)
    wood_up_vec_loc = get_basis_vector(wood_orientations_loc, basis_vec1).view(num_batch, 3)
    wood_roll_loc, wood_pitch_loc, _ = get_euler_xyz(wood_orientations_loc)
    wood_vel_loc = quat_rotate_inverse(torso_quat, wood_linvels)
    wood_angvel_loc = quat_rotate_inverse(torso_quat, wood_angvels)


    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)


    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, num_sensors * 6, num_dofs
    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),
            torso_position[:, :2], # stay at target position
            vel_loc,
            angvel_loc * angular_velocity_scale,
            normalize_angle(pitch).unsqueeze(-1),
            normalize_angle(roll).unsqueeze(-1),
            up_proj.unsqueeze(-1),
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            actions,
            wood_pos_loc,
            wood_vel_loc,
            wood_angvel_loc,
            wood_up_vec_loc
        ),
        dim=-1,
    )

    return obs, xys, prev_xys, up_vec

# @torch.jit.script
def get_observations_masa(
    torso_position,
    torso_rotation,
    velocity,
    ang_velocity,
    dof_pos,
    dof_vel,
    xys,
    dt,
    basis_vec1,
    dof_limits_lower,
    dof_limits_upper,
    dof_vel_scale,
    actions,
    angular_velocity_scale,
    quats_legs,
    wood_positions,
    wood_orientations,
    wood_linvels,
    wood_angvels
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, float, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    prev_xys = xys.clone()
    xys = torso_position[:,:2] / dt

    num_batch = velocity.shape[0]
    vel_locs = torch.zeros([num_batch,12], device=velocity.device, dtype=velocity.dtype)
    angvel_locs = torch.zeros([num_batch,12], device=velocity.device, dtype=velocity.dtype)
    pitches = torch.zeros([num_batch,4], device=velocity.device, dtype=velocity.dtype)
    rolls = torch.zeros([num_batch,4], device=velocity.device, dtype=velocity.dtype)
    up_proj = None
    up_vec = None
    wood_pos_locs = torch.zeros([num_batch,12], device=velocity.device, dtype=velocity.dtype)
    wood_vel_locs = torch.zeros([num_batch,12], device=velocity.device, dtype=velocity.dtype)
    wood_angvel_locs = torch.zeros([num_batch,12], device=velocity.device, dtype=velocity.dtype)
    wood_roll_locs = torch.zeros([num_batch,4], device=velocity.device, dtype=velocity.dtype)
    wood_pitch_locs = torch.zeros([num_batch,4], device=velocity.device, dtype=velocity.dtype)
    for i,quat_leg in enumerate(quats_legs):
        torso_r_leg = quat_mul(torso_rotation, quat_leg)
        torso_quat_leg, u_proj, u_vec = compute_up(
            torso_r_leg, basis_vec1, 2
        )
        vel_loc, angvel_loc, roll, pitch, yaw = compute_rot_notarget(
            torso_quat_leg, velocity, ang_velocity
        )
        vel_locs[:,i*3:(i+1)*3] = vel_loc
        angvel_locs[:,i*3:(i+1)*3] = angvel_loc
        pitches[:,i] = pitch
        rolls[:,i] = roll
        if up_vec is None: up_vec = u_vec
        if up_proj is None: up_proj = u_proj
        # wood
        wood_pos_locs[:,i*3:(i+1)*3] = quat_rotate_inverse(torso_r_leg, wood_positions-torso_position)
        wood_roll_locs[:,i], wood_pitch_locs[:,i], _ = get_euler_xyz(quat_mul(quat_conjugate(torso_r_leg), wood_orientations))
        wood_vel_locs[:,i*3:(i+1)*3] = quat_rotate_inverse(torso_r_leg, wood_linvels)
        wood_angvel_locs[:,i*3:(i+1)*3] = quat_rotate_inverse(torso_r_leg, wood_angvels)


    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)


    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, num_sensors * 6, num_dofs
    obs = torch.cat(
        (
            torso_position[:, 2].view(-1, 1),
            torso_position[:, :2], # stay at target position
            vel_locs,
            angvel_locs * angular_velocity_scale,
            normalize_angle(pitches),
            normalize_angle(rolls),
            up_proj.unsqueeze(-1),
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            actions,
            wood_pos_locs,
            wood_vel_locs,
            wood_angvel_locs,
            wood_roll_locs,
            wood_pitch_locs
        ),
        dim=-1,
    )

    return obs, xys, prev_xys, up_vec

# @torch.jit.script
def calculate_metrics(
    obs_buf,
    actions,
    up_weight,
    xys,
    prev_xys,
    actions_cost_scale,
    energy_cost_scale,
    termination_height,
    termination_height_wood,
    death_cost,
    num_dof,
    dof_at_limit_cost,
    alive_reward_scale,
    motor_effort_ratio,
    z_woods,
    z_proj_wood
):
    # type: (Tensor, Tensor, float, Tensor, Tensor, float, float, float, float, float, int, Tensor, float, Tensor, Tensor, Tensor) -> Tensor

    # aligning up axis of robot and environment
    up_reward = z_proj_wood * up_weight

    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    if obs_buf.shape[-1] == 48:
        # up_reward = torch.where(obs_buf[:, 10] > 0.93, z_proj_wood * up_weight, up_reward)
        electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 12+num_dof:12+num_dof*2])* motor_effort_ratio.unsqueeze(0), dim=-1)
    elif obs_buf.shape[-1] == 109:
        # up_reward = torch.where(z_proj_wood > 0.93, up_reward + up_weight, up_reward)
        electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 36+num_dof:36+num_dof*2])* motor_effort_ratio.unsqueeze(0), dim=-1)
    else:
        assert False, f"observation shape {obs_buf.shape[-1]} not exist"

    # reward for duration of staying alive
    alive_reward = torch.ones_like(obs_buf[...,0]) * alive_reward_scale

    target_tracking_reward = torch.exp(-torch.norm(xys, dim=-1)) * 0.1

    total_reward = (
        target_tracking_reward
        + alive_reward
        + up_reward
        # + heading_reward # no heading reward for ant central symmetry
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
    )

    # adjust reward for fallen agents
    total_reward = torch.where(
        torch.logical_or(z_proj_wood < 0.8, z_woods < 0.85), torch.ones_like(total_reward) * death_cost, total_reward
    )
    return total_reward

# @torch.jit.script
def is_done(
    obs_buf,
    termination_height,
    reset_buf,
    progress_buf,
    max_episode_length,
    z_woods,
    z_proj_wood
):
    # type: (Tensor, float, Tensor, Tensor, float, Tensor, Tensor) -> Tensor
    reset = torch.where(torch.logical_or(z_proj_wood < 0.8, z_woods < 2.+ termination_height), torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    return reset