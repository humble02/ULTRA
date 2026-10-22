import torch
from isaacgym.torch_utils import *

class SmoothRewards():
    def __init__(self, env):
        self.env = env
        self.reward_scales = self.env.cfg['smooth_rewards']['scales']
        if self.env.cfg['smooth_rewards']['scales'] is not None:
            self.reward_functions = []
            self.reward_names = []
            self.reward = []
            self.reward_buf = []
            for name, scale in self.env.cfg['smooth_rewards']['scales'].items():
                # if name == "termination":
                #     continue
                self.reward_names.append(name)
                name = "_reward_" + name
                self.reward_functions.append(getattr(self, name))
        else:
            raise ValueError("Smooth rewards scales must be defined in the configuration.")

    def _compute_smooth_reward(self):
        rew_buf = 0.0
        self.reward.clear()
        self.reward_buf.clear()
        # print("Smooth rewards:")
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]()
            self.reward.append(rew)
            # print(name, (rew * self.reward_scales[name]).mean().item())
            rew_buf += rew * self.reward_scales[name]
            self.reward_buf.append(rew * self.reward_scales[name])
            # print(name, (rew * self.reward_scales[name]).mean().item())

        # if "termination" in self.reward_scales:
        #     rew = self._reward_termination() * self.reward_scales["termination"]
        #     rew_buf += rew
        #     self.reward_buf.append(rew)
        #     print("termination", rew.mean().item())


        return rew_buf
    
    def _reward_base_lin_vel(self):
        return torch.norm(self.env._rigid_body_vel[:, 0, :], dim=-1)

    def _reward_ang_vel(self):
        return torch.sum(torch.square(self.env._rigid_body_ang_vel[:, 0, :]), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.env._dof_vel), dim=1)

    def _reward_action_rate(self):
        # TODO
        return torch.norm(self.env.last_actions - self.env.actions, dim=-1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.env.last_dof_vel - self.env._dof_vel) / self.env.dt), dim=1)

    def _reward_dof_acc_all(self):
        dof_vel_all = torch.stack(self.env.dof_vel_list, dim=0)
        # print(dof_vel_all)
        return torch.sum(torch.sum(torch.square((dof_vel_all[1:] - dof_vel_all[:1]) / self.env.dt), dim=0), dim=1)
    
    def _reward_delta_dof_vel_all(self):
        dof_vel_all = torch.stack(self.env.dof_vel_list, dim=0)
        # print(dof_vel_all)
        return torch.sum(torch.sum(torch.square((dof_vel_all[1:] - dof_vel_all[:1])), dim=0), dim=1)

    def _reward_delta_ang_vel_all(self):
        ang_vel_all = torch.stack(self.env.ang_vel_list, dim=0)
        # print(dof_vel_all)
        return torch.sum(torch.sum(torch.square((ang_vel_all[1:] - ang_vel_all[:1])), dim=0), dim=1)

    def _reward_torques(self):
        return torch.norm(self.env.torques, dim=-1)
    
    def _reward_torques_vel_all(self):
        torques_all = torch.stack(self.env.torques_list, dim=0)
        # print(dof_vel_all)
        return torch.sum(torch.sum(torch.square((torques_all[1:] - torques_all[:1]) / self.env.dt), dim=0), dim=1)
    
    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.env._dof_pos - self.env.dof_limits_lower).clip(max=0.)  # lower limit
        out_of_limits += (self.env._dof_pos - self.env.dof_limits_upper).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_dof_torque_limits(self):
        out_of_limits = torch.sum((torch.abs(self.env.torques) / self.env.torque_limits - self.env.cfg['smooth_rewards']['soft_torque_limit']).clip(min=0), dim=1)
        return out_of_limits
    
    def _reward_energy(self):
        return torch.norm(torch.abs(self.env.torques * self.env._dof_vel), dim=-1)
    
    def _reward_feet_orientation(self):
        left_quat = self.env._rigid_body_rot[:, self.env.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.env.gravity_vec)
        right_quat = self.env._rigid_body_rot[:, self.env.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.env.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1) **0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5

    def _reward_feet_contact_forces(self):
        rew = torch.norm(self.env._contact_forces[:, self.env.feet_indices, 2], dim=-1)
        rew[rew < self.env.cfg['smooth_rewards']['max_contact_force']] = 0
        rew[rew > self.env.cfg['smooth_rewards']['max_contact_force']] -= self.env.cfg['smooth_rewards']['max_contact_force']
        return rew
    
    def _reward_feet_stumble(self):
        rew = torch.any(
            torch.norm(self.env._contact_forces[:, self.env.feet_indices, :2], dim=2)
            > 4 * torch.abs(self.env._contact_forces[:, self.env.feet_indices, 2]),
            dim=1,
        )
        return rew.float()

    def _reward_foot_slip(self):
        contact = self.env._contact_forces[:, self.env.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(self.env._rigid_body_vel[:, self.env.feet_indices, :2], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        feet_on_ground_1 = self.env._curr_ref_obs[:,357:474].view(-1, 39, 3)[:, self.env.feet_indices, :].norm(dim=-1) < 0.04
        feet_on_ground_2 = self.env._curr_ref_obs[:,84:201].view(-1, 39, 3)[:, self.env.feet_indices, 2] < 0.05

        rew *= contact
        return torch.sum(rew, dim=1)
    
    def _reward_feet_distance(self):
        foot_pos = self.env._rigid_body_pos[:, self.env.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.env.cfg['smooth_rewards']['min_dist']
        max_df = self.env.cfg['smooth_rewards']['max_dist']
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        # d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        d_max = torch.clamp(foot_dist - max_df, 0, 1.0)  # NOTE: changed from 0.5 to 1.0
        return (torch.abs(d_min) * 100 + torch.abs(d_max) * 100) / 2
    
    def _reward_knee_distance(self):
        foot_pos = self.env._rigid_body_pos[:, self.env.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.env.cfg['smooth_rewards']['min_dist']
        max_df = self.env.cfg['smooth_rewards']['max_knee_dist']
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.abs(d_min) * 100 + torch.abs(d_max) * 100) / 2
    
    def _reward_collision(self):
        return torch.sum(
            1.0
            * (torch.norm(self.env._contact_forces[:, self.env.penalized_contact_indices, :], dim=-1) > 0.1),
            dim=1,
        )
        
    def _reward_stand_on_feet(self):
        # reward for standing on both feet
        contact = torch.norm(self.env._contact_forces[:, self.env.feet_indices], dim=-1) > 2.0
        stand_on_both = torch.sum(contact, dim=1) == 2
        feet_on_ground = self.env._curr_ref_obs[:,84:201].view(-1, 39, 3)[:, self.env.feet_indices, 2] < 0.04
        feet_on_ground_both = torch.sum(feet_on_ground, dim=1) == 2

        # Check if we're in the last 20 copied frames of the motion
        # max_episode_length includes the 20 extended frames
        # So frames in the last 20 frames are: progress_buf >= (max_episode_length - 20)
        in_extended_frames = self.env.progress_buf >= (self.env.max_episode_length[self.env.data_id] - 20)

        # Original reward: penalize if reference feet are on ground but robot is not standing
        basic_rew = (feet_on_ground_both & ~stand_on_both).float()

        # In extended frames: require robot to stand on both feet (stronger requirement)
        extended_rew = (~stand_on_both).float()

        # Use extended reward in last 20 frames, basic reward otherwise
        rew = torch.where(in_extended_frames, extended_rew, basic_rew)
        return rew

    def _reward_contact_change(self):
        """
        Penalize changes in contact state (contact on/off switches).
        This encourages stable contact patterns during locomotion.
        """
        # Get current contact forces for feet
        current_contact = torch.norm(self.env._contact_forces[:, :, :], dim=-1)  # [B, F]

        # # Get previous contact state (needs to be stored in env)
        # if not hasattr(self.env, '_prev_contact'):
        #     # Initialize on first call
        #     self.env._prev_contact = current_contact.clone()
        #     return torch.zeros(current_contact.shape[0], device=current_contact.device)

        # Calculate contact change (0 if same, 1 if changed)
        contact_change = torch.abs(current_contact - torch.norm(self.env.last_contact_forces, dim=-1))  # [B, F]

        # Sum across all feet
        rew = contact_change.sum(dim=1)  # [B]

        # Update previous contact for next iteration
        self.env._prev_contact = current_contact.clone()

        return rew 
    
    def _reward_dof_pos(self):
        dof_pos_prior = self.env._dof_pos[..., [1, 2, 3, 7, 8, 9]].abs().sum(dim=1)  # lower limit
        return dof_pos_prior
    
    def _reward_swing_clearance(self):
        """
        Reward: when a reference foot is moving (swing), the simulated foot should be off the ground.
        Returns: [batch] reward.
        Assumes:
        - self.env.feet_indices: list/tensor of foot body indices (F feet)
        - self.env._rigid_body_pos: [B, bodies, 3] simulated rigid body positions
        - (Option A) self.env._curr_ref_vel: [B, bodies, 3] reference body velocities (preferred if available)
        - (Option B) self.env._curr_ref_obs: flattened reference body positions; we finite-difference it and
            cache self.env._prev_ref_obs each step. The slice 357:474 -> (39 * 3) matches your layout.
        - self.env.dt: simulation timestep
        - (optional) self.env.contact_forces: [B, bodies, 3]
        """
        B = self.env._rigid_body_pos.shape[0]
        feet = self.env.feet_indices

        # Check if we're in the last 20 extended frames
        in_extended_frames = self.env.progress_buf >= (self.env.max_episode_length[self.env.data_id] - 20)

        ref_speed = torch.linalg.norm(self.env._curr_ref_obs[:, 357:474].view(B, 39, 3)[:, feet, :2], dim=-1)                        # [B, F]

        # Smooth swing gate: ~0 when slow, ~1 when above threshold
        v_thresh = 0.10  # m/s, tune
        beta = 0.05      # softness for the sigmoid gate
        swing_gate = torch.sigmoid((ref_speed - v_thresh) / (beta + 1e-8))    # [B, F]
        swing_gate[ref_speed < 0.05] = 0

        ref_height = self.env._curr_ref_obs[:,84:201].view(-1, 39, 3)[:, self.env.feet_indices, 2]

        h_thresh = 0.07  # m/s, tune
        beta = 0.04      # softness for the sigmoid gate
        height_gate = torch.sigmoid((ref_height - h_thresh) / (beta + 1e-8))    # [B, F]
        height_gate[ref_height < 0.04] = 0

        # --- 2) Simulated foot clearance term (smooth hinge above a target) ---
        sim_z = self.env._rigid_body_pos[:, feet, 2]                          # [B, F]
        clearance_target = 0.06  # meters; "off ground" target height; tune to your foot radius/terrain
        alpha = 0.02            # softness (meters) for smooth hinge (softplus)
        # softplus((z - target)/alpha) * alpha ≈ ReLU(z - target) but smooth near the knee
        height_term = torch.nn.functional.softplus((clearance_target - sim_z) / (alpha + 1e-8)) * alpha  # [B, F]
        # --- 3) Optional: no-contact bonus during swing (if contact sensors available) ---
        cf = self.env._contact_forces[:, feet, :]                          # [B, F, 3]
        contact = (torch.linalg.norm(cf, dim=-1) > 5.0).float()           # [B, F], 1 if in contact                             # 1 when no contact

        # --- 4) Combine terms and weight by reference swing speed (capped) ---
        w_height = 1.0
        w_noc = 0.6
        w_speed = 0.3
        speed_scale = torch.clamp(ref_speed, 0.0, 1.0)                        # [B, F]

        per_foot = torch.maximum(swing_gate, height_gate) * (w_height * height_term + w_noc * contact) * (1.0 + w_speed * speed_scale)  # [B, F]
        # print(sim_z, height_term, contact)
        rew = per_foot.sum(dim=1)                                             # [B]

        # In extended frames, do not penalize for not lifting feet (even if reference appears to be swinging)
        # The robot should keep both feet on ground to prepare for standing
        rew = torch.where(in_extended_frames, torch.zeros_like(rew), rew)

        return rew
    def _reward_termination(self):
        # self.reset_buf * ~self.time_out_buf

        # Goal-deadline failure can carry a penalty while the episode continues.
        # Keep the physical termination flag separate for correct PPO bootstrap.
        return getattr(self.env, "_reward_terminate_buf", self.env._terminate_buf) * 1.0
