import time
import torch
from env.tasks.ultra_g1_retarget import (
    UltraG1Retarget,
    compute_humanoid_reset as compute_humanoid_reset_retarget,
)
from env.tasks.humanoid_g1 import *
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *

from utils import torch_utils
from utils.box_pca import box_pca_axes
from rl_games.algos_torch import torch_ext
import torch.nn as nn
import torch.nn.functional as F
from learning import ultra_network_builder, ultra_models
import os
import numpy as np
import math


def get_all_paths(dir_path):
    paths = []
    for root, dirs, files in os.walk(dir_path):
        for name in files:
            paths.append(os.path.join(root, name))
    return paths

def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:, 0]
    y = quat_angle[:, 1]
    z = quat_angle[:, 2]
    w = quat_angle[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

class UltraDistillObjV2Point(UltraG1Retarget):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        # Reuse the 4096-env training frame, including in one-env inference.
        self._object_pca_axes = box_pca_axes(self.object_corners)
        self.action_buf = torch.zeros(
            (self.num_envs, 29), device=self.device, dtype=torch.float)
        self.mu_buf = torch.zeros(
            (self.num_envs, 29), device=self.device, dtype=torch.float)
        self.obs_buf_student = torch.zeros(
            (self.num_envs, cfg["env"]["numObsStudent"]), device=self.device, dtype=torch.float)
        self.obs_buf = torch.zeros(
            (self.num_envs, cfg["env"]["numObs"]), device=self.device, dtype=torch.float)
        self.num_obs = cfg["env"]["numObsStudent"]
        self.models = []
        self.running_means = []
        self.running_vars = []
        self.obs_buf = torch.zeros(
            (self.num_envs, cfg["env"]["numObs"]), device=self.device, dtype=torch.float)
        self.num_obs = cfg["env"]["numObsStudent"]
        obs_shape = cfg["env"]["numObs"]
        config = {
            'actions_num' : 29,
            'input_shape' : (obs_shape, ),
            'num_seqs' : cfg["env"]["numEnvs"] * 1,
            'value_size': 1,
        }
        print(config)
        network = ultra_network_builder.UltraBuilder()
        params = {
            "model": {
                "name": "ultra"
            },
            "network": {
                "name": "ultra",
                "separate": True,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {
                            "name": "default"
                        },
                        "sigma_init": {
                            "name": "const_initializer",
                            "val": -2.9
                        },
                        "fixed_sigma": True,
                        "learn_sigma": False
                    }
                },
                "mlp": {
                    "units": [1024, 1024, 512],
                    "activation": "relu",
                    "d2rl": False,
                    "initializer": {
                        "name": "default"
                    },
                    "regularizer": {
                        "name": "None"
                    }
                }
            }
        }
        network.load(params['network'])
        network = ultra_models.ModelUltraContinuous(network)
        model_path = cfg["env"]["teacherPolicy"]
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"env.teacherPolicy='{model_path}' not found (resolved relative to {os.getcwd()}). "
                "Train the Stage-2 teacher first (scripts/train_teacher.sh) or place a teacher checkpoint at this "
                "path, then set env.teacherPolicy in ultra/data/cfg/g1_student_vae.yaml.")
        ck = torch_ext.load_checkpoint(model_path)
        model = network.build(config)
        model.to(self.device)
        model.load_state_dict(ck['model'])
        self.model = model
        running_mean, running_var = ck['running_mean_std']['running_mean'], ck['running_mean_std']['running_var']
        self.running_mean = running_mean
        self.running_var = running_var

        self.head_camera_body_id = self._resolve_body_id(['d435_link', 'head_link'])
        env_cfg = cfg.get("env", {})
        # Point cloud domain randomization parameters
        self.point_noise_std = env_cfg.get("pointNoiseStd", 0.0)
        self.point_dropout_prob = env_cfg.get("pointDropoutProb", 0.0)
        self.point_outlier_prob = env_cfg.get("pointOutlierProb", 0.0)
        self.point_outlier_scale = env_cfg.get("pointOutlierScale", 0.5)
        self.point_depth_noise_scale = env_cfg.get("pointDepthNoiseScale", 0.0)
        self.point_density_min = env_cfg.get("pointDensityMin", 1.0)
        self.point_density_max = env_cfg.get("pointDensityMax", 1.0)
        self.point_cluster_noise_std = env_cfg.get("pointClusterNoiseStd", 0.0)
        self.point_scale_min = env_cfg.get("pointScaleMin", 1.0)
        self.point_scale_max = env_cfg.get("pointScaleMax", 1.0)
        self.point_translation_noise = env_cfg.get("pointTranslationNoise", 0.0)
        self.point_occlusion_prob = env_cfg.get("pointOcclusionProb", 0.0)
        self.camera_rot_noise = env_cfg.get("cameraRotNoise", 0.0)
        self.camera_pos_noise = env_cfg.get("cameraPosNoise", 0.0)
        self.point_fixed_grid_sampling = env_cfg.get("pointFixedGridSampling", False)
        grid_res = env_cfg.get("pointSurfaceGridResolution", 0)
        self.point_surface_grid_resolution = int(grid_res) if grid_res and grid_res > 0 else None
        self._camera_debug_points = None
        self._camera_debug_env = 0
        # Debug visualization geometries
        if hasattr(gymutil, "WireframeSphere"):
            # Green for visible/randomized points
            self._camera_debug_geom = gymutil.WireframeSphere(radius=0.015, color=(0.2, 0.8, 0.2))
            # Blue for raw visible points (before randomization)
            self._camera_debug_geom_raw = gymutil.WireframeSphere(radius=0.012, color=(0.2, 0.2, 0.8))
            # Red for camera position
            self._camera_debug_geom_cam = gymutil.WireframeSphere(radius=0.03, color=(0.9, 0.2, 0.2))
            # Yellow for PCA corners
            self._camera_debug_geom_corner = gymutil.WireframeSphere(radius=0.02, color=(0.9, 0.9, 0.2))
            # Orange for the (masked-in) object goal shown in task_mode=object_obs playback
            self._goal_debug_geom = gymutil.WireframeSphere(radius=0.03, color=(0.9, 0.6, 0.1))
        elif hasattr(gymutil, "WireframeSphereGeometry"):
            self._camera_debug_geom = gymutil.WireframeSphereGeometry(
                0.015, 8, 8, None, color=(0.2, 0.8, 0.2)
            )
            self._camera_debug_geom_raw = gymutil.WireframeSphereGeometry(
                0.012, 8, 8, None, color=(0.2, 0.2, 0.8)
            )
            self._camera_debug_geom_cam = gymutil.WireframeSphereGeometry(
                0.03, 8, 8, None, color=(0.9, 0.2, 0.2)
            )
            self._camera_debug_geom_corner = gymutil.WireframeSphereGeometry(
                0.02, 8, 8, None, color=(0.9, 0.9, 0.2)
            )
            self._goal_debug_geom = gymutil.WireframeSphereGeometry(
                0.03, 8, 8, None, color=(0.9, 0.6, 0.1)
            )
        else:
            self._camera_debug_geom = None
            self._camera_debug_geom_raw = None
            self._camera_debug_geom_cam = None
            self._camera_debug_geom_corner = None
            self._goal_debug_geom = None
        # Additional debug data
        self._camera_debug_raw_points = None  # Visible points before randomization
        self._camera_debug_camera_pos = None  # Camera position
        self._camera_debug_pca_corners = None  # PCA bounding box corners in world frame
        self._camera_debug_frustum_lines = None  # Camera frustum lines

        # Video recording
        self._video_frames = []
        self._video_writer = None
        self._video_fps = 60  # Match data fps
        self._video_output_path = None

        self.long_term_t = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)
        self.long_term_speed_threshold = cfg['env'].get('long_term_speed_threshold', 0.2)
        self.ig_interaction_threshold = env_cfg.get("igInteractionThreshold", 0.15)
        self.ig_transition_delta = env_cfg.get("igTransitionDelta", 0.0001)
        self.ig_hand_body_ids = env_cfg.get("igHandBodyIds", [28, 38])
        self.prev_hand_ig = torch.zeros(self.num_envs, device=self.device)
        self.prev_hand_ig_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_phase = torch.zeros((self.num_envs, 4), device=self.device)
        # self.use_sparse_long_term_condition = env_cfg.get("useSparseLongTermCondition", False)
        # self.long_term_condition_threshold = env_cfg.get("longTermConditionThreshold", 0.05)
        # self._sparse_condition_active = False
        # Observation masking probability schedule configuration
        # Strategy: Start with more information (high keep prob), gradually reduce to sparse observations
        self.obs_mask_schedule_start = cfg['env'].get('obs_mask_schedule_start', 500)
        self.obs_mask_schedule_duration = cfg['env'].get('obs_mask_schedule_duration', 5000)

        # Initial probabilities (early training - provide more info to bootstrap learning)
        self.obj_rot_keep_prob_init = cfg['env'].get('obs_obj_rot_keep_prob_init', 0.8)
        self.obj_trans_keep_prob_init = cfg['env'].get('obs_obj_trans_keep_prob_init', 0.9)
        self.obj_pos_keep_prob_init = cfg['env'].get('obs_obj_pos_keep_prob_init', 0.8)
        self.obj_point_keep_prob_init = cfg['env'].get('obs_obj_point_keep_prob_init', 0.8)
        self.human_move_keep_prob_init = cfg['env'].get('obs_human_move_keep_prob_init', 0.9)
        self.human_global_keep_prob_init = cfg['env'].get('obs_human_global_keep_prob_init', 0.9)
        self.human_local_keep_prob_init = cfg['env'].get('obs_human_local_keep_prob_init', 0.5)
        self.human_goal_keep_prob_init = cfg['env'].get('obs_human_goal_keep_prob_init', 0.9)

        # Final probabilities (late training - sparse observations for robustness)
        # Match typical deployment modes like sparse_track or object_obs
        self.obj_rot_keep_prob_final = cfg['env'].get('obs_obj_rot_keep_prob_final', 0.1)
        self.obj_trans_keep_prob_final = cfg['env'].get('obs_obj_trans_keep_prob_final', 0.5)
        self.obj_pos_keep_prob_final = cfg['env'].get('obs_obj_pos_keep_prob_final', 0.3)
        self.obj_point_keep_prob_final = cfg['env'].get('obs_obj_point_keep_prob_final', 0.3)
        self.human_move_keep_prob_final = cfg['env'].get('obs_human_move_keep_prob_final', 0.4)
        self.human_global_keep_prob_final = cfg['env'].get('obs_human_global_keep_prob_final', 0.4)
        self.human_local_keep_prob_final = cfg['env'].get('obs_human_local_keep_prob_final', 0.2)
        self.human_goal_keep_prob_final = cfg['env'].get('obs_human_goal_keep_prob_final', 0.3)

        # Current probabilities (will be updated via schedule)
        self.obj_rot_keep_prob = self.obj_rot_keep_prob_init
        self.obj_trans_keep_prob = self.obj_trans_keep_prob_init
        self.obj_pos_keep_prob = self.obj_pos_keep_prob_init
        self.obj_point_keep_prob = self.obj_point_keep_prob_init
        self.human_move_keep_prob = self.human_move_keep_prob_init
        self.human_global_keep_prob = self.human_global_keep_prob_init
        self.human_local_keep_prob = self.human_local_keep_prob_init
        self.human_goal_keep_prob = self.human_goal_keep_prob_init

        # Track current epoch for scheduling
        self.current_epoch = 0
        # Per-environment flags for which observations to keep (set at episode start)
        self.keep_obj_point_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.keep_obj_trans_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.keep_obj_rot_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.keep_obj_pos_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.keep_human_move_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.keep_goal_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.keep_global_goal_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.keep_local_goal_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.obs_mask_noise_std = cfg['env'].get('obs_mask_noise_std', 0.01)
        self.keep_mask_flip_prob = cfg['env'].get('obs_mask_flip_prob', 1e-3)
        self.obj_deviation_threshold = env_cfg.get("obj_deviation_threshold", 0.5)
        self.obj_deviated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._teacher_task_slice = None
        self._teacher_ig_slice = None
        self.debug_obs_layout = env_cfg.get("debug_obs_layout", False)
        self._debug_obs_layout_printed = False
        self.num_dof = int(self._dof_pos.shape[-1]) if hasattr(self, "_dof_pos") else int(cfg["env"].get("numDoF", 29))
        self.global_goal_dim = 3
        self.local_goal_dim = 2 + self.num_dof

        # Optional fixed observation preset for playback (full_track / sparse_track / object_obs).
        # When unset, modality masks are sampled per episode from the keep probabilities (training).
        self.task_mode = None
        self.obj_obs_mode = None
        self._task_mode_keep = None
        self._goal_debug_pos_buf = torch.zeros((self.num_envs, 3), device=self.device)
        self._goal_debug_enabled_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._goal_debug_max_envs = int(env_cfg.get("debug_viz_max_envs", 64))
        task_mode = env_cfg.get("task_mode")
        if task_mode:
            self.configure_task_mode(task_mode, env_cfg.get("obj_obs", "points"))
        return

    def update_obs_keep_probabilities(self, epoch):
        """
        Update observation keep probabilities with a smooth cosine schedule.
        Decreases probabilities from initial (high info) to final (sparse info) over training.

        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        if getattr(self, "_task_mode_keep", None) is not None:
            # A fixed task_mode preset is active (playback); keep the frozen probabilities.
            return None

        # Compute progress with cosine schedule for smooth transition
        progress = min(max(0, epoch - self.obs_mask_schedule_start) / self.obs_mask_schedule_duration, 1.0)
        # Cosine schedule: smooth S-curve from 0 to 1
        cosine_progress = (1 - math.cos(progress * math.pi)) / 2

        # Interpolate from initial to final probabilities (decreasing over time)
        self.obj_rot_keep_prob = self.obj_rot_keep_prob_init + (self.obj_rot_keep_prob_final - self.obj_rot_keep_prob_init) * cosine_progress
        self.obj_trans_keep_prob = self.obj_trans_keep_prob_init + (self.obj_trans_keep_prob_final - self.obj_trans_keep_prob_init) * cosine_progress
        self.obj_pos_keep_prob = self.obj_pos_keep_prob_init + (self.obj_pos_keep_prob_final - self.obj_pos_keep_prob_init) * cosine_progress
        self.obj_point_keep_prob = self.obj_point_keep_prob_init + (self.obj_point_keep_prob_final - self.obj_point_keep_prob_init) * cosine_progress
        self.human_move_keep_prob = self.human_move_keep_prob_init + (self.human_move_keep_prob_final - self.human_move_keep_prob_init) * cosine_progress
        self.human_global_keep_prob = self.human_global_keep_prob_init + (self.human_global_keep_prob_final - self.human_global_keep_prob_init) * cosine_progress
        self.human_local_keep_prob = self.human_local_keep_prob_init + (self.human_local_keep_prob_final - self.human_local_keep_prob_init) * cosine_progress
        self.human_goal_keep_prob = self.human_goal_keep_prob_init + (self.human_goal_keep_prob_final - self.human_goal_keep_prob_init) * cosine_progress

        return {
            'epoch': epoch,
            'progress': progress,
            'obj_rot_keep_prob': self.obj_rot_keep_prob,
            'obj_trans_keep_prob': self.obj_trans_keep_prob,
            'obj_pos_keep_prob': self.obj_pos_keep_prob,
            'obj_point_keep_prob': self.obj_point_keep_prob,
            'human_move_keep_prob': self.human_move_keep_prob,
            'human_global_keep_prob': self.human_global_keep_prob,
            'human_local_keep_prob': self.human_local_keep_prob,
            'human_goal_keep_prob': self.human_goal_keep_prob,
        }

    def _resolve_body_id(self, candidate_names):
        """Find the first available rigid body index from a list of names."""
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        for name in candidate_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, name)
            if body_id != -1:
                return body_id
        return -1

    def _get_head_camera_pose(self, env_ids, root_states):
        """Return (position, rotation) tensors for the robot head camera."""
        if self.head_camera_body_id >= 0:
            if env_ids is None:
                cam_pos = self._rigid_body_pos[:, self.head_camera_body_id, :]
                cam_rot = self._rigid_body_rot[:, self.head_camera_body_id, :]
            else:
                cam_pos = self._rigid_body_pos[env_ids][:, self.head_camera_body_id, :]
                cam_rot = self._rigid_body_rot[env_ids][:, self.head_camera_body_id, :]
        else:
            cam_pos = root_states[:, 0:3]
            cam_rot = root_states[:, 3:7]
        return cam_pos, cam_rot

    def _maybe_flip_keep_masks(self, env_ids):
        """Occasionally toggle keep masks to avoid fixed patterns at inference."""
        if self.keep_mask_flip_prob <= 0 or env_ids.numel() == 0:
            return
        for mask in (
            self.keep_obj_point_mask,
            self.keep_obj_trans_mask,
            self.keep_obj_rot_mask,
            self.keep_obj_pos_mask,
            self.keep_human_move_mask,
            self.keep_global_goal_mask,
            self.keep_local_goal_mask,
            self.keep_goal_mask,
        ):
            flips = torch.rand(env_ids.shape[0], device=self.device) < self.keep_mask_flip_prob
            if flips.any():
                mask_vals = mask[env_ids]
                mask_vals = torch.where(flips, ~mask_vals, mask_vals)
                mask[env_ids] = mask_vals

    def _apply_task_mode_masks(self, env_ids):
        """Force the modality masks of env_ids to the active task_mode preset. Returns False if no preset is set."""
        keep = getattr(self, "_task_mode_keep", None)
        if keep is None or env_ids.numel() == 0:
            return False
        self.keep_obj_point_mask[env_ids] = keep["obj_point"]
        self.keep_obj_trans_mask[env_ids] = keep["obj_trans"]
        self.keep_obj_rot_mask[env_ids] = keep["obj_rot"]
        self.keep_obj_pos_mask[env_ids] = keep["obj_pos"]
        self.keep_human_move_mask[env_ids] = keep["human_move"]
        self.keep_global_goal_mask[env_ids] = keep["global_goal"]
        self.keep_local_goal_mask[env_ids] = keep["local_goal"]
        self.keep_goal_mask[env_ids] = keep["goal"]
        return True

    def configure_task_mode(self, task_mode, obj_obs="points"):
        """Select which input modalities the student sees (used for playback / sim2sim parity).

        full_track  : dense reference tracking, every modality visible.
        sparse_track: sparse goal following, the local reference and object pose/points are hidden.
        object_obs  : goal-conditioned from object observations only (obj_obs = points | pos | none).
        """
        task_mode = str(task_mode).lower()
        obj_obs = str(obj_obs).lower()
        if task_mode in ("full_traj", "full_trajectory", "fulltraj"):
            task_mode = "full_track"
        print("[UltraDistillObjV2Point] task_mode =", task_mode, "obj_obs =", obj_obs)
        keep = {
            "obj_point": True,
            "obj_trans": True,
            "obj_rot": True,
            "obj_pos": True,
            "human_move": True,
            "global_goal": True,
            "local_goal": True,
            "goal": True,
        }
        if task_mode == "full_track":
            pass  # all visible, no masking
        elif task_mode == "sparse_track":
            keep["obj_pos"] = False
            keep["obj_point"] = False
            keep["local_goal"] = False
            keep["obj_rot"] = False
        elif task_mode == "object_obs":
            keep["local_goal"] = False
            keep["global_goal"] = False
            keep["obj_rot"] = False
            if obj_obs == "pos":
                keep["obj_point"] = False
            elif obj_obs == "points":
                keep["obj_pos"] = False
            elif obj_obs == "none":
                keep["obj_pos"] = False
                keep["obj_point"] = False
            else:
                raise ValueError(f"Unknown obj_obs mode: {obj_obs}")
        else:
            raise ValueError(f"Unknown task_mode preset: {task_mode}")

        self.task_mode = task_mode
        self.obj_obs_mode = obj_obs
        self._task_mode_keep = keep
        # Freeze the mask curriculum so the preset is not overridden by random flips.
        self.keep_mask_flip_prob = 0.0
        self.obj_rot_keep_prob = 1.0 if keep["obj_rot"] else 0.0
        self.obj_trans_keep_prob = 1.0 if keep["obj_trans"] else 0.0
        self.obj_pos_keep_prob = 1.0 if keep["obj_pos"] else 0.0
        self.obj_point_keep_prob = 1.0 if keep["obj_point"] else 0.0
        self.human_move_keep_prob = 1.0 if keep["human_move"] else 0.0
        self.human_global_keep_prob = 1.0 if keep["global_goal"] else 0.0
        self.human_local_keep_prob = 1.0 if keep["local_goal"] else 0.0
        self.human_goal_keep_prob = 1.0 if keep["goal"] else 0.0
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._apply_task_mode_masks(env_ids)

    def _finalize_student_obs(self, env_ids, obs, obs_shape, task_obs_size):
        """Expose mask bits, reorder goal segments, and leave raw observations intact."""
        self._maybe_flip_keep_masks(env_ids)
        mask_features = []

        def append_mask(mask_tensor, length):
            if length <= 0:
                return
            mask = mask_tensor[env_ids].float().unsqueeze(-1)
            mask_features.append(mask.expand(-1, length))

        global_goal_dim = min(self.global_goal_dim, obs.shape[-1])
        local_goal_dim = min(
            self.local_goal_dim,
            max(obs_shape - global_goal_dim, 0),
        )

        global_goal = torch.empty((obs.shape[0], 0), device=obs.device, dtype=obs.dtype)
        if global_goal_dim > 0:
            append_mask(self.keep_global_goal_mask, global_goal_dim)
            global_goal = obs[:, :global_goal_dim]

        local_goal = torch.empty((obs.shape[0], 0), device=obs.device, dtype=obs.dtype)
        if local_goal_dim > 0:
            start = global_goal_dim
            end = start + local_goal_dim
            append_mask(self.keep_local_goal_mask, local_goal_dim)
            local_goal = obs[:, start:end]

        task_start = obs_shape
        if task_obs_size > 0:
            task_end = obs_shape + task_obs_size
            trans_end = min(task_start + 3, task_end)
            if trans_end > task_start:
                append_mask(self.keep_obj_trans_mask, trans_end - task_start)
            rot_start = trans_end
            rot_end = min(rot_start + 6, task_end)
            if rot_end > rot_start:
                append_mask(self.keep_obj_rot_mask, rot_end - rot_start)
            pos_start = rot_end
            pos_end = min(pos_start + 3, task_end)
            if pos_end > pos_start:
                append_mask(self.keep_obj_pos_mask, pos_end - pos_start)
            point_start = pos_end
            if task_end > point_start:
                append_mask(self.keep_obj_point_mask, task_end - point_start)
        goal = self.goal_phase[env_ids]
        append_mask(self.keep_goal_mask, goal.shape[-1])

        humanoid_rest_start = global_goal_dim + local_goal_dim
        humanoid_rest = obs[:, humanoid_rest_start:obs_shape]
        task_part = obs[:, obs_shape:]
        remainder = torch.cat([humanoid_rest, task_part], dim=-1)

        components = [global_goal, goal]
        if local_goal_dim > 0:
            components.append(local_goal)
        components.append(remainder)
        combined = torch.cat(components, dim=-1)
        if mask_features:
            combined = torch.cat([combined, torch.cat(mask_features, dim=-1)], dim=-1)
        if self.debug_obs_layout and not self._debug_obs_layout_printed:
            mask_len = 0
            if mask_features:
                mask_len = sum(int(feat.shape[-1]) for feat in mask_features)
            print(
                "[UltraEnv] student_obs dims: global_goal={} command={} local_goal={} humanoid_rest={} "
                "task_obs={} mask={} total={}".format(
                    int(global_goal.shape[-1]),
                    int(goal.shape[-1]),
                    int(local_goal.shape[-1]),
                    int(humanoid_rest.shape[-1]),
                    int(task_part.shape[-1]),
                    mask_len,
                    int(combined.shape[-1]),
                )
            )
            self._debug_obs_layout_printed = True
        return combined

    def _compute_observations_student(self, env_ids=None):
        new_episode = torch.logical_or((self.progress_buf[env_ids] <= self.start_times[env_ids] + 1), self.long_term_t[env_ids] <= 0)
        # print(self.long_term_t.shape)
        # print(env_ids.shape)
        # print(new_episode.shape)
        new_ids = env_ids[new_episode]
        if new_ids.numel() > 0:
            ts = self.progress_buf[new_ids]
            ref_vel = self.hoi_refs[self.data_id[new_ids], 0, ts, 78:81]
            ref_speed = torch.norm(ref_vel, dim=-1)
            use_long = ref_speed < self.long_term_speed_threshold
            long_horizon = torch.randint_like(new_ids, 120) + 120
            short_horizon = torch.randint_like(new_ids, 60) + 60
            self.long_term_t[new_ids] = torch.where(use_long, long_horizon, short_horizon)
        if (env_ids is None):
            self.obs_buf_student[:] = self._compute_observations_iter(None, self.long_term_t, student_obs=True)
        else:
            self.obs_buf_student[env_ids] = self._compute_observations_iter(env_ids, self.long_term_t[env_ids], student_obs=True)
        return    

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(58+5 + self._num_actions * 3, device=self.device, dtype=torch.float)
        if not self.cfg['noise']['add_noise']:
            return noise_vec
        noise_scales = self.cfg['noise']['noise_scales']
        noise_level = self.cfg['noise']['noise_level']
        noise_vec[0:58] = noise_scales['dof_pos'] * noise_level
        noise_vec[58:58+3] = noise_scales['ang_vel'] * noise_level
        noise_vec[58+3:58+5] = noise_scales['imu'] * noise_level
        noise_vec[58+5 : 58+5 + self._num_actions] = (
            noise_scales['dof_pos'] * noise_level
        )
        noise_vec[58+5 + self._num_actions : 58+5 + self._num_actions * 2] = (
            noise_scales['dof_vel'] * noise_level
        )
        return noise_vec         

    def _compute_humanoid_obs(self, env_ids=None, ref_obs=None, next_ts=None, student_obs=False, local_ref_obs=None):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            contact_forces = self._contact_forces
            actions = self.action_history_buf
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            torques = self.torques
            last_dof_pos = self.last_dof_pos
            last_dof_vel = self.last_dof_vel
            humanoid_root_states = self._humanoid_root_states
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            contact_forces = self._contact_forces[env_ids]
            actions = self.action_history_buf[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            torques = self.torques[env_ids]
            last_dof_pos = self.last_dof_pos[env_ids]
            last_dof_vel = self.last_dof_vel[env_ids]
            humanoid_root_states = self._humanoid_root_states[env_ids]


        obs = self.compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                self._root_height_obs,
                                                contact_forces, self._contact_body_ids, ref_obs, self._key_body_ids,
                                                self._key_body_ids_gt, self._contact_body_ids_gt, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel, humanoid_root_states, env_ids, student_obs=student_obs, local_ref_obs=local_ref_obs)

        return obs
    
    def compute_humanoid_observations_max(self, body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids, ref_obs, key_body_ids, key_body_ids_gt, contact_body_ids_gt, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel, humanoid_root_states, env_ids, student_obs=False, local_ref_obs=None):
        root_pos = self._rand_vec(body_pos[:, 0, :], 0.0)
        root_rot = self._rand_vec(body_rot[:, 0, :], 0.)

        root_h = root_pos[:, 2:3]
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        if (not root_height_obs):
            root_h_obs = torch.zeros_like(root_h)
        else:
            root_h_obs = root_h

        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand_2 = heading_rot_expand.repeat((1, 39, 1))
        flat_heading_rot_2 = heading_rot_expand_2.reshape(heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1], 
                                                heading_rot_expand_2.shape[2])
        
        heading_rot_expand = heading_rot_expand.repeat((1, 39, 1))
        flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                heading_rot_expand.shape[2])

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, 39, 1))
        flat_heading_inv_rot = heading_inv_rot_expand.reshape(heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                heading_inv_rot_expand.shape[2])
        
        _ref_body_pos = ref_obs[:,84:201].view(-1, 39, 3)
        _body_pos = self._rand_vec(body_pos, 0.0)

        diff_global_body_pos = _ref_body_pos - _body_pos
        diff_local_body_pos_flat = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, 39 * 3)

        local_ref_body_pos = _ref_body_pos - root_pos.unsqueeze(1)  # preserves the body position
        local_ref_body_pos = torch_utils.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, 39 * 3)

        root_pos_expand = root_pos.unsqueeze(-2)
        local_body_pos = self._rand_vec(body_pos, 0.0) - root_pos_expand
        flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
        flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
        local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
        local_body_pos = local_body_pos[..., 3:] # remove root pos

        flat_body_rot = self._rand_vec(body_rot.reshape(body_rot.shape[0] * 39, body_rot.shape[2]), 0.)
        flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
        flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], 39 * flat_local_body_rot_obs.shape[1])
        
        ref_body_rot = ref_obs[:, 201:357].view(-1, 39, 4)
        ref_body_rot_no_hand = ref_body_rot
        body_rot_no_hand = self._rand_vec(body_rot, 0.)

        diff_global_body_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
        diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(flat_heading_rot, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot)
        diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
        diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

        local_ref_body_rot = torch_utils.quat_mul(flat_heading_rot, ref_body_rot_no_hand.reshape(-1, 4))
        local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

        ref_body_vel = ref_obs[:, 357:474].view(-1, 39, 3)
        # body_vel = self._rand_vec(body_vel, 0.1)
        diff_global_vel = ref_body_vel - body_vel
        diff_local_vel = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, 39 * 3)

        ref_body_ang_vel = ref_obs[:, 474:591]
        ref_body_ang_vel_no_hand = ref_body_ang_vel.view(-1, 39, 3)
        body_ang_vel_no_hand = body_ang_vel
        diff_global_ang_vel = ref_body_ang_vel_no_hand - body_ang_vel_no_hand
        diff_local_ang_vel = torch_utils.quat_rotate(flat_heading_rot, diff_global_ang_vel.view(-1, 3)).view(-1, 39 * 3)

        if (local_root_obs):
            root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
            local_body_rot_obs[..., 0:6] = root_rot_obs

        flat_body_vel = body_vel.reshape(body_vel.shape[0] * 39, body_vel.shape[2])
        flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
        local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], 39 * body_vel.shape[2])
        
        flat_body_ang_vel = body_ang_vel_no_hand.reshape(body_ang_vel.shape[0] * 39, body_ang_vel.shape[2])
        flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
        local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], 39 * body_ang_vel.shape[2])

        body_contact_buf = contact_forces.clone() #.view(contact_forces.shape[0],-1)
        contact = (torch.abs(body_contact_buf).sum(dim=-1) > 0.1).float()
        ref_body_contact = ref_obs[:,591:630]
        diff_body_contact = ref_body_contact - contact
        contact_new = torch.zeros_like(contact).to(contact.device)
        diff_body_contact_new = torch.zeros_like(diff_body_contact).to(contact.device)
        contact_new[:, [6, 7, 13, 14, 28, 38]] = contact[:, [6, 7, 13, 14, 28, 38]]
        diff_body_contact_new[:, [6, 7, 13, 14, 28, 38]] = diff_body_contact[:, [6, 7, 13, 14, 28, 38]]

        if student_obs:
            if local_ref_obs is None:
                local_ref_obs = ref_obs
            # obs_prop shape: [nuv_envs, 320] (1771 - 320 = 1451)
            root_ang_vel = humanoid_root_states[:, 10:13]
            ref_root_ang_vel = ref_body_ang_vel[:, 0:3]
            diff_root_ang_vel = ref_root_ang_vel - root_ang_vel
            base_quat = humanoid_root_states[:, 3:7]
            ref_base_quat = ref_body_rot[:, 0]
            roll, pitch, yaw = euler_from_quaternion(base_quat)
            roll_ref, pitch_ref, yaw_ref = euler_from_quaternion(ref_base_quat)
            imu_obs = torch.stack((roll, pitch), dim=1)
            imu_obs_all = torch.stack((roll, pitch, yaw), dim=1)
            imu_obs_ref_all = torch.stack((roll_ref, pitch_ref, yaw_ref), dim=1)
            diff_imu_obs_all = imu_obs_ref_all - imu_obs_all
            # imu_obs_ref = torch.stack((roll_ref, pitch_ref), dim=1)
            # diff_imu_obs_ref = imu_obs_ref - imu_obs
            dof_pos = self._rand_vec(dof_pos, 0.0)
            ref_dof_pos = ref_obs[:, 13:42]
            ref_dof_pos_local = local_ref_obs[:, 13:42]
            diff_dof_pos = dof_pos - ref_dof_pos
            ref_base_quat_local = local_ref_obs[:, 3:7]
            roll_ref_local, pitch_ref_local, yaw_ref_local = euler_from_quaternion(ref_base_quat_local)
            imu_obs_ref_all_local = torch.stack((roll_ref_local, pitch_ref_local, yaw_ref_local), dim=1)
            diff_imu_obs_all_local = imu_obs_ref_all_local - imu_obs_all
            root_pos = _body_pos[:, 0, :].clone()
            ref_root_pos = _ref_body_pos[:, 0, :].clone()
            diff_root_pos = root_pos - ref_root_pos 
            diff_root_pos_xy = diff_root_pos.clone()[..., :2]
            local_goal = torch.cat((diff_imu_obs_all_local[..., 0:2], dof_pos - ref_dof_pos_local), dim=-1)
            global_goal = torch.cat((diff_root_pos_xy, diff_imu_obs_all[..., 2:3]), dim=-1)
            _dof_vel = dof_vel.clone()
            _dof_vel[..., 12:15] *= 0.05
            # ref_root_ang_vel.zero_()
            # diff_root_ang_vel.zero_()
            # imu_obs_ref.zero_()
            # diff_imu_obs_ref.zero_()
            # diff_local_body_pos_flat.zero_()
            if self.cfg['env']['history_len'] > 0:
                # # obs shape: [nuv_envs, 2450]
                # print(root_ang_vel.shape[1] + ref_root_ang_vel.shape[1] + diff_root_ang_vel.shape[1] + imu_obs.shape[1] + imu_obs_ref.shape[1] + diff_imu_obs_ref.shape[1])
                obs_prop = torch.cat((global_goal, local_goal, root_ang_vel, imu_obs, dof_pos, _dof_vel, actions[:, -1, :], self.obs_history_buf.view(self.num_envs, -1)[env_ids]), dim=-1)
                # 405=29+29+29+3+2+29+29 + 92 * 25
                obs_buf = torch.cat(
                    (
                        root_ang_vel, # 3 | 70 [67:70]
                        imu_obs,
                        self.reindex((dof_pos)),
                        self.reindex(_dof_vel),
                        actions[:, -1, :],
                    ),
                    dim=-1,
                )
                if self.cfg['noise']['add_noise'] and self.headless:
                    noise = (2 * torch.rand_like(self.noise_scale_vec) - 1) * self.noise_scale_vec * min(
                        self.common_step_counter / (self.cfg['noise']['noise_increasing_steps'] * 32), 1.0
                    )
                    obs_buf += noise[58:]
                    obs_prop[:, 34:126] += noise[58:]
                elif self.cfg['noise']['add_noise'] and not self.headless:
                    noise = (2 * torch.rand_like(self.noise_scale_vec) - 1) * self.noise_scale_vec
                    obs_buf += noise[58:]
                    obs_prop[:, 34:126] += noise[58:]
                else:
                    obs_buf += 0.0

                if env_ids.shape[0] == self.num_envs:
                    self.obs_history_buf = torch.where(
                        (self.episode_length_buf <= 1)[:, None, None],
                        torch.stack([obs_buf] * self.cfg['env']['history_len'], dim=1),
                        torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1),
                    )
                else:
                    self.obs_history_buf[env_ids] = torch.stack([obs_buf] * self.cfg['env']['history_len'], dim=1)
                return obs_prop
            else:
                obs_prop = torch.cat((global_goal, local_goal, ref_dof_pos, diff_dof_pos, root_ang_vel, imu_obs, dof_pos, dof_vel, actions[:, -1, :]), dim=-1)
                return obs_prop
        else:
            # # obs shape: [nuv_envs, 1771]
            # # local_body_vel = local_body_vel * 0
            # # local_body_vel = local_body_vel * 0
            # local_body_ang_vel = local_body_ang_vel * 0
            # # diff_local_vel = diff_local_vel * 0
            # diff_local_ang_vel = diff_local_ang_vel * 0
            # # actions = actions * 0
            # # dof_vel = dof_vel * 0
            torques = torques * 0
            # # last_dof_vel = last_dof_vel * 0
            # print(root_h_obs.shape[1] + local_body_pos.shape[1] + local_body_rot_obs.shape[1] + local_body_vel.shape[1] + local_body_ang_vel.shape[1] + contact_new.shape[1] + diff_local_body_pos_flat.shape[1] + diff_local_body_rot_obs.shape[1] + diff_body_contact_new.shape[1] + local_ref_body_pos.shape[1] + local_ref_body_rot.shape[1] + diff_local_vel.shape[1] + diff_local_ang_vel.shape[1] + actions.shape[1])
            obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact_new, diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact_new, local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel, actions[:, -1, :], self._rand_vec(dof_pos, 0.0), self._rand_vec(dof_vel, 0.), torques, self._rand_vec(last_dof_pos, 0.0), self._rand_vec(last_dof_vel, 0.)), dim=-1)
            return obs


    def _compute_observations_iter(self, env_ids=None, delta_t=1, student_obs=False):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
            ts = self.progress_buf.clone()
            # Clamp ts to valid range to avoid index out of bounds during data replay
            ts = torch.clamp(ts, max=self.max_episode_length[self.data_id[env_ids]]-1)
            self._curr_ref_obs = self.hoi_data[self.data_id[env_ids], ts].clone()
            next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]]-1)
            next_ts_local = torch.clamp(ts + 1, max=self.max_episode_length[self.data_id[env_ids]]-1)
            # For "stand still" environments, use the fixed frame instead of advancing
            next_ts = torch.where(self.is_stand_still, self.stand_still_frame, next_ts)
            next_ts_local = torch.where(self.is_stand_still, self.stand_still_frame, next_ts_local)
            self._curr_ref_obs[next_ts<=ts] = self.hoi_data[self.data_id[next_ts<=ts], next_ts[next_ts<=ts]].clone()
            ref_obs = self.hoi_data[self.data_id[env_ids], next_ts].clone()
            ref_obs_local = self.hoi_data[self.data_id[env_ids], next_ts_local].clone()
            self._update_goal_debug(ref_obs, env_ids)
            obs = self._compute_humanoid_obs(env_ids, ref_obs, next_ts, student_obs, local_ref_obs=ref_obs_local)
            obs_shape = obs.shape[-1]
            task_obs, obj_points = self._compute_task_obs(env_ids, ref_obs, student_obs)
            obs = torch.cat([obs, task_obs], dim=-1)
            ig_all, ig_norm = self._compute_ig_features(env_ids, obj_points, student_obs)
            self._update_goal_phase(env_ids, ig_norm)
            if not student_obs:
                self._update_object_deviation(env_ids, obj_points, ref_obs, ig_all)
                ig_size = ig_all.view(env_ids.shape[0], -1).shape[-1]
                self._teacher_task_slice = (obs_shape, obs_shape + task_obs.shape[-1])
                self._teacher_ig_slice = (
                    obs_shape + task_obs.shape[-1],
                    obs_shape + task_obs.shape[-1] + ig_size * 2,
                )

            if student_obs:
                task_obs_size = task_obs.shape[-1] if task_obs is not None else 0
                return self._finalize_student_obs(env_ids, obs, obs_shape, task_obs_size)

            ig_features = ig_all.view(env_ids.shape[0], -1)
            ref_ig = ref_obs[:, 630:].view(env_ids.shape[0], 39, 3)
            ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
            ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()  
            ref_ig = ref_ig.view(env_ids.shape[0], -1)          
            obs = torch.cat((obs, ig_features, ref_ig - ig_features), dim=-1)


        else:
            ts = self.progress_buf[env_ids].clone()
            # Clamp ts to valid range to avoid index out of bounds during data replay
            ts = torch.clamp(ts, max=self.max_episode_length[self.data_id[env_ids]]-1)
            self._curr_ref_obs[env_ids] = self.hoi_data[self.data_id[env_ids], ts].clone()
            next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]]-1)
            next_ts_local = torch.clamp(ts + 1, max=self.max_episode_length[self.data_id[env_ids]]-1)
            # For "stand still" environments, use the fixed frame instead of advancing
            next_ts = torch.where(self.is_stand_still[env_ids], self.stand_still_frame[env_ids], next_ts)
            next_ts_local = torch.where(self.is_stand_still[env_ids], self.stand_still_frame[env_ids], next_ts_local)
            self._curr_ref_obs[env_ids[next_ts<=ts]] = self.hoi_data[self.data_id[env_ids[next_ts<=ts]], next_ts[next_ts<=ts]].clone()
            ref_obs = self.hoi_data[self.data_id[env_ids], next_ts].clone()
            ref_obs_local = self.hoi_data[self.data_id[env_ids], next_ts_local].clone()
            self._update_goal_debug(ref_obs, env_ids)
            obs = self._compute_humanoid_obs(env_ids, ref_obs, next_ts, student_obs, local_ref_obs=ref_obs_local)
            obs_shape = obs.shape[-1]
            task_obs, obj_points = self._compute_task_obs(env_ids, ref_obs, student_obs)
            obs = torch.cat([obs, task_obs], dim=-1)
            ig_all, ig_norm = self._compute_ig_features(env_ids, obj_points, student_obs)
            self._update_goal_phase(env_ids, ig_norm)
            if not student_obs:
                self._update_object_deviation(env_ids, obj_points, ref_obs, ig_all)
                ig_size = ig_all.view(env_ids.shape[0], -1).shape[-1]
                self._teacher_task_slice = (obs_shape, obs_shape + task_obs.shape[-1])
                self._teacher_ig_slice = (
                    obs_shape + task_obs.shape[-1],
                    obs_shape + task_obs.shape[-1] + ig_size * 2,
                )
            if student_obs:
                task_obs_size = task_obs.shape[-1] if task_obs is not None else 0
                return self._finalize_student_obs(env_ids, obs, obs_shape, task_obs_size)
            ig_features = ig_all.view(env_ids.shape[0], -1)
            ref_ig = ref_obs[:, 630:].view(env_ids.shape[0], 39, 3)
            ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
            ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()  
            ref_ig = ref_ig.view(env_ids.shape[0], -1)          
            obs = torch.cat((obs, ig_features, ref_ig - ig_features), dim=-1)
            return obs


        return
    
    def _compute_task_obs(self, env_ids=None, ref_obs=None, is_student=False):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_states = self._target_states
            current_object_ids = self.object_id[self.data_id]
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]
            current_object_ids = self.object_id[self.data_id[env_ids]]

        camera_pos, camera_rot = self._get_head_camera_pose(env_ids, root_states)

        # Use PCA corner observations if enabled
        if is_student:
            # Get precomputed PCA corners for current objects
            pca_corners = self.object_corners[current_object_ids]
            obs, obj_points = compute_obj_observations_pca_corners(
                root_states, tar_states, pca_corners, ref_obs, camera_pos, camera_rot,
                point_noise_std=self.point_noise_std,
                point_dropout_prob=self.point_dropout_prob,
                point_outlier_prob=self.point_outlier_prob,
                point_outlier_scale=self.point_outlier_scale,
                point_depth_noise_scale=self.point_depth_noise_scale,
                point_density_min=self.point_density_min,
                point_density_max=self.point_density_max,
                point_cluster_noise_std=self.point_cluster_noise_std,
                point_scale_min=self.point_scale_min,
                point_scale_max=self.point_scale_max,
                point_translation_noise=self.point_translation_noise,
                point_occlusion_prob=self.point_occlusion_prob,
                camera_rot_noise=self.camera_rot_noise,
                camera_pos_noise=self.camera_pos_noise,
                fixed_surface_sampling=self.point_fixed_grid_sampling,
                surface_grid_resolution=self.point_surface_grid_resolution,
                pca_axes=self._object_pca_axes[current_object_ids],
            )
        else:
            # Use original point cloud observations
            obs, obj_points = compute_obj_observations(root_states, tar_states, self.object_points[current_object_ids], ref_obs)

        return obs, obj_points
    
    def _compute_ig_features(self, env_ids, obj_points, student_obs):
        if env_ids is None:
            key_body_pose = self._rigid_body_pos.clone()
            body_rot = self._rigid_body_rot[:, 0, :]
        else:
            key_body_pose = self._rigid_body_pos[env_ids].clone()
            body_rot = self._rigid_body_rot[env_ids][:, 0, :]

        if student_obs:
            key_body_pose = self._rand_vec(key_body_pose, 0.01)

        ig = compute_sdf(key_body_pose, obj_points).view(-1, 3)
        heading_rot = torch_utils.calc_heading_quat_inv(self._rand_vec(body_rot, 0.1))
        heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1], 1).view(-1, 4)
        ig = quat_rotate(heading_rot_extend, ig).view(key_body_pose.shape[0], -1, 3)
        ig_norm = ig.norm(dim=-1, keepdim=True)
        ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
        return ig_all, ig_norm

    def _update_object_deviation(self, env_ids, obj_points, ref_obs, ig_all):
        """Update object deviation mask using reset_ig + contact_reset + object_reset."""
        if env_ids is None or obj_points is None or ig_all is None:
            return
        current_object_ids = self.object_id[self.data_id[env_ids]]
        object_points = self.object_points[current_object_ids]
        ref_obj_pos = ref_obs[:, 71:74]
        ref_obj_rot = ref_obs[:, 74:78]
        obj_rot_extend = ref_obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        ref_obj_points = (
            torch_utils.quat_rotate(obj_rot_extend, object_points_extend)
            .view(ref_obj_rot.shape[0], object_points.shape[1], 3)
            + ref_obj_pos.unsqueeze(1)
        )
        max_dist = (obj_points - ref_obj_points).norm(dim=-1).max(dim=-1)[0]
        object_reset = max_dist > self.obj_deviation_threshold

        ref_ig = ref_obs[:, 630:].view(env_ids.shape[0], 39, 3)
        ig_diff = (ig_all - ref_ig).pow(2).sum(dim=-1).sqrt()
        denom_ref = torch.clamp(ref_ig.pow(2).sum(dim=-1).sqrt(), min=0.5)
        denom_ig = torch.clamp(ig_all.pow(2).sum(dim=-1).sqrt(), min=0.5)
        reset_ig_1 = (ig_diff / denom_ref).max(dim=-1)[0].max(dim=-1)[0] > 1.0
        reset_ig_2 = (ig_diff / denom_ig).max(dim=-1)[0].max(dim=-1)[0] > 1.0
        reset_ig = torch.logical_or(reset_ig_1, reset_ig_2)

        if hasattr(self, "contact_reset"):
            contact_reset_flag = torch.any(self.contact_reset[env_ids] > 19, dim=-1)
        else:
            contact_reset_flag = torch.zeros_like(reset_ig, dtype=torch.bool)

        self.obj_deviated[env_ids] = object_reset | reset_ig | contact_reset_flag

    def _mask_teacher_obs(self, obs):
        """Mask object/IG parts for teacher policy when object deviates."""
        if self._teacher_task_slice is None or self._teacher_ig_slice is None:
            return obs
        teacher_obs = obs.clone()
        deviated = self.obj_deviated
        if deviated.any():
            task_start, task_end = self._teacher_task_slice
            ig_start, ig_end = self._teacher_ig_slice
            teacher_obs[deviated, task_start:task_end] = 0.0
            teacher_obs[deviated, ig_start:ig_end] = 0.0
        return teacher_obs

    def _update_goal_phase(self, env_ids, ig_norm):
        ig_norm = ig_norm.squeeze(-1)
        hand_norm = ig_norm[..., self.ig_hand_body_ids]
        current = hand_norm.mean(dim=-1)

        prev = torch.where(
            self.prev_hand_ig_valid[env_ids],
            self.prev_hand_ig[env_ids],
            current,
        )
        delta = prev - current

        near_mask = current <= self.ig_interaction_threshold
        phase_mask = self.progress_buf[env_ids] > (self.max_episode_length[self.data_id[env_ids]] // 2)

        approaching = (~near_mask & (delta > self.ig_transition_delta)).float()
        leaving = (~near_mask & (delta < -self.ig_transition_delta)).float()

        steady_mask = (~near_mask) & (delta.abs() <= self.ig_transition_delta)
        approaching = torch.where(steady_mask, (~phase_mask).float(), approaching)
        leaving = torch.where(steady_mask, phase_mask.float(), leaving)

        interacting = near_mask.float()
        goal_vec = torch.stack((approaching, leaving), dim=-1)

        self.prev_hand_ig[env_ids] = current.detach()
        self.prev_hand_ig_valid[env_ids] = True
        remaining = (self.long_term_t[env_ids]).float()
        max_len = 240
        time_to_go = torch.clamp(remaining / (max_len + 1e-6), 0.0, 1.0).unsqueeze(1)
        self.goal_phase[env_ids] = torch.cat(
            [
                (self.progress_buf[env_ids] > (max_len - 20)).float().unsqueeze(1),
                goal_vec,
                time_to_go,
            ],
            dim=-1,
        )


        return goal_vec

    def _update_goal_debug(self, ref_obs, env_ids):
        """Cache the object goal position for the debug viewer (task_mode=object_obs only)."""
        if self.viewer is None or not self.debug_viz:
            return
        if getattr(self, "task_mode", None) != "object_obs":
            return
        if ref_obs is None or ref_obs.numel() == 0:
            return
        if env_ids is None or env_ids.numel() == 0:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        obj_goal = ref_obs[:, 71:74]
        self._goal_debug_pos_buf[env_ids] = obj_goal.detach()
        if self.keep_obj_trans_mask is not None:
            self._goal_debug_enabled_buf[env_ids] = self.keep_obj_trans_mask[env_ids]
        else:
            self._goal_debug_enabled_buf[env_ids] = True

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        # self._num_actions = self.num_prim
        return

    # def get_task_obs_size_detail(self):
    #     task_obs_detail = super().get_task_obs_size_detail()
    #     task_obs_detail['num_prim'] = self.num_prim
    #     return task_obs_detail


    def step(self, weights):

        self.pre_physics_step(weights)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        with torch.no_grad():
            curr_obs = ((self.obs_buf - self.running_mean.float().to(self.device)) / torch.sqrt(self.running_var.float().to(self.device) + 1e-05))
            curr_obs = torch.clamp(curr_obs, min=-5.0, max=5.0)
            self.model.eval()
            teacher_obs = self._mask_teacher_obs(curr_obs)
            input_dict = {
                'is_train': False,
                'prev_actions': None, 
                'obs' : teacher_obs,
                'rnn_states' : None
            }
            res_dict = self.model(input_dict)
            teacher_action = torch.clamp(res_dict['actions'], min=-1.0, max=1.0)
            mu = res_dict['mus']
            self.action_buf = teacher_action     
            self.mu_buf = mu
            # print('env', self.obs_buf.shape)
        

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)


    def reset(self, env_ids=None):
        super().reset(env_ids=env_ids)
        if isinstance(env_ids, list):
            env_ids = to_torch(env_ids, device=self.device, dtype=torch.long)
        elif env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        num_reset_envs = env_ids.shape[0]
        self.prev_hand_ig_valid[env_ids] = False
        self.prev_hand_ig[env_ids] = 0.0
        self.goal_phase[env_ids] = 0.0
        if num_reset_envs > 0:
            if not self._apply_task_mode_masks(env_ids):
                rand_vals = torch.rand(num_reset_envs, device=self.device)
                self.keep_obj_point_mask[env_ids] = rand_vals < self.obj_point_keep_prob
                rand_vals = torch.rand(num_reset_envs, device=self.device)
                self.keep_obj_trans_mask[env_ids] = rand_vals < self.obj_trans_keep_prob
                rand_vals = torch.rand(num_reset_envs, device=self.device)
                self.keep_obj_rot_mask[env_ids] = rand_vals < self.obj_rot_keep_prob
                rand_vals = torch.rand(num_reset_envs, device=self.device)
                self.keep_obj_pos_mask[env_ids] = rand_vals < self.obj_pos_keep_prob
                rand_vals = torch.rand(num_reset_envs, device=self.device)
                self.keep_human_move_mask[env_ids] = rand_vals < self.human_move_keep_prob
                rand_vals = torch.rand(num_reset_envs, device=self.device)
                self.keep_global_goal_mask[env_ids] = rand_vals < self.human_global_keep_prob
                rand_vals = torch.rand(num_reset_envs, device=self.device)
                self.keep_local_goal_mask[env_ids] = rand_vals < self.human_local_keep_prob
                rand_vals = torch.rand(num_reset_envs, device=self.device)
                self.keep_goal_mask[env_ids] = rand_vals < self.human_goal_keep_prob
        
        
        # if self.object_history_buf is not None and env_ids.shape[0] > 0:
        #     self.object_history_buf[env_ids] = 0.0
        if env_ids.shape[0] > 0:
            self._compute_observations_student(env_ids)
        with torch.no_grad():
            curr_obs = ((self.obs_buf - self.running_mean.float().to(self.device)) / torch.sqrt(self.running_var.float().to(self.device) + 1e-05))
            curr_obs = torch.clamp(curr_obs, min=-5.0, max=5.0)
            self.model.eval()
            teacher_obs = self._mask_teacher_obs(curr_obs)
            input_dict = {
                'is_train': False,
                'prev_actions': None, 
                'obs' : teacher_obs,
                'rnn_states' : None
            }
            res_dict = self.model(input_dict)
            # print(res_dict)
            teacher_action = torch.clamp(res_dict['actions'], min=-1.0, max=1.0)
            mu = res_dict['mus']
            self.action_buf = teacher_action     
            self.mu_buf = mu    

        return
    
    def post_physics_step(self):
        super().post_physics_step()
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self.long_term_t -= 1
        self._compute_observations_student(env_ids)
        return

    def _compute_reset(self):
        reset_ig = torch.zeros_like(self.reset_buf, dtype=torch.bool)
        contact_reset = torch.zeros_like(self.reset_buf, dtype=torch.bool)
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset_retarget(
            self.reset_buf,
            self.progress_buf,
            self.obs_buf,
            self._contact_forces,
            self._rigid_body_pos,
            self.max_episode_length[self.data_id],
            self._enable_early_termination,
            self._termination_heights,
            self._termination_heights_init,
            self._curr_ref_obs,
            self._curr_obs,
            self.start_times,
            self.rollout_length,
            reset_ig,
            contact_reset,
            torch.logical_or(
                self.is_stand_still,
                self.progress_buf >= self.max_episode_length[self.data_id] - 20,
            ),
        )
        if hasattr(self, "obs_history_buf"):
            self.obs_history_buf[self.reset_buf > 0] *= 0
        return
    
    def _set_state_from_dataset_frame(self, time):
        """Apply a dataset frame to the sim state without stepping physics."""
        t = int(time)
        if t == 0:
            self.data_id = to_torch([
                torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][
                    torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())
                ]
                for i in range(self.num_envs)
            ], device=self.device, dtype=torch.long)

        env_ids = to_torch(
            [i for i in range(self.num_envs) if t < self.max_episode_length[self.data_id[i]]],
            device=self.device,
            dtype=torch.long
        )
        if env_ids.numel() == 0:
            return env_ids

        self._target_states[env_ids, :3] = self.hoi_refs[self.data_id[env_ids], 0, t, 71:74]
        self._target_states[env_ids, 3:7] = self.hoi_refs[self.data_id[env_ids], 0, t, 74:78]
        self._target_states[env_ids, 7:10] = self.hoi_refs[self.data_id[env_ids], 0, t, 78:81]
        self._target_states[env_ids, 10:13] = self.hoi_refs[self.data_id[env_ids], 0, t, 81:84]

        _humanoid_root_pos = self.hoi_refs[self.data_id[env_ids], 0, t, 0:3]
        _humanoid_root_pos[..., 2:3] += 0.02
        _humanoid_root_rot = self.hoi_refs[self.data_id[env_ids], 0, t, 3:7]
        self._humanoid_root_states[env_ids, 0:3] = _humanoid_root_pos
        self._humanoid_root_states[env_ids, 3:7] = _humanoid_root_rot
        self._humanoid_root_states[env_ids, 7:10] = self.hoi_refs[self.data_id[env_ids], 0, t, 7:10]
        self._humanoid_root_states[env_ids, 10:13] = self.hoi_refs[self.data_id[env_ids], 0, t, 10:13]

        self._dof_pos[env_ids] = self.hoi_refs[self.data_id[env_ids], 0, t, 13:42]
        self._dof_vel[env_ids] = self.hoi_refs[self.data_id[env_ids], 0, t, 42:71]

        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        self._refresh_sim_tensors()
        return env_ids

    def start_video_recording(self, output_path=None, fps=60):
        """Start recording video frames.

        Args:
            output_path: Path to save the video (e.g., 'output.mp4'). If None, auto-generates path.
            fps: Frames per second for the output video.
        """
        if output_path is None:
            dataname = self.motion_file[-1].split('/')[-1].replace('.pt', '') if self.motion_file else 'replay'
            os.makedirs("ultra/data/videos", exist_ok=True)
            output_path = f"ultra/data/videos/{dataname}_replay.mp4"

        self._video_frames = []
        self._video_fps = fps
        self._video_output_path = output_path
        print(f"[VIDEO] Started recording. Will save to: {output_path}")

    def capture_frame(self):
        """Capture current viewer frame for video recording."""
        if self.viewer is None or self._video_output_path is None:
            return

        # Try direct frame capture first
        try:
            viewport = self.gym.get_viewer_size(self.viewer)
            width, height = viewport.x, viewport.y
            self.gym.render_all_camera_sensors(self.sim)
            frame = self.gym.get_viewer_image(self.viewer, self.sim)

            if frame is not None:
                frame_np = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 4)
                frame_rgb = frame_np[:, :, :3]
                self._video_frames.append(frame_rgb.copy())
                return
        except Exception:
            pass

        # Fallback: save to temp file and read back
        import tempfile
        temp_path = os.path.join(tempfile.gettempdir(), f"_isaacgym_frame_{len(self._video_frames):05d}.png")
        try:
            self.gym.write_viewer_image_to_file(self.viewer, temp_path)
            from PIL import Image
            img = Image.open(temp_path)
            frame_rgb = np.array(img.convert('RGB'))
            self._video_frames.append(frame_rgb)
            os.remove(temp_path)
        except Exception as e:
            print(f"[VIDEO] Warning: Failed to capture frame: {e}")

    def stop_video_recording(self):
        """Stop recording and save the video file."""
        if not self._video_frames or self._video_output_path is None:
            print("[VIDEO] No frames captured or recording not started.")
            return None

        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False

        output_path = self._video_output_path

        if has_cv2:
            # Use OpenCV for video writing
            height, width = self._video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self._video_fps, (width, height))

            for frame in self._video_frames:
                # OpenCV uses BGR format
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            writer.release()
            print(f"[VIDEO] Saved {len(self._video_frames)} frames to: {output_path}")
        else:
            # Fallback: try imageio
            try:
                import imageio
                imageio.mimwrite(output_path, self._video_frames, fps=self._video_fps)
                print(f"[VIDEO] Saved {len(self._video_frames)} frames to: {output_path} (using imageio)")
            except ImportError:
                # Last resort: save as individual frames
                frames_dir = output_path.replace('.mp4', '_frames')
                os.makedirs(frames_dir, exist_ok=True)
                from PIL import Image
                for i, frame in enumerate(self._video_frames):
                    img = Image.fromarray(frame)
                    img.save(os.path.join(frames_dir, f"frame_{i:05d}.png"))
                print(f"[VIDEO] OpenCV/imageio not available. Saved {len(self._video_frames)} frames to: {frames_dir}/")
                print(f"        Use ffmpeg to convert: ffmpeg -framerate {self._video_fps} -i {frames_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p {output_path}")

        # Clear frames
        self._video_frames = []
        self._video_output_path = None
        return output_path

    def play_dataset_step(self, time, visualize_camera_points=True, camera_env_id=0, camera_apply_randomization=False):
        """Play a single frame from the dataset with optional point cloud visualization.

        Args:
            time: Frame index to play
            visualize_camera_points: Whether to visualize the point cloud
            camera_env_id: Which environment to visualize
            camera_apply_randomization: Whether to apply domain randomization to points.
                                        Set to False for clean debugging of geometry/camera.

        Debug visualization legend:
            - GREEN spheres: Final points (after randomization if enabled)
            - BLUE spheres: Raw visible points (always without randomization)
            - RED sphere: Camera position (d435_link)
            - YELLOW spheres: PCA bounding box corners in world frame
            - WHITE lines: Camera frustum
            - CYAN line: Camera forward direction
        """
        t = int(time)

        # Set state from dataset (this initializes data_id at t=0)
        env_ids = self._set_state_from_dataset_frame(time)

        if env_ids.numel() > 0:
            # IMPORTANT: Set progress_buf to current frame to avoid index out of bounds
            # in _compute_observations_iter when step() is called
            self.progress_buf[env_ids] = t

            # Get actions from dataset - use data_id to index correctly for num_envs
            self.actions = self.hoi_refs[self.data_id[env_ids], 0, t, 13:42]

            # Pad actions if not all envs have data for this frame
            if self.actions.shape[0] < self.num_envs:
                full_actions = torch.zeros(self.num_envs, 29, device=self.device)
                full_actions[env_ids] = self.actions
                self.actions = full_actions

            # Step physics (but clamp progress_buf to avoid going past data length)
            self.step(self.actions)

            # After step, clamp progress_buf to valid range for each env
            for i in range(self.num_envs):
                max_len = self.max_episode_length[self.data_id[i]] - 1
                self.progress_buf[i] = min(self.progress_buf[i].item(), max_len)

        if visualize_camera_points:
            self.visualize_dataset_camera_points(
                frame_idx=time,
                env_id=camera_env_id,
                apply_randomization=camera_apply_randomization,
                use_current_state=True
            )

        # Capture frame for video if recording is active
        if self._video_output_path is not None:
            self.capture_frame()

    def visualize_dataset_camera_points(self, frame_idx, env_id=0, apply_randomization=True, use_current_state=False):
        """Visualize the sampled camera points for a dataset frame in the viewer.

        Debug visualization shows:
        - GREEN spheres: Final points (randomized if apply_randomization=True, else visible)
        - BLUE spheres: Raw visible points (before domain randomization)
        - RED sphere: Camera position
        - YELLOW spheres: PCA bounding box corners in world frame
        - WHITE lines: Camera frustum and viewing direction
        """
        if self.viewer is None:
            if not getattr(self, "_warned_no_viewer", False):
                print("Viewer is not initialized; camera-point debug rendering is skipped in headless mode.")
                self._warned_no_viewer = True
            return

        if use_current_state:
            env_ids = to_torch([env_id], device=self.device, dtype=torch.long)
        else:
            env_ids = self._set_state_from_dataset_frame(frame_idx)
            if env_ids.numel() == 0:
                print(f"No dataset data available at frame {frame_idx}.")
                return

        if env_id >= self.num_envs:
            print(f"Env id {env_id} is out of range (num_envs={self.num_envs}).")
            return

        if not use_current_state and not (env_ids == env_id).any():
            print(f"Env id {env_id} does not contain data for frame {frame_idx}.")
            return

        env_tensor = to_torch([env_id], device=self.device, dtype=torch.long)
        root_states = self._humanoid_root_states[env_tensor]
        tar_states = self._target_states[env_tensor]
        current_object_ids = self.object_id[self.data_id[env_tensor]]
        pca_corners = self.object_corners[current_object_ids]
        ref_obs = self.hoi_refs[self.data_id[env_tensor], 0, frame_idx].to(self.device)
        camera_pos, camera_rot = self._get_head_camera_pose(env_tensor, root_states)

        with torch.no_grad():
            # BLUE points: Clean point cloud (no geometry noise, no domain randomization)
            # This shows the TRUE visible points based on actual object pose and camera
            _, clean_visible_world, _ = compute_obj_observations_pca_corners(
                root_states,
                tar_states,
                pca_corners.clone(),  # Clone to avoid mutation
                ref_obs,
                camera_pos,
                camera_rot,
                point_noise_std=0.0,  # No point noise
                point_dropout_prob=0.0,
                point_outlier_prob=0.0,
                point_depth_noise_scale=0.0,
                point_density_min=1.0,
                point_density_max=1.0,
                point_cluster_noise_std=0.0,
                point_scale_min=1.0,
                point_scale_max=1.0,
                point_translation_noise=0.0,
                point_occlusion_prob=0.0,
                camera_rot_noise=0.0,  # No camera noise
                camera_pos_noise=0.0,
                return_randomized_points=True,
                disable_geometry_noise=True,  # Disable all geometry perturbation
                fixed_surface_sampling=self.point_fixed_grid_sampling,
                surface_grid_resolution=self.point_surface_grid_resolution
            )

            # GREEN points: Noisy point cloud (with all domain randomization)
            # This shows what the policy actually sees during training
            _, _, noisy_randomized_world = compute_obj_observations_pca_corners(
                root_states,
                tar_states,
                pca_corners.clone(),
                ref_obs,
                camera_pos,
                camera_rot,
                point_noise_std=self.point_noise_std,
                point_dropout_prob=self.point_dropout_prob,
                point_outlier_prob=self.point_outlier_prob,
                point_outlier_scale=self.point_outlier_scale,
                point_depth_noise_scale=self.point_depth_noise_scale,
                point_density_min=self.point_density_min,
                point_density_max=self.point_density_max,
                point_cluster_noise_std=self.point_cluster_noise_std,
                point_scale_min=self.point_scale_min,
                point_scale_max=self.point_scale_max,
                point_translation_noise=self.point_translation_noise,
                point_occlusion_prob=self.point_occlusion_prob,
                camera_rot_noise=self.camera_rot_noise,
                camera_pos_noise=self.camera_pos_noise,
                return_randomized_points=True,
                disable_geometry_noise=False,  # Enable geometry perturbation
                fixed_surface_sampling=self.point_fixed_grid_sampling,
                surface_grid_resolution=self.point_surface_grid_resolution
            )

        # Store debug data
        # BLUE: Clean visible points (true geometry, correct occlusion)
        # GREEN: Noisy/randomized points (what policy sees during training)
        self._camera_debug_raw_points = clean_visible_world.squeeze(0).detach().cpu()
        self._camera_debug_points = noisy_randomized_world.squeeze(0).detach().cpu()
        self._camera_debug_camera_pos = camera_pos.squeeze(0).detach().cpu()
        self._camera_debug_show_randomized = apply_randomization

        # Compute PCA corners in world frame (use ACTUAL tar_states, not randomized)
        # This shows where the corners SHOULD be based on the true object pose
        tar_pos_actual = tar_states[:, 0:3]  # Don't use _rand_vec here
        tar_rot_actual = tar_states[:, 3:7]
        pca_corners_local = pca_corners.squeeze(0) if pca_corners.dim() == 3 else pca_corners
        corners_world = torch_utils.quat_rotate(
            tar_rot_actual.expand(pca_corners_local.shape[0], -1),
            pca_corners_local
        ) + tar_pos_actual
        self._camera_debug_pca_corners = corners_world.detach().cpu()

        # Debug: print corner bounds vs object position
        print(f"  PCA corners (local) min: {pca_corners_local.min(dim=0)[0].cpu().numpy()}, max: {pca_corners_local.max(dim=0)[0].cpu().numpy()}")
        print(f"  PCA corners (world) min: {corners_world.min(dim=0)[0].cpu().numpy()}, max: {corners_world.max(dim=0)[0].cpu().numpy()}")

        # Compute camera frustum lines for visualization
        forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0)
        up_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0)
        cam_forward = quat_rotate(camera_rot, forward_local).squeeze(0)
        cam_up = quat_rotate(camera_rot, up_local).squeeze(0)
        cam_forward = F.normalize(cam_forward, dim=-1)
        cam_up = F.normalize(cam_up, dim=-1)
        cam_right = F.normalize(torch.cross(cam_forward, cam_up, dim=-1), dim=-1)

        # Store frustum data for line drawing
        frustum_length = 1.5  # meters
        h_fov = math.radians(45)  # half of 90 deg horizontal FOV
        v_fov = math.radians(30)  # half of 60 deg vertical FOV
        cam_pos_np = camera_pos.squeeze(0).detach().cpu()

        # Compute 4 frustum corner directions
        frustum_corners = []
        for h_sign in [-1, 1]:
            for v_sign in [-1, 1]:
                dir_vec = (cam_forward +
                          h_sign * math.tan(h_fov) * cam_right +
                          v_sign * math.tan(v_fov) * cam_up)
                dir_vec = F.normalize(dir_vec, dim=-1)
                corner = cam_pos_np + frustum_length * dir_vec.detach().cpu()
                frustum_corners.append(corner)

        # Frustum lines: from camera to each corner, and connecting corners
        self._camera_debug_frustum_lines = {
            'camera_pos': cam_pos_np.numpy(),
            'corners': [c.numpy() for c in frustum_corners],
            'forward_end': (cam_pos_np + frustum_length * cam_forward.detach().cpu()).numpy()
        }

        self._camera_debug_env = int(env_id)

        # Print debug info
        print(f"\n[DEBUG] Frame {frame_idx}, Env {env_id}:")
        print(f"  Camera pos: {camera_pos.squeeze(0).cpu().numpy()}")
        print(f"  Object pos: {tar_pos_actual.squeeze(0).cpu().numpy()}")
        print(f"  Num visible points: {clean_visible_world.shape[1]}")
        print(f"  Clean points range (BLUE - true geometry): min={clean_visible_world.min(dim=1)[0].cpu().numpy()}, max={clean_visible_world.max(dim=1)[0].cpu().numpy()}")
        print(f"  Noisy points range (GREEN - with DR): min={noisy_randomized_world.min(dim=1)[0].cpu().numpy()}, max={noisy_randomized_world.max(dim=1)[0].cpu().numpy()}")
        # Compute randomization effect
        point_shift = (noisy_randomized_world - clean_visible_world).norm(dim=-1).mean()
        print(f"  Avg point shift (clean->noisy): {point_shift.item():.4f} m")
        print(f"  Domain randomization params: noise_std={self.point_noise_std}, dropout={self.point_dropout_prob}, "
              f"depth_noise={self.point_depth_noise_scale}, occlusion={self.point_occlusion_prob}")

        self._update_debug_viz()

    def _update_debug_viz(self):
        super()._update_debug_viz()
        if self.viewer is None:
            return
        env_id = min(self._camera_debug_env, len(self.envs) - 1)
        env_ptr = self.envs[env_id]

        # Draw GREEN spheres: Final points (randomized or visible)
        if self._camera_debug_points is not None and self._camera_debug_geom is not None:
            for point in self._camera_debug_points:
                pose = gymapi.Transform(
                    p=gymapi.Vec3(float(point[0]), float(point[1]), float(point[2])),
                    r=gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                )
                gymutil.draw_lines(self._camera_debug_geom, self.gym, self.viewer, env_ptr, pose)

        # Draw BLUE spheres: Raw visible points (before randomization)
        if self._camera_debug_raw_points is not None and self._camera_debug_geom_raw is not None:
            for point in self._camera_debug_raw_points:
                pose = gymapi.Transform(
                    p=gymapi.Vec3(float(point[0]), float(point[1]), float(point[2])),
                    r=gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                )
                gymutil.draw_lines(self._camera_debug_geom_raw, self.gym, self.viewer, env_ptr, pose)

        # Draw RED sphere: Camera position
        if self._camera_debug_camera_pos is not None and self._camera_debug_geom_cam is not None:
            cam_pos = self._camera_debug_camera_pos
            pose = gymapi.Transform(
                p=gymapi.Vec3(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])),
                r=gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            )
            gymutil.draw_lines(self._camera_debug_geom_cam, self.gym, self.viewer, env_ptr, pose)

        # Draw YELLOW spheres: PCA bounding box corners
        if self._camera_debug_pca_corners is not None and self._camera_debug_geom_corner is not None:
            for corner in self._camera_debug_pca_corners:
                pose = gymapi.Transform(
                    p=gymapi.Vec3(float(corner[0]), float(corner[1]), float(corner[2])),
                    r=gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                )
                gymutil.draw_lines(self._camera_debug_geom_corner, self.gym, self.viewer, env_ptr, pose)

        # Draw camera frustum lines
        if self._camera_debug_frustum_lines is not None:
            frustum = self._camera_debug_frustum_lines
            cam_pos = frustum['camera_pos']
            corners = frustum['corners']
            forward_end = frustum['forward_end']

            # Helper function to add a line
            def add_line(p1, p2, color):
                vertices = np.array([p1, p2], dtype=np.float32)
                colors = np.array([color, color], dtype=np.float32)
                self.gym.add_lines(self.viewer, env_ptr, 1, vertices, colors)

            # Draw lines from camera to each frustum corner (white)
            white = [1.0, 1.0, 1.0]
            for corner in corners:
                add_line(cam_pos, corner, white)

            # Draw frustum rectangle at the far end
            # corners order: [(-h,-v), (-h,+v), (+h,-v), (+h,+v)]
            add_line(corners[0], corners[1], white)  # left edge
            add_line(corners[2], corners[3], white)  # right edge
            add_line(corners[0], corners[2], white)  # bottom edge
            add_line(corners[1], corners[3], white)  # top edge

            # Draw forward direction (cyan)
            cyan = [0.0, 1.0, 1.0]
            add_line(cam_pos, forward_end, cyan)

        # Draw ORANGE spheres: object goal (task_mode=object_obs playback)
        goal_pos_buf = getattr(self, "_goal_debug_pos_buf", None)
        if (getattr(self, "_goal_debug_geom", None) is not None and goal_pos_buf is not None
                and getattr(self, "task_mode", None) == "object_obs"):
            max_envs = min(self.num_envs, self._goal_debug_max_envs)
            enabled_ids = self._goal_debug_enabled_buf[:max_envs].nonzero(as_tuple=False).flatten().tolist()
            for goal_env_id in enabled_ids:
                goal_pos = goal_pos_buf[goal_env_id].detach().cpu()
                pose = gymapi.Transform(
                    p=gymapi.Vec3(float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2])),
                    r=gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                )
                gymutil.draw_lines(self._goal_debug_geom, self.gym, self.viewer, self.envs[goal_env_id], pose)

def _rand_vec(vec, scale=0.1):
    return vec + (2 * torch.rand_like(vec) - 1) * scale

def _sample_box_surface_points(local_min, local_max, num_points, deterministic=False, grid_resolution=None):
    """Sample points on the surfaces of a bounding box.

    When ``deterministic`` is True, a fixed u-v grid is used on every face so that
    the same candidate directions are produced every frame (closer to a depth sensor
    with fixed pixels). Otherwise random surface sampling is used.
    """
    device = local_min.device
    batch = local_min.shape[0]
    spans = torch.clamp(local_max - local_min, min=1e-5)

    if deterministic:
        # Determine grid resolution so that we always have >= num_points candidates.
        if grid_resolution is None or grid_resolution <= 0:
            per_face = max(1, math.ceil(num_points / 6))
            grid_resolution = max(2, int(math.ceil(math.sqrt(per_face))))
        else:
            grid_resolution = max(2, int(grid_resolution))
        uv_lin = torch.linspace(0.0, 1.0, steps=grid_resolution, device=device)
        u, v = torch.meshgrid(uv_lin, uv_lin, indexing='ij')
        uv = torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1)  # (grid^2, 2)
        uv = uv.unsqueeze(0).expand(batch, -1, -1)
        total_per_face = grid_resolution * grid_resolution

        face_defs = [(-1, 0), (1, 0), (-1, 1), (1, 1), (-1, 2), (1, 2)]
        faces_pts = []
        faces_normals = []
        for sign, axis in face_defs:
            coords = [None, None, None]
            other_axes = [ax for ax in range(3) if ax != axis]
            u_axis, v_axis = other_axes

            u_span = spans[:, u_axis:u_axis+1].unsqueeze(1)
            v_span = spans[:, v_axis:v_axis+1].unsqueeze(1)
            u_base = local_min[:, u_axis:u_axis+1].unsqueeze(1)
            v_base = local_min[:, v_axis:v_axis+1].unsqueeze(1)
            coords[u_axis] = u_base + uv[:, :, 0:1] * u_span
            coords[v_axis] = v_base + uv[:, :, 1:2] * v_span

            fixed_val = local_min[:, axis:axis+1] if sign < 0 else local_max[:, axis:axis+1]
            coords[axis] = fixed_val.unsqueeze(1).expand(-1, total_per_face, -1)

            faces_pts.append(torch.cat(coords, dim=-1))
            normal = torch.zeros(batch, total_per_face, 3, device=device)
            normal[:, :, axis] = -1.0 if sign < 0 else 1.0
            faces_normals.append(normal)

        points = torch.cat(faces_pts, dim=1)
        normals = torch.cat(faces_normals, dim=1)
        total_candidates = points.shape[1]
        if total_candidates < num_points:
            repeat = math.ceil(num_points / total_candidates)
            points = points.repeat(1, repeat, 1)
            normals = normals.repeat(1, repeat, 1)
        return points[:, :num_points, :], normals[:, :num_points, :]

    rand_vals = torch.rand(batch, num_points, 3, device=device)
    points = local_min.unsqueeze(1) + rand_vals * spans.unsqueeze(1)

    faces = torch.randint(0, 6, (batch, num_points), device=device)
    flat_points = points.view(-1, 3)
    flat_min = local_min.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 3)
    flat_max = local_max.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 3)
    faces_flat = faces.view(-1)

    flat_normals = torch.zeros_like(flat_points)
    face_axis = [(0, 1, 0), (2, 3, 1), (4, 5, 2)]
    for f_min, f_max, axis in face_axis:
        mask_min = faces_flat == f_min
        if mask_min.any():
            flat_points[mask_min, axis] = flat_min[mask_min, axis]
            flat_normals[mask_min, axis] = -1.0
        mask_max = faces_flat == f_max
        if mask_max.any():
            flat_points[mask_max, axis] = flat_max[mask_max, axis]
            flat_normals[mask_max, axis] = 1.0

    return flat_points.view(batch, num_points, 3), flat_normals.view(batch, num_points, 3)

def _select_visible_points(points_world, camera_pos, cam_forward, cam_right, cam_up,
                           num_points, horiz_fov_deg=90.0, vert_fov_deg=60.0,
                           normals_world=None):
    """Select up to num_points that fall inside the camera frustum and face the camera.

    Args:
        points_world: (batch, num_candidates, 3) points in world frame
        camera_pos: (batch, 3) camera position
        cam_forward, cam_right, cam_up: (batch, 3) camera axes
        num_points: number of points to select
        horiz_fov_deg, vert_fov_deg: camera field of view
        normals_world: (batch, num_candidates, 3) surface normals in world frame (optional)
                       If provided, points facing away from camera are filtered out (back-face culling)

    Returns:
        selected: (batch, num_points, 3) selected visible points
    """
    h_half = math.radians(horiz_fov_deg * 0.5)
    v_half = math.radians(vert_fov_deg * 0.5)

    vec = points_world - camera_pos.unsqueeze(1)  # Direction from camera to point
    dist = torch.norm(vec, dim=-1, keepdim=True).clamp_min(1e-6)
    dirs = vec / dist  # Normalized direction from camera to point

    forward_comp = (dirs * cam_forward.unsqueeze(1)).sum(dim=-1)
    horiz = torch.atan2((dirs * cam_right.unsqueeze(1)).sum(dim=-1), forward_comp.clamp_min(1e-6))
    vert = torch.atan2((dirs * cam_up.unsqueeze(1)).sum(dim=-1), forward_comp.clamp_min(1e-6))

    # Frustum visibility check
    visible_mask = (forward_comp > 0.0) & (horiz.abs() <= h_half) & (vert.abs() <= v_half)

    # Back-face culling: only keep points whose normal faces toward the camera
    # A point is visible if the dot product of its normal and the view direction (point to camera) is positive
    # i.e., normal · (-dirs) > 0, which means normal · dirs < 0
    if normals_world is not None:
        # Dot product of normal with direction from point to camera (-dirs)
        # If normal points toward camera, this should be positive
        normal_dot_view = -(normals_world * dirs).sum(dim=-1)  # (batch, num_candidates)
        # Point is visible only if normal faces the camera (dot product > 0)
        facing_camera = normal_dot_view > 0.0
        visible_mask = visible_mask & facing_camera

    visible_scores = torch.where(visible_mask, forward_comp, torch.full_like(forward_comp, -1e6))
    topk_visible = torch.topk(visible_scores, k=num_points, dim=1)
    topk_backup = torch.topk(forward_comp, k=num_points, dim=1)
    valid_mask = topk_visible.values > -5e5
    chosen_indices = torch.where(valid_mask, topk_visible.indices, topk_backup.indices)

    selected = torch.gather(points_world, 1, chosen_indices.unsqueeze(-1).expand(-1, num_points, 3))
    return selected

def _apply_point_domain_randomization(points_world, camera_pos, root_pos,
                                       noise_std=0.0, dropout_prob=0.0,
                                       outlier_prob=0.0, outlier_scale=0.5,
                                       depth_noise_scale=0.0,
                                       density_min=1.0, density_max=1.0,
                                       cluster_noise_std=0.0,
                                       scale_min=1.0, scale_max=1.0,
                                       translation_noise=0.0,
                                       occlusion_prob=0.0):
    """
    Comprehensive point cloud domain randomization for sim-to-real transfer.

    Args:
        points_world: (B, N, 3) point cloud in world frame
        camera_pos: (B, 3) camera position
        root_pos: (B, 3) humanoid root position (for centering transforms)
        noise_std: base Gaussian noise std (meters)
        dropout_prob: probability of dropping each point
        outlier_prob: probability of replacing a point with random outlier
        outlier_scale: max distance for outlier points from camera
        depth_noise_scale: depth-dependent noise multiplier
        density_min/max: random subsampling range (0.5 = keep 50% of points)
        cluster_noise_std: points drift toward their neighbors
        scale_min/max: random scale factor for entire point cloud
        translation_noise: random translation offset (calibration error)
        occlusion_prob: probability of occluding a random region

    Returns:
        randomized: (B, N, 3) randomized point cloud
        dropout_mask: (B, N, 1) boolean mask of dropped points
    """
    batch, num_points = points_world.shape[:2]
    device = points_world.device
    dropout_mask = None
    randomized = points_world.clone()

    # 1. Random scale (simulate object size uncertainty)
    if scale_min < scale_max:
        scale = torch.empty(batch, 1, 1, device=device).uniform_(scale_min, scale_max)
        center = randomized.mean(dim=1, keepdim=True)
        randomized = (randomized - center) * scale + center

    # 2. Random translation offset (calibration error)
    if translation_noise > 0.0:
        offset = torch.randn(batch, 1, 3, device=device) * translation_noise
        randomized = randomized + offset

    # 3. Depth-dependent noise (farther points have more noise)
    if depth_noise_scale > 0.0:
        depths = (randomized - camera_pos.unsqueeze(1)).norm(dim=-1, keepdim=True)
        depth_noise = torch.randn_like(randomized) * depths * depth_noise_scale
        randomized = randomized + depth_noise

    # 4. Base Gaussian noise
    if noise_std > 0.0:
        randomized = randomized + torch.randn_like(randomized) * noise_std

    # 5. Clustering noise (points drift toward neighbors)
    if cluster_noise_std > 0.0:
        # Shift each point toward a random neighbor
        neighbor_idx = torch.randint(0, num_points, (batch, num_points), device=device)
        neighbors = torch.gather(randomized, 1, neighbor_idx.unsqueeze(-1).expand(-1, -1, 3))
        drift = (neighbors - randomized) * torch.randn(batch, num_points, 1, device=device) * cluster_noise_std
        randomized = randomized + drift

    # 6. Outlier injection (random spikes simulating sensor noise)
    if outlier_prob > 0.0:
        outlier_mask = torch.rand(batch, num_points, device=device) < outlier_prob
        if outlier_mask.any():
            # Random points around camera
            random_dir = F.normalize(torch.randn(batch, num_points, 3, device=device), dim=-1)
            random_dist = torch.rand(batch, num_points, 1, device=device) * outlier_scale
            outliers = camera_pos.unsqueeze(1) + random_dir * random_dist
            randomized = torch.where(outlier_mask.unsqueeze(-1), outliers, randomized)

    # 7. Random occlusion (drop points in a random region)
    if occlusion_prob > 0.0:
        # For each batch, possibly occlude a spherical region
        do_occlude = torch.rand(batch, device=device) < occlusion_prob
        if do_occlude.any():
            # Pick random occlusion center from existing points
            center_idx = torch.randint(0, num_points, (batch,), device=device)
            occlusion_centers = torch.gather(randomized, 1,
                center_idx.view(batch, 1, 1).expand(-1, -1, 3)).squeeze(1)  # (B, 3)
            occlusion_radius = torch.rand(batch, 1, device=device) * 0.3 + 0.1  # 0.1-0.4m radius

            dist_to_center = (randomized - occlusion_centers.unsqueeze(1)).norm(dim=-1)  # (B, N)
            region_mask = (dist_to_center < occlusion_radius) & do_occlude.unsqueeze(1)

            if region_mask.any():
                cam_expand = camera_pos.unsqueeze(1).expand_as(randomized)
                randomized = torch.where(region_mask.unsqueeze(-1), cam_expand, randomized)
                if dropout_mask is None:
                    dropout_mask = region_mask.unsqueeze(-1)
                else:
                    dropout_mask = dropout_mask | region_mask.unsqueeze(-1)

    # 8. Density variation (random subsampling by zeroing out points)
    if density_min < 1.0:
        keep_ratio = torch.empty(batch, 1, device=device).uniform_(density_min, density_max)
        density_mask = torch.rand(batch, num_points, device=device) > keep_ratio
        if density_mask.any():
            cam_expand = camera_pos.unsqueeze(1).expand_as(randomized)
            randomized = torch.where(density_mask.unsqueeze(-1), cam_expand, randomized)
            if dropout_mask is None:
                dropout_mask = density_mask.unsqueeze(-1)
            else:
                dropout_mask = dropout_mask | density_mask.unsqueeze(-1)

    # 9. Random dropout (independent point dropout)
    if dropout_prob > 0.0:
        point_dropout = torch.rand(batch, num_points, device=device) < dropout_prob
        if point_dropout.any():
            cam_expand = camera_pos.unsqueeze(1).expand_as(randomized)
            randomized = torch.where(point_dropout.unsqueeze(-1), cam_expand, randomized)
            if dropout_mask is None:
                dropout_mask = point_dropout.unsqueeze(-1)
            else:
                dropout_mask = dropout_mask | point_dropout.unsqueeze(-1)

    return randomized, dropout_mask

def compute_obj_observations_pca_corners(root_states, tar_states, pca_corners, ref_obs, camera_pos, camera_rot,
                                         point_noise_std=0.0, point_dropout_prob=0.0,
                                         point_outlier_prob=0.0, point_outlier_scale=0.5,
                                         point_depth_noise_scale=0.0,
                                         point_density_min=1.0, point_density_max=1.0,
                                         point_cluster_noise_std=0.0,
                                         point_scale_min=1.0, point_scale_max=1.0,
                                         point_translation_noise=0.0,
                                         point_occlusion_prob=0.0,
                                         camera_rot_noise=0.0, camera_pos_noise=0.0,
                                         return_randomized_points=False,
                                         disable_geometry_noise=False,
                                         fixed_surface_sampling=False,
                                         surface_grid_resolution=None,
                                         pca_axes=None):
    """
    Deployment-friendly observation using a sampled visible point cloud.
    Sample 64 surface points from the PCA box that fall inside the humanoid head camera frustum.

    Args:
        root_states: (B, 7) humanoid root [pos(3), rot(4)]
        tar_states: (B, 13) object [pos(3), rot(4), vel(3), ang_vel(3)]
        pca_corners: (B, 8, 3) precomputed PCA corners in object local frame
        ref_obs: reference observation from motion data
        camera_pos: (B, 3) rigid-body camera positions (e.g., d435 link)
        camera_rot: (B, 4) rigid-body camera quaternions
        point_*: various domain randomization parameters
        camera_rot_noise: camera rotation noise in radians
        camera_pos_noise: camera position noise in meters
        fixed_surface_sampling: if True use a deterministic UV grid per face, mimicking a fixed-pixel depth camera
        surface_grid_resolution: optional override for the UV grid resolution

    Returns:
        obs: (B, 12 + 64*3) concatenated difference features and visible points in humanoid frame
        points_world: (B, 64, 3) selected points in world frame for IG computation

    Observation breakdown:
    - diff_local_obj_pos (3)
    - diff_local_obj_rot_obs (6)
    - local_obj_pos (3)
    - visible_points_relative (64 × 3)
    """

    # Apply geometry noise for training robustness (disable for clean visualization)
    if disable_geometry_noise:
        root_pos = root_states[:, 0:3].clone()
        root_rot = root_states[:, 3:7].clone()
        tar_pos = tar_states[:, 0:3].clone()
        tar_rot = tar_states[:, 3:7].clone()
    else:
        root_pos = _rand_vec(root_states[:, 0:3], 0.0)
        root_rot = _rand_vec(root_states[:, 3:7], 0.0)
        tar_pos = _rand_vec(tar_states[:, 0:3], 0.0)
        tar_rot = _rand_vec(tar_states[:, 3:7], 0.0)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

    _ref_obj_pos = ref_obs[:,71:74]
    diff_global_obj_pos = _ref_obj_pos - tar_pos
    diff_local_obj_pos_flat = torch_utils.quat_rotate(heading_rot, diff_global_obj_pos)

    ref_obj_rot = ref_obs[:,74:78]
    diff_global_obj_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_obj_rot), tar_rot)
    diff_local_obj_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot)
    diff_local_obj_rot_obs = torch_utils.quat_to_tan_norm(diff_local_obj_rot_flat)

    # Get humanoid heading rotation (yaw-only)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    local_obj_pos = tar_pos - root_pos
    local_obj_pos[..., -1] = tar_pos[..., -1]
    local_obj_pos = quat_rotate(heading_rot, local_obj_pos)

    # Handle pca_corners shape: ensure (B, 8, 3)
    # NOTE: We do NOT add noise to pca_corners - these are fixed geometric properties
    # of the object. Perception noise is already modeled via tar_pos/tar_rot noise.
    if pca_corners.dim() == 2:
        pca_corners = pca_corners.unsqueeze(0).expand(root_pos.shape[0], -1, -1)

    corner_center = pca_corners.mean(dim=1, keepdim=True)
    centered_corners = pca_corners - corner_center
    cov = torch.matmul(centered_corners.transpose(1, 2), centered_corners) / float(centered_corners.shape[1])
    if pca_axes is None:
        pca_axes = box_pca_axes(pca_corners, cov)
    else:
        pca_axes = pca_axes.to(device=cov.device, dtype=cov.dtype)
    aligned_corners = torch.matmul(centered_corners, pca_axes)
    local_min = aligned_corners.min(dim=1).values
    local_max = aligned_corners.max(dim=1).values
    num_visible = 64
    num_candidates = max(num_visible * 8, num_visible)
    aligned_candidates, aligned_normals = _sample_box_surface_points(
        local_min,
        local_max,
        num_candidates,
        deterministic=fixed_surface_sampling,
        grid_resolution=surface_grid_resolution,
    )
    local_candidates = torch.matmul(aligned_candidates, pca_axes.transpose(1, 2)) + corner_center
    local_normals = torch.matmul(aligned_normals, pca_axes.transpose(1, 2))

    tar_rot_expand = tar_rot.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, 4)
    points_world = torch_utils.quat_rotate(
        tar_rot_expand,
        local_candidates.reshape(-1, 3)
    ).reshape(root_pos.shape[0], num_candidates, 3) + tar_pos.unsqueeze(1)

    # Transform normals to world frame (rotation only, no translation)
    normals_world = torch_utils.quat_rotate(
        tar_rot_expand,
        local_normals.reshape(-1, 3)
    ).reshape(root_pos.shape[0], num_candidates, 3)

    # Apply camera pose noise (simulate calibration error)
    noisy_camera_pos = camera_pos
    noisy_camera_rot = camera_rot
    if camera_pos_noise > 0.0:
        noisy_camera_pos = camera_pos + torch.randn_like(camera_pos) * camera_pos_noise
    if camera_rot_noise > 0.0:
        # Add small rotation noise via axis-angle
        axis = F.normalize(torch.randn(camera_rot.shape[0], 3, device=camera_rot.device), dim=-1)
        angle = torch.randn(camera_rot.shape[0], 1, device=camera_rot.device) * camera_rot_noise
        half_angle = angle / 2
        noise_quat = torch.cat([axis * torch.sin(half_angle), torch.cos(half_angle)], dim=-1)
        noisy_camera_rot = torch_utils.quat_mul(camera_rot, noise_quat)

    forward_local = torch.tensor([1.0, 0.0, 0.0], device=camera_pos.device, dtype=camera_pos.dtype).unsqueeze(0)
    up_local = torch.tensor([0.0, 0.0, 1.0], device=camera_pos.device, dtype=camera_pos.dtype).unsqueeze(0)
    forward_local = forward_local.expand(noisy_camera_rot.shape[0], -1)
    up_local = up_local.expand(noisy_camera_rot.shape[0], -1)
    cam_forward = quat_rotate(noisy_camera_rot, forward_local)
    cam_forward = F.normalize(cam_forward, dim=-1)
    cam_up = quat_rotate(noisy_camera_rot, up_local)
    cam_up = F.normalize(cam_up, dim=-1)
    cam_right = F.normalize(torch.cross(cam_forward, cam_up, dim=-1), dim=-1)
    cam_up = F.normalize(torch.cross(cam_right, cam_forward, dim=-1), dim=-1)

    # Select visible points with back-face culling (points whose normal faces away from camera are filtered out)
    visible_world = _select_visible_points(
        points_world, noisy_camera_pos, cam_forward, cam_right, cam_up, num_visible,
        normals_world=normals_world
    )

    randomized_world, dropout_mask = _apply_point_domain_randomization(
        visible_world, noisy_camera_pos, root_pos,
        noise_std=point_noise_std,
        dropout_prob=point_dropout_prob,
        outlier_prob=point_outlier_prob,
        outlier_scale=point_outlier_scale,
        depth_noise_scale=point_depth_noise_scale,
        density_min=point_density_min,
        density_max=point_density_max,
        cluster_noise_std=point_cluster_noise_std,
        scale_min=point_scale_min,
        scale_max=point_scale_max,
        translation_noise=point_translation_noise,
        occlusion_prob=point_occlusion_prob
    )

    # Transform to camera/head frame (fully consistent 3D coordinates)
    cam_rot_inv = torch_utils.quat_inverse(noisy_camera_rot)
    cam_rot_inv_expand = cam_rot_inv.unsqueeze(1).expand(-1, num_visible, -1)
    relative_points = quat_rotate(
        cam_rot_inv_expand.reshape(-1, 4),
        (randomized_world - noisy_camera_pos.unsqueeze(1)).reshape(-1, 3)
    ).reshape(root_pos.shape[0], num_visible, 3)

    if dropout_mask is not None:
        keep_mask = (~dropout_mask).float()
        relative_points = relative_points * keep_mask

    obs = torch.cat(
        [
            diff_local_obj_pos_flat,
            diff_local_obj_rot_obs,
            local_obj_pos,
            relative_points.reshape(root_pos.shape[0], -1),
        ],
        dim=-1,
    )

    if return_randomized_points:
        return obs, visible_world, randomized_world
    return obs, visible_world


def compute_obj_observations(root_states, tar_states, object_points, ref_obs):
    """
    Original object observation using surface point cloud.
    This version uses heavy surface vectors and is less deployment-friendly.
    """
    root_pos = _rand_vec(root_states[:, 0:3], 0.0)
    root_rot = _rand_vec(root_states[:, 3:7], 0.)

    tar_pos = _rand_vec(tar_states[:, 0:3], 0.0)
    tar_rot = _rand_vec(tar_states[:, 3:7], 0.)
    tar_vel = _rand_vec(tar_states[:, 7:10], 0.)
    tar_ang_vel = tar_states[:, 10:13]

    obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    object_points_extend = object_points.view(-1, 3)
    obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    _ref_obj_pos = ref_obs[:,71:74]
    diff_global_obj_pos = _ref_obj_pos - tar_pos
    diff_local_obj_pos_flat = torch_utils.quat_rotate(heading_rot, diff_global_obj_pos)

    local_ref_obj_pos = _ref_obj_pos - root_pos  # preserves the body position
    local_ref_obj_pos = torch_utils.quat_rotate(heading_rot, local_ref_obj_pos)

    ref_obj_rot = ref_obs[:,74:78]
    diff_global_obj_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_obj_rot), tar_rot)
    diff_local_obj_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot)  # Need to be change of basis
    diff_local_obj_rot_obs = torch_utils.quat_to_tan_norm(diff_local_obj_rot_flat)

    local_ref_obj_rot = torch_utils.quat_mul(heading_rot, ref_obj_rot)
    local_ref_obj_rot = torch_utils.quat_to_tan_norm(local_ref_obj_rot)

    ref_obj_vel = ref_obs[:,78:81]
    diff_global_vel = ref_obj_vel - tar_vel
    diff_local_vel = torch_utils.quat_rotate(heading_rot, diff_global_vel)

    ref_obj_ang_vel = ref_obs[:,81:84]
    diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
    diff_local_ang_vel = torch_utils.quat_rotate(heading_rot, diff_global_ang_vel)

    # local_tar_ang_vel = local_tar_ang_vel * 0
    # # local_tar_vel = local_tar_vel * 0
    # # diff_local_vel = diff_local_vel * 0
    # diff_local_ang_vel = diff_local_ang_vel * 0
    obs = torch.cat([local_tar_vel, local_tar_ang_vel, diff_local_obj_pos_flat, diff_local_obj_rot_obs, diff_local_vel, diff_local_ang_vel], dim=-1)
    return obs, obj_points
