import numpy as np
import torch
import torch.nn.functional as F
import mujoco as mj
import os
import math
from utils import torch_utils_mujoco
from utils.box_pca import box_pca_axes
from collections import OrderedDict

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

def build_human_only_mappings(model,
                              pelvis_name="pelvis",
                              human_body_names=None,
                              contact_body_names=None):
    """
    Returns:
      body_ids           : np.int32 [Nb] (root first; expect Nb=39)
      contact_body_ids   : np.int32 [Nb] or subset (aligned to body_ids if you pass the same list)
      joint_ids          : np.int32 [Nu] (from actuators)
      dof_qpos_ids       : np.int32 [Nu] (addresses into qpos)
      dof_qvel_ids       : np.int32 [Nu] (addresses into qvel)
      actuator_ids       : np.int32 [Nu] (0..nu-1)
    """
    # 1) Bodies (prefer user-supplied list to preserve training order)
    human_body_names = ['pelvis', 'imu_in_pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'pelvis_contour_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 
                        'waist_yaw_link', 'waist_roll_link', 'torso_link', 'd435_link', 'head_link', 'imu_in_torso', 
                        'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'left_rubber_hand', 
                        'logo_link', 'mid360_link', 
                        'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 'right_rubber_hand']
    body_ids = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n)
                         for n in human_body_names], dtype=np.int32)
    if np.any(body_ids < 0):
        bad = [n for n, i in zip(human_body_names, body_ids) if i < 0]
        raise ValueError(f"Unknown human body names: {bad}")

    # 2) Contact bodies: default to "all human bodies" so you get 39 booleans
    if contact_body_names is None:
        contact_body_names = human_body_names
    contact_body_ids = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n)
                                 for n in contact_body_names], dtype=np.int32)

    # 3) Joints/DOFs from actuators (object has none → human-only)
    joint_ids = []
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]  # joint-based motor in your XML
        if jid >= 0:
            joint_ids.append(jid)
    joint_ids = np.array(joint_ids, dtype=np.int32)

    # Addresses
    dof_qpos_ids = model.jnt_qposadr[joint_ids].astype(np.int32)
    dof_qvel_ids = model.jnt_dofadr[joint_ids].astype(np.int32)

    actuator_ids = np.arange(model.nu, dtype=np.int32)

    return body_ids, contact_body_ids, joint_ids, dof_qpos_ids, dof_qvel_ids, actuator_ids

def wxyz_to_xyzw(q):  # q: (..., 4)
    # (w,x,y,z) -> (x,y,z,w)
    if isinstance(q, torch.Tensor):
        w = q[..., 0:1]
        xyz = q[..., 1:4]
        return torch.cat([xyz, w], dim=-1)
    else:
        # numpy path
        w = q[..., 0:1]
        xyz = q[..., 1:4]
        return np.concatenate([xyz, w], axis=-1)

def xyzw_to_wxyz(q):
    # (x,y,z,w) -> (w,x,y,z)
    if isinstance(q, torch.Tensor):
        xyz = q[..., 0:3]
        w = q[..., 3:4]
        return torch.cat([w, xyz], dim=-1)
    else:
        xyz = q[..., 0:3]
        w = q[..., 3:4]
        return np.concatenate([w, xyz], axis=-1)


def _body_world_velocities(model, data, body_ids):
    """
    Compute world-frame linear and angular velocities for each body origin
    using Jacobians:  v = J * qvel
    Returns:
        lin (B, Nb, 3), ang (B, Nb, 3) as torch tensors with B=1
    """
    nv = model.nv
    jacp = np.zeros((3, nv), dtype=np.float64)
    jacr = np.zeros((3, nv), dtype=np.float64)

    lin = []
    ang = []
    for bid in body_ids:
        jacp.fill(0.0); jacr.fill(0.0)
        # Jacobian at the body origin (not mj_jacBodyCom).
        mj.mj_jacBody(model, data, jacp, jacr, bid)
        v_lin = jacp @ data.qvel
        v_ang = jacr @ data.qvel
        lin.append(v_lin)
        ang.append(v_ang)

    lin = torch.from_numpy(np.asarray(lin, dtype=np.float64)).unsqueeze(0).to(torch.float32)  # (1, Nb, 3)
    ang = torch.from_numpy(np.asarray(ang, dtype=np.float64)).unsqueeze(0).to(torch.float32)  # (1, Nb, 3)
    return lin, ang


def _single_body_world_velocities(model, data, body_id):
    """World-frame linear & angular body-origin velocities via Jacobians."""
    nv = model.nv
    jacp = np.zeros((3, nv), dtype=np.float64)
    jacr = np.zeros((3, nv), dtype=np.float64)
    mj.mj_jacBody(model, data, jacp, jacr, int(body_id))
    v_lin = jacp @ data.qvel
    v_ang = jacr @ data.qvel
    v_lin = torch.from_numpy(v_lin).to(torch.float32).unsqueeze(0)  # (1,3)
    v_ang = torch.from_numpy(v_ang).to(torch.float32).unsqueeze(0)  # (1,3)
    return v_lin, v_ang

def _gather_contact_forces(model, data, body_ids):
    """
    World-frame external forces; MuJoCo spatial wrenches use [torque, force].
    Returns (1, Nb, 3) torch tensor.
    """
    # These arrays are not updated by mj_step unless a sensor requires them.
    mj.mj_rnePostConstraint(model, data)
    f_world = data.cfrc_ext[body_ids, 3:6].copy()
    return torch.from_numpy(f_world).unsqueeze(0).to(torch.float32)


def _gather_body_poses(model, data, body_ids):
    """
    Returns:
        body_pos: (1, Nb, 3)
        body_rot: (1, Nb, 4)
    """
    pos = data.xpos[body_ids, :].copy()      # (Nb, 3)
    quat_wxyz = data.xquat[body_ids, :].copy()  # (Nb, 4), MuJoCo = (w,x,y,z)
    quat_xyzw = wxyz_to_xyzw(quat_wxyz)         # convert for Isaac-style utils

    body_pos = torch.from_numpy(pos).unsqueeze(0).to(torch.float32)     # (1, Nb, 3)
    body_rot = torch.from_numpy(quat_xyzw).unsqueeze(0).to(torch.float32)  # (1, Nb, 4) in xyzw
    return body_pos, body_rot



def _gather_dof_slices(data, dof_qpos_ids, dof_qvel_ids):
    """
    Map your policy DOFs to qpos/qvel slices.
    - dof_qpos_ids: indices into qpos for your joints (list[int] or np.array)
    - dof_qvel_ids: indices into qvel for your joints
    Returns:
        dof_pos, dof_vel: all shaped (1, Ndof)
    """
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    dof_pos = torch.from_numpy(qpos[dof_qpos_ids].astype(np.float32)).unsqueeze(0)
    dof_vel = torch.from_numpy(qvel[dof_qvel_ids].astype(np.float32)).unsqueeze(0)

    return dof_pos, dof_vel

def compute_sdf(points1, points2):
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing='ij')
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat

def _rand_vec(vec, scale=0.0):
    if scale == 0:
        return vec
    return vec + (2 * torch.rand_like(vec) - 1) * scale

def _sample_box_surface_points(local_min, local_max, num_points, deterministic=False, grid_resolution=None):
    """Sample points on the surfaces of an oriented bounding box.

    When deterministic=True, tile a fixed UV grid on each face to mimic a depth camera with stable pixels.
    """
    device = local_min.device
    batch = local_min.shape[0]
    spans = torch.clamp(local_max - local_min, min=1e-5)

    if deterministic:
        if grid_resolution is None or grid_resolution <= 0:
            per_face = max(1, math.ceil(num_points / 6))
            grid_resolution = max(2, int(math.ceil(math.sqrt(per_face))))
        else:
            grid_resolution = max(2, int(grid_resolution))
        uv_lin = torch.linspace(0.0, 1.0, steps=grid_resolution, device=device)
        u, v = torch.meshgrid(uv_lin, uv_lin, indexing='ij')
        uv = torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1).unsqueeze(0).expand(batch, -1, -1)
        total_per_face = grid_resolution * grid_resolution

        face_defs = [(-1, 0), (1, 0), (-1, 1), (1, 1), (-1, 2), (1, 2)]
        face_points = []
        face_normals = []
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

            face_points.append(torch.cat(coords, dim=-1))
            normal = torch.zeros(batch, total_per_face, 3, device=device)
            normal[:, :, axis] = -1.0 if sign < 0 else 1.0
            face_normals.append(normal)

        points = torch.cat(face_points, dim=1)
        normals = torch.cat(face_normals, dim=1)
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
    flat_normals = torch.zeros_like(flat_points)
    flat_min = local_min.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 3)
    flat_max = local_max.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 3)
    faces_flat = faces.view(-1)

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
        neighbor_idx = torch.randint(0, num_points, (batch, num_points), device=device)
        neighbors = torch.gather(randomized, 1, neighbor_idx.unsqueeze(-1).expand(-1, -1, 3))
        drift = (neighbors - randomized) * torch.randn(batch, num_points, 1, device=device) * cluster_noise_std
        randomized = randomized + drift

    # 6. Outlier injection (random spikes simulating sensor noise)
    if outlier_prob > 0.0:
        outlier_mask = torch.rand(batch, num_points, device=device) < outlier_prob
        if outlier_mask.any():
            random_dir = F.normalize(torch.randn(batch, num_points, 3, device=device), dim=-1)
            random_dist = torch.rand(batch, num_points, 1, device=device) * outlier_scale
            outliers = camera_pos.unsqueeze(1) + random_dir * random_dist
            randomized = torch.where(outlier_mask.unsqueeze(-1), outliers, randomized)

    # 7. Random occlusion (drop points in a random region)
    if occlusion_prob > 0.0:
        do_occlude = torch.rand(batch, device=device) < occlusion_prob
        if do_occlude.any():
            center_idx = torch.randint(0, num_points, (batch,), device=device)
            occlusion_centers = torch.gather(randomized, 1,
                center_idx.view(batch, 1, 1).expand(-1, -1, 3)).squeeze(1)
            occlusion_radius = torch.rand(batch, 1, device=device) * 0.3 + 0.1

            dist_to_center = (randomized - occlusion_centers.unsqueeze(1)).norm(dim=-1)
            region_mask = (dist_to_center < occlusion_radius) & do_occlude.unsqueeze(1)

            if region_mask.any():
                cam_expand = camera_pos.unsqueeze(1).expand_as(randomized)
                randomized = torch.where(region_mask.unsqueeze(-1), cam_expand, randomized)
                if dropout_mask is None:
                    dropout_mask = region_mask.unsqueeze(-1)
                else:
                    dropout_mask = dropout_mask | region_mask.unsqueeze(-1)

    # 8. Density variation (random subsampling)
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


def _select_visible_points(points_world, camera_pos, cam_forward, cam_right, cam_up,
                           num_points, horiz_fov_deg=90.0, vert_fov_deg=60.0,
                           normals_world=None):
    """Select up to num_points that fall inside the camera frustum and face the camera."""
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


def contact_indicator_for_bodies(model, data, body_ids, device=None):
    """
    Returns a torch tensor of shape (1, len(body_ids)).
    Each entry is 1.0 if that body is in contact (with ground or any other body),
    else 0.0.

    Args:
        model: mujoco.MjModel
        data: mujoco.MjData
        body_ids: list/array of body ids to check
    """
    body_ids = list(body_ids)
    indicator = torch.zeros((1, len(body_ids)), dtype=torch.float32, device=device)

    # Go through all active contacts
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]

        # If either geom’s parent body is in our target set, mark it
        for j, bid in enumerate(body_ids):
            if b1 == bid or b2 == bid:
                indicator[0, j] = 1.0

    return indicator



class MujocoObs:
    def __init__(self, model, object_body_name, max_episode_length, hoi_data, object_points, object_corners=None, actuator_ids=None,
                 local_root_obs=False, root_height_obs=True, history_step=0, device="cpu",
                 obj_rot_keep_prob=1.0, obj_trans_keep_prob=1.0, obj_point_keep_prob=1.0, obj_pos_keep_prob=1.0,
                 human_move_keep_prob=1.0, human_global_keep_prob=None, human_local_keep_prob=None,
                 human_goal_keep_prob=1.0, mask_flip_prob=0.0,
                 point_fixed_grid_sampling=False, point_surface_grid_resolution=None,
                 point_noise_std=0.0, point_dropout_prob=0.0,
                 point_outlier_prob=0.0, point_outlier_scale=0.5,
                 point_depth_noise_scale=0.0,
                 point_density_min=1.0, point_density_max=1.0,
                 point_cluster_noise_std=0.0,
                 point_scale_min=1.0, point_scale_max=1.0,
                 point_translation_noise=0.0,
                 point_occlusion_prob=0.0,
                 camera_rot_noise=0.0, camera_pos_noise=0.0,
                 goal_phase_dim=4, goal_phase_max_len=240.0,
                 obj_goal_decouple="off", obj_goal_z_threshold=0.15,
                 goal_achievement_enabled=False, goal_pos_threshold=0.3):
        self.model = model
        
        # Build human-only mappings (pass your known training order if you have it)
        human_body_names_39 = None  # or a concrete list to lock training order
        contact_body_names  = None  # default = all 39

        (body_ids,
        contact_body_ids,
        joint_ids,
        dof_qpos_ids,
        dof_qvel_ids,
        actuator_ids) = build_human_only_mappings(
            model,
            pelvis_name="pelvis",
            human_body_names=human_body_names_39,
            contact_body_names=contact_body_names
        )
        self.body_ids = np.asarray(body_ids, dtype=np.int32)
        self.contact_body_ids = np.asarray(contact_body_ids, dtype=np.int32)
        self.dof_qpos_ids = np.asarray(dof_qpos_ids, dtype=np.int32)
        self.dof_qvel_ids = np.asarray(dof_qvel_ids, dtype=np.int32)
        # # print(self.dof_qpos_ids, self.dof_qvel_ids, actuator_ids)
        self.actuator_ids = None if actuator_ids is None else np.asarray(actuator_ids, dtype=np.int32)
        self.local_root_obs = local_root_obs
        self.root_height_obs = root_height_obs
        self.device = device
        self.object_body_name = object_body_name
        self.max_episode_length = max_episode_length
        self.hoi_data = hoi_data
        self.object_points = object_points
        self.object_corners = object_corners
        self.history_step = history_step
        if self.history_step > 0:
            self.obs_history_buf = torch.zeros(
                1, # num_envs
                self.history_step,
                92, # self.cfg['env']['numObsProprio'],
                # device=self.device,
                dtype=torch.float,
            )
        self.long_term_t = torch.zeros((1,), dtype=torch.long)
        self.current_goal_ts = None  # Track the current goal timestep for sparse tracking
        self.frozen_ts = None  # Frozen reference time when goal not achieved
        # stateful last_* buffers (1, Ndof)
        self.last_dof_pos = None
        self.last_dof_vel = None
        self.global_goal_dim = 3
        self.goal_phase_dim = int(goal_phase_dim)
        if self.goal_phase_dim not in (3, 4):
            raise ValueError(f"goal_phase_dim must be 3 or 4, got {self.goal_phase_dim}")
        self.goal_phase_max_len = float(goal_phase_max_len)
        self.command_dim = self.goal_phase_dim  # goal_phase: [progress_end, approaching, leaving] (+ time_to_go)
        self.local_goal_dim = 31
        self.task_obs_dim = 204
        self.mask_dim = 242
        self.command_goal = torch.zeros((1, self.command_dim), dtype=torch.float32)
        self.long_term_speed_threshold = 0.2

        # Goal achievement checking for sparse tracking
        self.goal_achievement_enabled = goal_achievement_enabled
        self.goal_pos_threshold = goal_pos_threshold  # Distance threshold in meters
        self.ig_interaction_threshold = 0.15
        self.ig_transition_delta = 0.0001
        self.ig_hand_body_ids = [28, 38]
        self.prev_hand_ig = torch.zeros((1,), dtype=torch.float32)
        self.prev_hand_ig_valid = torch.zeros((1,), dtype=torch.bool)

        self.obj_rot_keep_prob = obj_rot_keep_prob
        self.obj_trans_keep_prob = obj_trans_keep_prob
        self.obj_point_keep_prob = obj_point_keep_prob
        self.obj_pos_keep_prob = obj_pos_keep_prob
        self.human_move_keep_prob = human_move_keep_prob
        self.human_global_keep_prob = human_global_keep_prob if human_global_keep_prob is not None else human_move_keep_prob
        self.human_local_keep_prob = human_local_keep_prob if human_local_keep_prob is not None else human_move_keep_prob
        self.human_goal_keep_prob = human_goal_keep_prob
        self.keep_mask_flip_prob = mask_flip_prob

        self.keep_obj_point_mask = None
        self.keep_obj_trans_mask = None
        self.keep_obj_rot_mask = None
        self.keep_obj_pos_mask = None
        self.keep_human_move_mask = None
        self.keep_global_goal_mask = None
        self.keep_local_goal_mask = None
        self.keep_goal_mask = None
        self.point_fixed_grid_sampling = point_fixed_grid_sampling
        self.point_surface_grid_resolution = point_surface_grid_resolution

        # Domain randomization parameters
        self.point_noise_std = point_noise_std
        self.point_dropout_prob = point_dropout_prob
        self.point_outlier_prob = point_outlier_prob
        self.point_outlier_scale = point_outlier_scale
        self.point_depth_noise_scale = point_depth_noise_scale
        self.point_density_min = point_density_min
        self.point_density_max = point_density_max
        self.point_cluster_noise_std = point_cluster_noise_std
        self.point_scale_min = point_scale_min
        self.point_scale_max = point_scale_max
        self.point_translation_noise = point_translation_noise
        self.point_occlusion_prob = point_occlusion_prob
        self.camera_rot_noise = camera_rot_noise
        self.camera_pos_noise = camera_pos_noise
        self.obj_goal_decouple = str(obj_goal_decouple).lower()
        self.obj_goal_z_threshold = float(obj_goal_z_threshold)

    def configure_task_mode(self, task_mode, obj_obs="points"):
        """Override keep masks based on a named task preset."""
        task_mode = str(task_mode).lower()
        obj_obs = str(obj_obs).lower()
        self.task_mode = task_mode
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
            # Match the native Gym preset: dense tracking exposes every modality.
            pass
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

        self.keep_obj_point_mask = torch.tensor(keep["obj_point"], dtype=torch.bool)
        self.keep_obj_trans_mask = torch.tensor(keep["obj_trans"], dtype=torch.bool)
        self.keep_obj_rot_mask = torch.tensor(keep["obj_rot"], dtype=torch.bool)
        self.keep_obj_pos_mask = torch.tensor(keep["obj_pos"], dtype=torch.bool)
        self.keep_human_move_mask = torch.tensor(keep["human_move"], dtype=torch.bool)
        self.keep_global_goal_mask = torch.tensor(keep["global_goal"], dtype=torch.bool)
        self.keep_local_goal_mask = torch.tensor(keep["local_goal"], dtype=torch.bool)
        self.keep_goal_mask = torch.tensor(keep["goal"], dtype=torch.bool)

    def set_command_goal(self, command_goal):
        if not isinstance(command_goal, torch.Tensor):
            command_goal = torch.as_tensor(command_goal, dtype=torch.float32)
        self.command_goal = command_goal.view(1, -1)

    def _maybe_sample_keep_masks(self):
        if self.keep_obj_point_mask is None:
            self.keep_obj_point_mask = torch.rand(1) < self.obj_point_keep_prob
        if self.keep_obj_trans_mask is None:
            self.keep_obj_trans_mask = torch.rand(1) < self.obj_trans_keep_prob
        if self.keep_obj_rot_mask is None:
            self.keep_obj_rot_mask = torch.rand(1) < self.obj_rot_keep_prob
        if self.keep_obj_pos_mask is None:
            self.keep_obj_pos_mask = torch.rand(1) < self.obj_pos_keep_prob
        if self.keep_human_move_mask is None:
            self.keep_human_move_mask = torch.rand(1) < self.human_move_keep_prob
        if self.keep_global_goal_mask is None:
            self.keep_global_goal_mask = torch.rand(1) < self.human_global_keep_prob
        if self.keep_local_goal_mask is None:
            self.keep_local_goal_mask = torch.rand(1) < self.human_local_keep_prob
        if self.keep_goal_mask is None:
            self.keep_goal_mask = torch.rand(1) < self.human_goal_keep_prob

    def _maybe_flip_keep_masks(self):
        if self.keep_mask_flip_prob <= 0:
            self._maybe_sample_keep_masks()
            return
        self._maybe_sample_keep_masks()
        if torch.rand(1) < self.keep_mask_flip_prob:
            self.keep_obj_point_mask = ~self.keep_obj_point_mask
        if torch.rand(1) < self.keep_mask_flip_prob:
            self.keep_obj_trans_mask = ~self.keep_obj_trans_mask
        if torch.rand(1) < self.keep_mask_flip_prob:
            self.keep_obj_rot_mask = ~self.keep_obj_rot_mask
        if torch.rand(1) < self.keep_mask_flip_prob:
            self.keep_obj_pos_mask = ~self.keep_obj_pos_mask
        if torch.rand(1) < self.keep_mask_flip_prob:
            self.keep_human_move_mask = ~self.keep_human_move_mask
        if torch.rand(1) < self.keep_mask_flip_prob:
            self.keep_global_goal_mask = ~self.keep_global_goal_mask
        if torch.rand(1) < self.keep_mask_flip_prob:
            self.keep_local_goal_mask = ~self.keep_local_goal_mask
        if torch.rand(1) < self.keep_mask_flip_prob:
            self.keep_goal_mask = ~self.keep_goal_mask

    def _finalize_student_obs(
        self,
        obs,
        obs_shape,
        task_obs_size,
        force_global_goal_mask=None,
        force_local_goal_mask=None,
        force_goal_mask=None,
        force_obj_point_mask=None,
        force_obj_trans_mask=None,
        force_obj_rot_mask=None,
        force_obj_pos_mask=None,
    ):
        self._maybe_flip_keep_masks()
        mask_features = []

        def append_mask(mask_tensor, length):
            if length <= 0:
                return
            mask = mask_tensor.float().to(obs.device).unsqueeze(-1)
            mask_features.append(mask.expand(1, length))

        global_goal_dim = min(self.global_goal_dim, obs.shape[-1])
        local_goal_dim = min(self.local_goal_dim, max(obs_shape - global_goal_dim, 0))

        global_goal = torch.empty((1, 0), device=obs.device, dtype=obs.dtype)
        if global_goal_dim > 0:
            global_goal = obs[:, :global_goal_dim]
            if force_global_goal_mask is None:
                append_mask(self.keep_global_goal_mask, global_goal_dim)
            else:
                append_mask(force_global_goal_mask, global_goal_dim)

        local_goal = torch.empty((1, 0), device=obs.device, dtype=obs.dtype)
        if local_goal_dim > 0:
            start = global_goal_dim
            end = start + local_goal_dim
            local_goal = obs[:, start:end]
            if force_local_goal_mask is None:
                append_mask(self.keep_local_goal_mask, local_goal_dim)
            else:
                append_mask(force_local_goal_mask, local_goal_dim)

        task_start = obs_shape
        if task_obs_size > 0:
            task_end = obs_shape + task_obs_size
            trans_end = min(task_start + 3, task_end)
            if trans_end > task_start:
                if force_obj_trans_mask is None:
                    append_mask(self.keep_obj_trans_mask, trans_end - task_start)
                else:
                    append_mask(force_obj_trans_mask, trans_end - task_start)
            rot_start = trans_end
            rot_end = min(rot_start + 6, task_end)
            if rot_end > rot_start:
                if force_obj_rot_mask is None:
                    append_mask(self.keep_obj_rot_mask, rot_end - rot_start)
                else:
                    append_mask(force_obj_rot_mask, rot_end - rot_start)
            pos_start = rot_end
            pos_end = min(pos_start + 3, task_end)
            if pos_end > pos_start:
                if force_obj_pos_mask is None:
                    append_mask(self.keep_obj_pos_mask, pos_end - pos_start)
                else:
                    append_mask(force_obj_pos_mask, pos_end - pos_start)
            point_start = pos_end
            if task_end > point_start:
                if force_obj_point_mask is None:
                    append_mask(self.keep_obj_point_mask, task_end - point_start)
                else:
                    append_mask(force_obj_point_mask, task_end - point_start)

        goal = self.command_goal.to(obs.device, obs.dtype)
        if force_goal_mask is None:
            append_mask(self.keep_goal_mask, goal.shape[-1])
        else:
            append_mask(force_goal_mask, goal.shape[-1])

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
        return combined

    def _append_goal_viz(self, obs_dict, ref_obs):
        """Add goal visualization targets based on enabled masks."""
        obj_goal_pos = ref_obs[:, 71:74].clone()
        human_goal_pos = ref_obs[:, 0:3].clone()
        obj_enabled = bool(self.keep_obj_trans_mask.item()) if self.keep_obj_trans_mask is not None else False
        human_enabled = bool(self.keep_global_goal_mask.item()) if self.keep_global_goal_mask is not None else False

        obs_dict["viz_obj_goal_pos"] = obj_goal_pos
        obs_dict["viz_obj_goal_enabled"] = torch.tensor([obj_enabled], device=obj_goal_pos.device)
        if "viz_obj_goal_decoupled_pos" in obs_dict:
            decouple_enabled = obj_enabled and self.obj_goal_decouple != "off"
            obs_dict["viz_obj_goal_decoupled_enabled"] = torch.tensor([decouple_enabled], device=obj_goal_pos.device)
        obs_dict["viz_human_goal_pos"] = human_goal_pos
        obs_dict["viz_human_goal_enabled"] = torch.tensor([human_enabled], device=human_goal_pos.device)

    def _decouple_obj_trans_goal(self, diff_local_obj_pos_flat):
        """Optionally decouple object translation goal: vertical first, then horizontal."""
        mode = self.obj_goal_decouple
        if mode == "off":
            return diff_local_obj_pos_flat
        if diff_local_obj_pos_flat.shape[-1] < 3:
            return diff_local_obj_pos_flat
        z_thresh = max(self.obj_goal_z_threshold, 1e-6)
        z_err = diff_local_obj_pos_flat[..., 2:3]
        xy = diff_local_obj_pos_flat[..., 0:2]
        if mode == "hard":
            mask = z_err.abs() > z_thresh
            xy = torch.where(mask.expand_as(xy), torch.zeros_like(xy), xy)
        elif mode == "soft":
            scale = (1.0 - (z_err.abs() / z_thresh)).clamp(min=0.0, max=1.0)
            xy = xy * scale
        return torch.cat([xy, z_err], dim=-1)

    def _ensure_last_buffers(self, dof_pos, dof_vel):
        if self.last_dof_pos is None:
            self.last_dof_pos = torch.zeros_like(dof_pos.clone())
        if self.last_dof_vel is None:
            self.last_dof_vel = torch.zeros_like(dof_vel.clone())

    def _get_head_camera_pose(self, data, root_states):
        """Return (position, rotation) tensors for the robot head camera."""
        # Try to get the d435_link body (camera link)
        try:
            camera_bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "d435_link")
            if camera_bid >= 0:
                cam_pos = torch.from_numpy(data.xpos[camera_bid].copy()).unsqueeze(0).float()  # (1, 3)
                cam_quat_wxyz = torch.from_numpy(data.xquat[camera_bid].copy()).float()  # (4,)
                cam_rot = wxyz_to_xyzw(cam_quat_wxyz).unsqueeze(0)  # (1, 4)
                return cam_pos, cam_rot
        except:
            pass

        # Fall back to root states if d435_link not found
        cam_pos = root_states[:, 0:3]
        cam_rot = root_states[:, 3:7]
        return cam_pos, cam_rot

    def _check_goal_achieved(self, data, goal_ts):
        """Check if the current object position is within threshold of the goal position."""
        if not self.goal_achievement_enabled:
            return True  # Always achieved if checking is disabled

        # Get current object position
        obj_bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.object_body_name)
        current_obj_pos = torch.from_numpy(data.xpos[obj_bid].copy()).float()  # (3,)

        # Get goal object position from reference data
        goal_ref = self.hoi_data[goal_ts]
        goal_obj_pos = goal_ref[71:74]  # obj_pos is at indices 71:74

        # Compute distance
        dist = torch.norm(current_obj_pos - goal_obj_pos).item()

        return dist < self.goal_pos_threshold

    def _compute_observations_iter(self, data, curr_t, delta_t=1, actions=None, torques=None, student_obs=False, episode_length=0):
        ts = curr_t
        if student_obs:
            # Check if we need to sample a new goal or keep the current one
            need_new_goal = False
            is_frozen = False  # Whether to freeze ts and long_term_t

            if episode_length <= 1:
                # First step of episode - always sample new goal
                need_new_goal = True
                self.current_goal_ts = None
                self.frozen_ts = None
            elif self.long_term_t.item() <= 0:
                # Horizon exhausted - check if goal was achieved before sampling new one
                if self.goal_achievement_enabled and self.current_goal_ts is not None:
                    goal_achieved = self._check_goal_achieved(data, self.current_goal_ts)
                    if goal_achieved:
                        need_new_goal = True
                        self.frozen_ts = None  # Unfreeze
                        print("Goal achieved at ts", ts)
                    else:
                        # Goal NOT achieved - freeze both ts and long_term_t
                        is_frozen = True
                        if self.frozen_ts is None:
                            # Record the timestep when we started freezing
                            self.frozen_ts = ts
                else:
                    need_new_goal = True

            # Use frozen_ts if we're frozen, otherwise use current ts
            effective_ts = self.frozen_ts if is_frozen and self.frozen_ts is not None else ts

            if need_new_goal:
                # Sample new horizon based on reference velocity at current timestep
                curr_ref_obs = self.hoi_data[None, min(effective_ts, self.max_episode_length-1)].clone()
                ref_vel = curr_ref_obs[:, 78:81]
                ref_speed = torch.norm(ref_vel, dim=-1)
                use_long = ref_speed < self.long_term_speed_threshold
                long_horizon = torch.randint(
                    120, 240, (1,), device=self.long_term_t.device, dtype=self.long_term_t.dtype
                )
                short_horizon = torch.randint(
                    60, 120, (1,), device=self.long_term_t.device, dtype=self.long_term_t.dtype
                )
                self.long_term_t = torch.where(use_long, long_horizon, short_horizon)
                # Update the current goal timestep
                self.current_goal_ts = min(effective_ts + self.long_term_t.item(), self.max_episode_length-1)

            # Use long_term_t as the reference horizon for sparse tracking (global goal)
            if self.current_goal_ts is not None:
                next_ts = self.current_goal_ts
            else:
                next_ts = min(effective_ts + self.long_term_t.item(), self.max_episode_length-1)
                self.current_goal_ts = next_ts

            # Build the command with the same horizon as the target pose.
        else:
            # For teacher, use delta_t directly
            next_ts = min(ts + delta_t, self.max_episode_length-1)

        # Always get the next immediate step for local goal computation
        next_ts_local = min(ts + 1, self.max_episode_length-1)

        ref_obs = self.hoi_data[None, next_ts].clone()
        ref_obs_local = self.hoi_data[None, next_ts_local].clone()
        next_ts_16 = min(ts + 16, self.max_episode_length-1)
        ref_obs_16 = self.hoi_data[None, next_ts_16].clone()

        obs, key_body_pose, key_body_rot, obs_dict = self._compute_humanoid_obs(data, ref_obs, ref_obs_16, actions, torques, episode_length, student_obs=student_obs, local_ref_obs=ref_obs_local)
        obs_shape = obs.shape[-1]
        task_obs, obj_points, obj_obs_dict = self._compute_task_obs(data, ref_obs, is_student=student_obs)
        # print('task_obs', task_obs.shape, 'obs', obs.shape)
        if not student_obs:
            task_obs = task_obs * 0
        obs = torch.cat([obs, task_obs], dim=-1)
        obs_dict.update(obj_obs_dict)
        # SDF/interaction-graph terms are always computed against world-frame object points
        # (the returned obj_points may be expressed in the camera frame for the student).
        obj_points_ig = obj_obs_dict.get("obj_points_world", obj_points)
        if student_obs:
            ig = compute_sdf(key_body_pose, obj_points_ig).view(1, -1, 3)
            ig_norm = ig.norm(dim=-1, keepdim=True)
            goal_phase = self._update_goal_phase(ig_norm, episode_length)
            self.command_goal = goal_phase
            task_obs_size = task_obs.shape[-1] if task_obs is not None else 0
            obs_final = self._finalize_student_obs(obs, obs_shape=obs_shape, task_obs_size=task_obs_size)
            self._append_goal_viz(obs_dict, ref_obs)
            if not is_frozen:
                self.long_term_t = torch.clamp(self.long_term_t - 1, min=0)
            return obs_final, obs_dict
        ig = compute_sdf(key_body_pose, obj_points_ig).view(-1, 3)
        heading_rot = torch_utils_mujoco.calc_heading_quat_inv(key_body_rot[:, 0, :])
        heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1], 1).view(-1, 4)
        ig = torch_utils_mujoco.quat_rotate(heading_rot_extend, ig).view(1, -1, 3)
        ig_norm = ig.norm(dim=-1, keepdim=True)
        ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
        ig = ig_all.view(1, -1)
        ig_all = ig_all.view(1, -1)    
        ref_ig = ref_obs[:, 630:].view(1, 39, 3)
        ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
        ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()  
        ref_ig = ref_ig.view(1, -1)
        ref_ig *= 0
        ig_all *= 0
        ig *= 0
        obs_dict.update(OrderedDict([('ig', ig_all), ('diff_ig', ref_ig-ig)]))          
        return torch.cat((obs,ig_all,ref_ig-ig),dim=-1), obs_dict

    def _compute_observations(self, data, curr_t, actions, torques, student_obs=False, episode_length=0, return_dict=False):
        obs_1, obs_dict_1 = self._compute_observations_iter(data, curr_t, 1, actions, torques, student_obs=student_obs, episode_length=episode_length)
        if student_obs:
            if return_dict:
                return obs_1, obs_dict_1
            return obs_1
        obs_2, obs_dict_2 = self._compute_observations_iter(data, curr_t, 16, actions, torques, student_obs=student_obs, episode_length=episode_length)
        # save_json_ordered(obs_dict_1, output_file)
        # return obs_1
        return torch.cat((obs_1, obs_2), dim=-1)

    def _update_goal_phase(self, ig_norm, episode_length):
        if episode_length <= 0:
            self.prev_hand_ig_valid[:] = False

        ig_norm = ig_norm.squeeze(-1)
        hand_norm = ig_norm[..., self.ig_hand_body_ids]
        current = hand_norm.mean(dim=-1)

        prev = torch.where(self.prev_hand_ig_valid, self.prev_hand_ig, current)
        delta = prev - current

        near_mask = current <= self.ig_interaction_threshold
        phase_mask = torch.full_like(
            current, episode_length > (self.max_episode_length // 2), dtype=torch.bool
        )

        approaching = (~near_mask & (delta > self.ig_transition_delta)).float()
        leaving = (~near_mask & (delta < -self.ig_transition_delta)).float()

        steady_mask = (~near_mask) & (delta.abs() <= self.ig_transition_delta)
        approaching = torch.where(steady_mask, (~phase_mask).float(), approaching)
        leaving = torch.where(steady_mask, phase_mask.float(), leaving)

        if self.command_dim >= 4:
            progress_end = torch.full_like(
                current, episode_length > (self.goal_phase_max_len - 20), dtype=torch.float32
            )
        else:
            progress_end = torch.full_like(
                current, episode_length > (self.max_episode_length - 20), dtype=torch.float32
            )

        self.prev_hand_ig = current.detach()
        self.prev_hand_ig_valid[:] = True

        if self.command_dim >= 4:
            remaining = self.long_term_t.to(device=current.device, dtype=torch.float32)
            time_to_go = torch.clamp(
                remaining / (self.goal_phase_max_len + 1e-6), 0.0, 1.0
            ).view_as(current)
            return torch.cat(
                [
                    progress_end.unsqueeze(1),
                    approaching.unsqueeze(1),
                    leaving.unsqueeze(1),
                    time_to_go.unsqueeze(1),
                ],
                dim=-1,
            )

        # Return [progress_end, approaching, leaving] to match Isaac Gym version
        return torch.cat([progress_end.unsqueeze(1), approaching.unsqueeze(1), leaving.unsqueeze(1)], dim=-1)

    def _compute_humanoid_obs(self, data, ref_obs, ref_obs_16, actions, torques, episode_length=0, student_obs=False, local_ref_obs=None):
        """
        Equivalent to your Isaac call:
          obs = compute_humanoid_observations_max(
              body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs, self._root_height_obs,
              contact_forces, self._contact_body_ids, ref_obs, self._key_body_ids, self._key_body_ids_gt,
              self._contact_body_ids_gt, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel,
              student_obs=student_obs, local_ref_obs=local_ref_obs
          )
        Returns: obs (1, D)
        """
        # --- Poses
        body_pos, body_rot = _gather_body_poses(self.model, data, self.body_ids)        # (1, Nb, 3/4)
        # --- Velocities (world)
        body_vel, body_ang_vel = _body_world_velocities(self.model, data, self.body_ids) # (1, Nb, 3)
        # --- Contact forces on selected bodies
        contact_forces = _gather_contact_forces(self.model, data, self.body_ids)

        # --- DOF slices and actuator signals
        dof_pos, dof_vel = _gather_dof_slices(
            data, self.dof_qpos_ids, self.dof_qvel_ids
        )

        # --- last_* buffers
        self._ensure_last_buffers(dof_pos, dof_vel)
        last_dof_pos = self.last_dof_pos
        last_dof_vel = self.last_dof_vel

        # Move to device if needed
        humanoid_root_states = torch.cat(
            [body_pos[:, 0, :], body_rot[:, 0, :], body_vel[:, 0, :], body_ang_vel[:, 0, :]],
            dim=-1,
        )

        tensors = [body_pos, body_rot, body_vel, body_ang_vel, contact_forces, actions,
                   dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel, ref_obs, ref_obs_16, humanoid_root_states]
        if local_ref_obs is not None:
            tensors.append(local_ref_obs)
        tensors = [t.to(self.device) for t in tensors]
        if local_ref_obs is not None:
            (body_pos, body_rot, body_vel, body_ang_vel, contact_forces, actions,
             dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel, ref_obs, ref_obs_16, humanoid_root_states, local_ref_obs) = tensors
        else:
            (body_pos, body_rot, body_vel, body_ang_vel, contact_forces, actions,
             dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel, ref_obs, ref_obs_16, humanoid_root_states) = tensors

        # Build the proprioceptive observation with the same routine used in IsaacGym training.
        obs, obs_dict = self.compute_humanoid_observations_max(
            body_pos, body_rot, body_vel, body_ang_vel,
            self.local_root_obs, self.root_height_obs,
            contact_forces, ref_obs, ref_obs_16, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel,
            humanoid_root_states, episode_length, student_obs=student_obs, local_ref_obs=local_ref_obs
        )

        # update last_* for next step
        self.last_dof_pos = dof_pos.detach()
        self.last_dof_vel = dof_vel.detach()
        return obs, body_pos, body_rot, obs_dict

    def _compute_task_obs(self, data, ref_obs=None,
                        root_body_name="pelvis", is_student=False):
        """
        Builds Isaac-style root_states and tar_states from MuJoCo, then calls
        compute_obj_observations(root_states, tar_states, object_points, ref_obs).

        Returns:
            obs:        (1, 21)
            obj_points: (1, N, 3) world points (as returned by compute_obj_observations)
        """
        assert ref_obs is not None, "ref_obs is required"
        device = ref_obs.device if isinstance(ref_obs, torch.Tensor) else "cpu"

        # -------- root (pelvis) pose -> root_states[:, :7] = [pos(3), rot_xyzw(4)]
        root_bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, root_body_name)
        root_pos = torch.from_numpy(data.xpos[root_bid].copy()).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,3)
        root_quat_wxyz = torch.from_numpy(data.xquat[root_bid].copy()).to(device=device, dtype=torch.float32)        # (4,)
        root_rot = wxyz_to_xyzw(root_quat_wxyz).unsqueeze(0)  # (1,4)
        root_states = torch.cat([root_pos, root_rot], dim=-1) # (1,7)

        # -------- object pose + world velocities -> tar_states[:, :13]
        obj_bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.object_body_name)
        tar_pos = torch.from_numpy(data.xpos[obj_bid].copy()).to(device=device, dtype=torch.float32).unsqueeze(0)     # (1,3)
        tar_quat_wxyz = torch.from_numpy(data.xquat[obj_bid].copy()).to(device=device, dtype=torch.float32)           # (4,)
        tar_rot = wxyz_to_xyzw(tar_quat_wxyz).unsqueeze(0)  # (1,4)
        tar_vel, tar_ang_vel = _single_body_world_velocities(self.model, data, obj_bid)  # (1,3), (1,3)
        tar_vel, tar_ang_vel = tar_vel.to(device), tar_ang_vel.to(device)
        tar_states = torch.cat([tar_pos, tar_rot, tar_vel, tar_ang_vel], dim=-1)  # (1,13)

        # -------- object local point cloud -> (1, N, 3)
        # self.object_points should be LOCAL (centered), like in your Isaac code.
        obj_local = self.object_points
        if obj_local is not None:
            if not isinstance(obj_local, torch.Tensor):
                obj_local = torch.as_tensor(obj_local, dtype=torch.float32, device=device)
            else:
                obj_local = obj_local.to(device=device, dtype=torch.float32)
            if obj_local.dim() == 2:
                obj_local = obj_local.unsqueeze(0)  # (1, N, 3)

        # Get camera pose from head camera body (d435_link)
        camera_pos, camera_rot = self._get_head_camera_pose(data, root_states)
        camera_pos = camera_pos.to(device)
        camera_rot = camera_rot.to(device)

        if is_student and self.object_corners is not None:
            pca_corners = self.object_corners
            if not isinstance(pca_corners, torch.Tensor):
                pca_corners = torch.as_tensor(pca_corners, dtype=torch.float32, device=device)
            else:
                pca_corners = pca_corners.to(device=device, dtype=torch.float32)
            if pca_corners.dim() == 2:
                pca_corners = pca_corners.unsqueeze(0)  # (1, 8, 3)
            obs, obj_points, obs_dict = self.compute_obj_observations_pca_corners(
                root_states,
                tar_states,
                pca_corners,
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
                fixed_surface_sampling=self.point_fixed_grid_sampling,
                surface_grid_resolution=self.point_surface_grid_resolution,
            )
        else:
            obs, obj_points, obs_dict = self.compute_obj_observations(
                root_states, tar_states, obj_local, ref_obs, camera_pos=camera_pos, camera_rot=camera_rot
            )
        return obs, obj_points, obs_dict

    def compute_humanoid_observations_max(self, body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, ref_obs, ref_obs_16, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel, humanoid_root_states, episode_length=0, student_obs=False, local_ref_obs=None):
        root_pos = _rand_vec(body_pos[:, 0, :], 0.0)
        root_rot = _rand_vec(body_rot[:, 0, :], 0.0)

        root_h = root_pos[:, 2:3]
        heading_rot = torch_utils_mujoco.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils_mujoco.calc_heading_quat(root_rot)

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
        _body_pos = _rand_vec(body_pos, 0.0)

        diff_global_body_pos = _ref_body_pos - _body_pos
        diff_local_body_pos_flat = torch_utils_mujoco.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, 39 * 3)

        local_ref_body_pos = _ref_body_pos - root_pos.unsqueeze(1)  # preserves the body position
        local_ref_body_pos = torch_utils_mujoco.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, 39 * 3)

        root_pos_expand = root_pos.unsqueeze(-2)
        local_body_pos = _body_pos - root_pos_expand
        flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
        flat_local_body_pos = torch_utils_mujoco.quat_rotate(flat_heading_rot, flat_local_body_pos)
        local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
        local_body_pos = local_body_pos[..., 3:] # remove root pos

        flat_body_rot = _rand_vec(body_rot.reshape(body_rot.shape[0] * 39, body_rot.shape[2]), 0.0)
        flat_local_body_rot = torch_utils_mujoco.quat_mul(flat_heading_rot, flat_body_rot)
        flat_local_body_rot_obs = torch_utils_mujoco.quat_to_tan_norm(flat_local_body_rot)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], 39 * flat_local_body_rot_obs.shape[1])
        
        ref_body_rot = ref_obs[:, 201:357].view(-1, 39, 4)
        ref_body_rot_no_hand = ref_body_rot
        body_rot_no_hand = _rand_vec(body_rot, 0.0)

        diff_global_body_rot = torch_utils_mujoco.quat_mul_norm(torch_utils_mujoco.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
        diff_local_body_rot_flat = torch_utils_mujoco.quat_mul(torch_utils_mujoco.quat_mul(flat_heading_rot, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot)
        diff_local_body_rot_obs = torch_utils_mujoco.quat_to_tan_norm(diff_local_body_rot_flat)
        diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

        local_ref_body_rot = torch_utils_mujoco.quat_mul(flat_heading_rot, ref_body_rot_no_hand.reshape(-1, 4))
        local_ref_body_rot = torch_utils_mujoco.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

        ref_body_vel = ref_obs[:, 357:474].view(-1, 39, 3)
        _body_vel = body_vel
        diff_global_vel = ref_body_vel - _body_vel
        diff_local_vel = torch_utils_mujoco.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, 39 * 3)

        ref_body_ang_vel = ref_obs[:, 474:591]
        ref_body_ang_vel_no_hand = ref_body_ang_vel.view(-1, 39, 3)
        body_ang_vel_no_hand = body_ang_vel
        diff_global_ang_vel = ref_body_ang_vel_no_hand - body_ang_vel_no_hand
        diff_local_ang_vel = torch_utils_mujoco.quat_rotate(flat_heading_rot, diff_global_ang_vel.view(-1, 3)).view(-1, 39 * 3)

        if (local_root_obs):
            root_rot_obs = torch_utils_mujoco.quat_to_tan_norm(root_rot)
            local_body_rot_obs[..., 0:6] = root_rot_obs

        flat_body_vel = body_vel.reshape(body_vel.shape[0] * 39, body_vel.shape[2])
        flat_local_body_vel = torch_utils_mujoco.quat_rotate(flat_heading_rot, flat_body_vel)
        local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], 39 * body_vel.shape[2])
        
        flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * 39, body_ang_vel.shape[2])
        flat_local_body_ang_vel = torch_utils_mujoco.quat_rotate(flat_heading_rot, flat_body_ang_vel)
        local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], 39 * body_ang_vel.shape[2])

        body_contact_buf = contact_forces.clone()
        contact = (torch.abs(body_contact_buf).sum(dim=-1) > 0.1).float()
        ref_body_contact = ref_obs[:,591:630]
        diff_body_contact = ref_body_contact - contact
        contact_new = torch.zeros_like(contact).to(contact.device)
        diff_body_contact_new = torch.zeros_like(diff_body_contact).to(contact.device)
        contact_indices = [6, 7, 13, 14, 28, 38]
        contact_new[:, contact_indices] = contact[:, contact_indices]
        diff_body_contact_new[:, contact_indices] = diff_body_contact[:, contact_indices]
        if student_obs:
            # Use local_ref_obs if provided, otherwise fall back to ref_obs
            if local_ref_obs is None:
                local_ref_obs = ref_obs

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

            dof_pos_rand = _rand_vec(dof_pos, 0.0)
            ref_dof_pos = ref_obs[:, 13:42]
            # Use local_ref_obs for computing local goal (next step)
            ref_dof_pos_local = local_ref_obs[:, 13:42]
            diff_dof_pos = dof_pos_rand - ref_dof_pos

            # Compute local goal from local reference (next step)
            ref_base_quat_local = local_ref_obs[:, 3:7]
            roll_ref_local, pitch_ref_local, yaw_ref_local = euler_from_quaternion(ref_base_quat_local)
            imu_obs_ref_all_local = torch.stack((roll_ref_local, pitch_ref_local, yaw_ref_local), dim=1)
            diff_imu_obs_all_local = imu_obs_ref_all_local - imu_obs_all

            root_pos = _body_pos[:, 0, :].clone()
            # Gym uses the sampled future reference for global goals in every
            # preset, while local goals use the next frame. XY stays in meters.
            ref_root_pos = _ref_body_pos[:, 0, :].clone()
            diff_imu_for_global = diff_imu_obs_all
            diff_root_pos = root_pos - ref_root_pos
            diff_root_pos_xy = diff_root_pos[..., :2]
            # Local goal uses local reference (next step)
            local_goal = torch.cat((diff_imu_obs_all_local[..., 0:2], dof_pos_rand - ref_dof_pos_local), dim=-1)
            # Global goal uses long-term reference
            global_goal = torch.cat((diff_root_pos_xy, diff_imu_for_global[..., 2:3]), dim=-1)

            last_actions = actions[:, -1, :] if actions.dim() == 3 else actions
            # Match the IsaacGym student observation: waist joint velocities are scaled by 0.05
            # (see UltraDistillObjV2Point._compute_humanoid_obs). Work on a copy so the raw
            # dof_vel returned in obs_dict / used for last_dof_vel stays unscaled.
            dof_vel_obs = dof_vel.clone()
            dof_vel_obs[:, 12:15] *= 0.05

            if self.history_step > 0:
                obs_prop = torch.cat(
                    (
                        global_goal,
                        local_goal,
                        root_ang_vel,
                        imu_obs,
                        dof_pos_rand,
                        dof_vel_obs,
                        last_actions,
                        self.obs_history_buf.view(1, -1),
                    ),
                    dim=-1,
                )
                obs_buf = torch.cat(
                    (
                        root_ang_vel,
                        imu_obs,
                        dof_pos_rand,
                        dof_vel_obs,
                        last_actions,
                    ),
                    dim=-1,
                )
                # print('obs_buf', obs_buf.shape, 'obs_prop', obs_prop.shape)
                if episode_length <= 1:  # same refill condition as IsaacGym (episode_length_buf <= 1)
                    self.obs_history_buf = torch.stack([obs_buf] * self.history_step, dim=1)
                else:
                    self.obs_history_buf = torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1)

                obs_dict = OrderedDict([
                    ("root_h_obs",               root_h_obs),                # (B, 1)
                    ("local_body_pos",           local_body_pos),            # (B, 39*3-3 = 114)  (root removed)
                    ("local_body_rot_obs",       local_body_rot_obs),        # (B, 39*6)
                    ("local_body_vel",           local_body_vel),            # (B, 39*3)
                    ("local_body_ang_vel",       local_body_ang_vel),        # (B, 39*3)
                    ("contact",                  contact),                   # (B, 39)
                    ("diff_local_body_pos_flat", diff_local_body_pos_flat),  # (B, 39*3)
                    ("diff_local_body_rot_obs",  diff_local_body_rot_obs),   # (B, 39*6)
                    ("diff_body_contact",        diff_body_contact_new),         # (B, 39)
                    ("local_ref_body_pos",       local_ref_body_pos),        # (B, 39*3)
                    ("local_ref_body_rot",       local_ref_body_rot),        # (B, 39*6)
                    ("diff_local_vel",           diff_local_vel),            # (B, 39*3)
                    ("diff_local_ang_vel",       diff_local_ang_vel),        # (B, 39*3)
                    ("actions",                  last_actions),                   # (B, A)
                    ("dof_pos",                  dof_pos_rand),                   # (B, Q)
                    ("dof_vel",                  dof_vel),                   # (B, Q)
                    ("torques",                  torques),                   # (B, Q)
                    ("last_dof_pos",             last_dof_pos),              # (B, Q)
                    ("last_dof_vel",             last_dof_vel),              # (B, Q)
                ])
                # print('obs_prop', obs_prop.shape)
                return obs_prop, obs_dict
            else:
                obs = torch.cat((global_goal, local_goal, ref_dof_pos, diff_dof_pos, root_ang_vel, imu_obs, dof_pos_rand, dof_vel, last_actions), dim=-1)
                obs_dict = OrderedDict([
                    ("root_h_obs",               root_h_obs),                # (B, 1)
                    ("local_body_pos",           local_body_pos),            # (B, 39*3-3 = 114)  (root removed)
                    ("local_body_rot_obs",       local_body_rot_obs),        # (B, 39*6)
                    ("local_body_vel",           local_body_vel),            # (B, 39*3)
                    ("local_body_ang_vel",       local_body_ang_vel),        # (B, 39*3)
                    ("contact",                  contact),                   # (B, 39)
                    ("diff_local_body_pos_flat", diff_local_body_pos_flat),  # (B, 39*3)
                    ("diff_local_body_rot_obs",  diff_local_body_rot_obs),   # (B, 39*6)
                    ("diff_body_contact",        diff_body_contact_new),         # (B, 39)
                    ("local_ref_body_pos",       local_ref_body_pos),        # (B, 39*3)
                    ("local_ref_body_rot",       local_ref_body_rot),        # (B, 39*6)
                    ("diff_local_vel",           diff_local_vel),            # (B, 39*3)
                    ("diff_local_ang_vel",       diff_local_ang_vel),        # (B, 39*3)
                    ("actions",                  last_actions),                   # (B, A)
                    ("dof_pos",                  dof_pos_rand),                   # (B, Q)
                    ("dof_vel",                  dof_vel),                   # (B, Q)
                    ("torques",                  torques),                   # (B, Q)
                    ("last_dof_pos",             last_dof_pos),              # (B, Q)
                    ("last_dof_vel",             last_dof_vel),              # (B, Q)
                ])
                return obs, obs_dict

        else:
            # local_body_vel = local_body_vel * 0
            # local_body_ang_vel = local_body_ang_vel * 0
            # local_body_ang_vel = local_body_ang_vel * 0
            # diff_local_vel = diff_local_ang_vel * 0
            # actions = actions * 0
            # dof_vel = dof_vel * 0
            # torques = torques * 0
            # last_dof_vel = last_dof_vel * 0

            obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact_new, diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact_new, local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel, actions, dof_pos, dof_vel, torques, last_dof_pos, last_dof_vel), dim=-1)
            obs_dict = OrderedDict([
                    ("root_h_obs",               root_h_obs),                # (B, 1)
                    ("local_body_pos",           local_body_pos),            # (B, 39*3-3 = 114)  (root removed)
                    ("local_body_rot_obs",       local_body_rot_obs),        # (B, 39*6)
                    ("local_body_vel",           local_body_vel),            # (B, 39*3)
                    ("local_body_ang_vel",       local_body_ang_vel),        # (B, 39*3)
                    ("contact",                  contact),                   # (B, 39)
                    ("diff_local_body_pos_flat", diff_local_body_pos_flat),  # (B, 39*3)
                    ("diff_local_body_rot_obs",  diff_local_body_rot_obs),   # (B, 39*6)
                    ("diff_body_contact",        diff_body_contact_new),         # (B, 39)
                    ("local_ref_body_pos",       local_ref_body_pos),        # (B, 39*3)
                    ("local_ref_body_rot",       local_ref_body_rot),        # (B, 39*6)
                    ("diff_local_vel",           diff_local_vel),            # (B, 39*3)
                    ("diff_local_ang_vel",       diff_local_ang_vel),        # (B, 39*3)
                    ("actions",                  actions),                   # (B, A)
                    ("dof_pos",                  dof_pos),                   # (B, Q)
                    ("dof_vel",                  dof_vel),                   # (B, Q)
                    ("torques",                  torques),                   # (B, Q)
                    ("last_dof_pos",             last_dof_pos),              # (B, Q)
                    ("last_dof_vel",             last_dof_vel),              # (B, Q)
                ])

            return obs, obs_dict
    
    def compute_obj_observations(self, root_states, tar_states, object_points, ref_obs, camera_pos=None, camera_rot=None):
        root_pos = root_states[:, 0:3]
        root_rot = root_states[:, 3:7]

        tar_pos = tar_states[:, 0:3]
        tar_rot = tar_states[:, 3:7]
        tar_vel = tar_states[:, 7:10]
        tar_ang_vel = tar_states[:, 10:13]
        tar_ang_vel_ori = tar_ang_vel.clone()

        obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        obj_points_world = torch_utils_mujoco.quat_rotate(
            obj_rot_extend, object_points_extend
        ).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)
        obj_points = obj_points_world

        heading_rot = torch_utils_mujoco.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils_mujoco.calc_heading_quat(root_rot)

        local_tar_pos = tar_pos - root_pos
        local_tar_pos[..., -1] = tar_pos[..., -1]
        local_tar_pos = torch_utils_mujoco.quat_rotate(heading_rot, local_tar_pos)
        local_tar_vel = torch_utils_mujoco.quat_rotate(heading_rot, tar_vel)
        local_tar_ang_vel = torch_utils_mujoco.quat_rotate(heading_rot, tar_ang_vel)

        local_tar_rot = torch_utils_mujoco.quat_mul(heading_rot, tar_rot)
        local_tar_rot_obs = torch_utils_mujoco.quat_to_tan_norm(local_tar_rot)

        _ref_obj_pos = ref_obs[:, 71:74]
        diff_global_obj_pos = _ref_obj_pos - tar_pos
        diff_local_obj_pos_flat = torch_utils_mujoco.quat_rotate(heading_rot, diff_global_obj_pos)
        diff_local_obj_pos_flat = self._decouple_obj_trans_goal(diff_local_obj_pos_flat)
        obj_goal_decoupled_world = None
        if self.obj_goal_decouple != "off":
            diff_global_obj_pos_decoupled = torch_utils_mujoco.quat_rotate(heading_inv_rot, diff_local_obj_pos_flat)
            obj_goal_decoupled_world = tar_pos + diff_global_obj_pos_decoupled

        local_ref_obj_pos = _ref_obj_pos - root_pos  # preserves the body position
        local_ref_obj_pos = torch_utils_mujoco.quat_rotate(heading_rot, local_ref_obj_pos)

        ref_obj_rot = ref_obs[:, 74:78]
        diff_global_obj_rot = torch_utils_mujoco.quat_mul_norm(torch_utils_mujoco.quat_inverse(ref_obj_rot), tar_rot)
        diff_local_obj_rot_flat = torch_utils_mujoco.quat_mul(torch_utils_mujoco.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot)  # Need to be change of basis
        diff_local_obj_rot_obs = torch_utils_mujoco.quat_to_tan_norm(diff_local_obj_rot_flat)

        local_ref_obj_rot = torch_utils_mujoco.quat_mul(heading_rot, ref_obj_rot)
        local_ref_obj_rot = torch_utils_mujoco.quat_to_tan_norm(local_ref_obj_rot)

        ref_obj_vel = ref_obs[:, 78:81]
        diff_global_vel = ref_obj_vel - tar_vel
        diff_local_vel = torch_utils_mujoco.quat_rotate(heading_rot, diff_global_vel)

        ref_obj_ang_vel = ref_obs[:, 81:84]
        diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
        diff_local_ang_vel = torch_utils_mujoco.quat_rotate(heading_rot, diff_global_ang_vel)

        # local_tar_ang_vel = local_tar_ang_vel * 0
        # local_tar_vel = local_tar_vel * 0
        # diff_local_vel = diff_local_vel * 0
        # diff_local_ang_vel = diff_local_ang_vel * 0
        local_tar_ang_vel = local_tar_ang_vel
        local_tar_vel = local_tar_vel
        diff_local_vel = diff_local_vel
        diff_local_ang_vel = diff_local_ang_vel
        obs = torch.cat([local_tar_vel, local_tar_ang_vel, diff_local_obj_pos_flat, diff_local_obj_rot_obs, diff_local_vel, diff_local_ang_vel], dim=-1)
        obs_dict = OrderedDict([
            ("local_tar_vel",            local_tar_vel),                # (B, 1)
            ("local_tar_ang_vel",        local_tar_ang_vel),            # (B, 39*3-3 = 114)  (root removed)
            ("diff_local_obj_pos_flat",  diff_local_obj_pos_flat),        # (B, 39*6)
            ("diff_local_obj_rot_obs",   diff_local_obj_rot_obs),            # (B, 39*3)
            ("diff_local_vel",           diff_local_vel),        # (B, 39*3)
            ("diff_local_ang_vel",       diff_local_ang_vel),                   # (B, 39)
            ("tar_ang_vel_ori", tar_ang_vel_ori),
            ("heading_rot", heading_rot),
        ])
        if obj_goal_decoupled_world is not None:
            obs_dict["viz_obj_goal_decoupled_pos"] = obj_goal_decoupled_world
        # The dense-point path returns WORLD-frame points, like the IsaacGym task's compute_obj_observations.
        # (Only the PCA-corner / egocentric path expresses points in the camera frame.) When a camera pose is
        # given, a camera-frame copy is exported for visualization only.
        obs_dict["obj_points_world"] = obj_points_world
        if camera_pos is not None and camera_rot is not None:
            cam_rot_inv = torch_utils_mujoco.quat_inverse(camera_rot)
            cam_rot_inv_expand = cam_rot_inv.unsqueeze(1).expand(-1, obj_points_world.shape[1], -1).reshape(-1, 4)
            obs_dict["obj_points_camera"] = torch_utils_mujoco.quat_rotate(
                cam_rot_inv_expand,
                (obj_points_world - camera_pos.unsqueeze(1)).reshape(-1, 3),
            ).reshape(obj_points_world.shape[0], obj_points_world.shape[1], 3)
            obs_dict["viz_camera_pos"] = camera_pos
            obs_dict["viz_camera_rot"] = camera_rot
        return obs, obj_points, obs_dict

    def compute_obj_observations_pca_corners(self, root_states, tar_states, pca_corners, ref_obs, camera_pos, camera_rot,
                                              point_noise_std=0.0, point_dropout_prob=0.0,
                                              point_outlier_prob=0.0, point_outlier_scale=0.5,
                                              point_depth_noise_scale=0.0,
                                              point_density_min=1.0, point_density_max=1.0,
                                              point_cluster_noise_std=0.0,
                                              point_scale_min=1.0, point_scale_max=1.0,
                                              point_translation_noise=0.0,
                                              point_occlusion_prob=0.0,
                                              camera_rot_noise=0.0, camera_pos_noise=0.0,
                                              disable_geometry_noise=False,
                                              fixed_surface_sampling=False,
                                              surface_grid_resolution=None):
        """
        Deployment-friendly observation using sampled surface points from a PCA box.
        Includes back-face culling for realistic depth camera simulation.

        Observation breakdown:
        - diff_local_obj_pos (3)
        - diff_local_obj_rot_obs (6)
        - local_obj_pos (3)
        - relative_points (64 x 3)

        Args:
            disable_geometry_noise: If True, skip noise on positions/rotations (for visualization)
            fixed_surface_sampling: Use a deterministic UV grid per face to mimic fixed camera pixels
            surface_grid_resolution: Optional resolution override for deterministic sampling
        """
        # Apply noise to state estimation (positions and rotations)
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

        heading_rot = torch_utils_mujoco.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils_mujoco.calc_heading_quat(root_rot)

        _ref_obj_pos = ref_obs[:, 71:74]
        diff_global_obj_pos = _ref_obj_pos - tar_pos
        diff_local_obj_pos_flat = torch_utils_mujoco.quat_rotate(heading_rot, diff_global_obj_pos)
        diff_local_obj_pos_flat = self._decouple_obj_trans_goal(diff_local_obj_pos_flat)
        obj_goal_decoupled_world = None
        if self.obj_goal_decouple != "off":
            diff_global_obj_pos_decoupled = torch_utils_mujoco.quat_rotate(heading_inv_rot, diff_local_obj_pos_flat)
            obj_goal_decoupled_world = tar_pos + diff_global_obj_pos_decoupled

        ref_obj_rot = ref_obs[:, 74:78]
        diff_global_obj_rot = torch_utils_mujoco.quat_mul_norm(torch_utils_mujoco.quat_inverse(ref_obj_rot), tar_rot)
        diff_local_obj_rot_flat = torch_utils_mujoco.quat_mul(
            torch_utils_mujoco.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)),
            heading_inv_rot,
        )
        diff_local_obj_rot_obs = torch_utils_mujoco.quat_to_tan_norm(diff_local_obj_rot_flat)

        local_obj_pos = tar_pos - root_pos
        local_obj_pos[..., -1] = tar_pos[..., -1]
        local_obj_pos = torch_utils_mujoco.quat_rotate(heading_rot, local_obj_pos)

        # NOTE: We do NOT add noise to pca_corners - these are fixed geometric properties
        if pca_corners.dim() == 2:
            pca_corners = pca_corners.unsqueeze(0).expand(root_pos.shape[0], -1, -1)

        corner_center = pca_corners.mean(dim=1, keepdim=True)
        centered_corners = pca_corners - corner_center
        cov = torch.matmul(centered_corners.transpose(1, 2), centered_corners) / float(centered_corners.shape[1])
        pca_axes = box_pca_axes(pca_corners, cov)
        aligned_corners = torch.matmul(centered_corners, pca_axes)
        local_min = aligned_corners.min(dim=1).values
        local_max = aligned_corners.max(dim=1).values

        # Oversample for visibility filtering
        num_points = 64
        num_candidates = max(num_points * 8, num_points)
        aligned_candidates, aligned_normals = _sample_box_surface_points(
            local_min,
            local_max,
            num_candidates,
            deterministic=fixed_surface_sampling,
            grid_resolution=surface_grid_resolution,
        )
        local_candidates = torch.matmul(aligned_candidates, pca_axes.transpose(1, 2)) + corner_center
        local_normals = torch.matmul(aligned_normals, pca_axes.transpose(1, 2))

        # Transform points and normals to world frame
        tar_rot_expand = tar_rot.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, 4)
        points_world = torch_utils_mujoco.quat_rotate(
            tar_rot_expand,
            local_candidates.reshape(-1, 3),
        ).reshape(root_pos.shape[0], num_candidates, 3) + tar_pos.unsqueeze(1)

        normals_world = torch_utils_mujoco.quat_rotate(
            tar_rot_expand,
            local_normals.reshape(-1, 3),
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
            noisy_camera_rot = torch_utils_mujoco.quat_mul(camera_rot, noise_quat)

        # Compute camera frame from camera rotation (not root rotation!)
        forward_local = torch.tensor([1.0, 0.0, 0.0], device=camera_pos.device, dtype=camera_pos.dtype).unsqueeze(0)
        up_local = torch.tensor([0.0, 0.0, 1.0], device=camera_pos.device, dtype=camera_pos.dtype).unsqueeze(0)
        forward_local = forward_local.expand(noisy_camera_rot.shape[0], -1)
        up_local = up_local.expand(noisy_camera_rot.shape[0], -1)
        cam_forward = torch_utils_mujoco.quat_rotate(noisy_camera_rot, forward_local)
        cam_forward = F.normalize(cam_forward, dim=-1)
        cam_up = torch_utils_mujoco.quat_rotate(noisy_camera_rot, up_local)
        cam_up = F.normalize(cam_up, dim=-1)
        cam_right = F.normalize(torch.cross(cam_forward, cam_up, dim=-1), dim=-1)
        cam_up = F.normalize(torch.cross(cam_right, cam_forward, dim=-1), dim=-1)

        # Select visible points using frustum and back-face culling
        visible_world = _select_visible_points(
            points_world, noisy_camera_pos, cam_forward, cam_right, cam_up, num_points,
            normals_world=normals_world,
        )

        # Apply domain randomization to visible points
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

        # The trained policy consumes points in the full head-camera frame.
        cam_rot_inv = torch_utils_mujoco.quat_inverse(noisy_camera_rot)
        cam_rot_inv_expand = cam_rot_inv.unsqueeze(1).expand(-1, num_points, -1)
        relative_points = torch_utils_mujoco.quat_rotate(
            cam_rot_inv_expand.reshape(-1, 4),
            (randomized_world - noisy_camera_pos.unsqueeze(1)).reshape(-1, 3),
        ).reshape(root_pos.shape[0], num_points, 3)

        # Apply dropout mask
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

        # Compute PCA corners in world frame for visualization (use actual tar_states)
        tar_pos_actual = tar_states[:, 0:3]
        tar_rot_actual = tar_states[:, 3:7]
        pca_corners_local = pca_corners.squeeze(0) if pca_corners.dim() == 3 else pca_corners
        pca_corners_world = torch_utils_mujoco.quat_rotate(
            tar_rot_actual.expand(pca_corners_local.shape[0], -1),
            pca_corners_local
        ) + tar_pos_actual

        obs_dict = OrderedDict([
            ("diff_local_obj_pos_flat", diff_local_obj_pos_flat),
            ("diff_local_obj_rot_obs", diff_local_obj_rot_obs),
            ("local_obj_pos", local_obj_pos),
            ("relative_points", relative_points.reshape(root_pos.shape[0], -1)),
            # Visualization data (similar to Isaac Gym debug info)
            ("viz_points_clean", visible_world),          # BLUE: Clean visible points (before randomization)
            ("viz_points_randomized", randomized_world),  # GREEN: Randomized/noisy points (what policy sees)
            ("viz_camera_pos", camera_pos),               # RED: Camera position
            ("viz_camera_rot", camera_rot),               # Camera rotation (xyzw)
            ("viz_pca_corners", pca_corners_world),       # YELLOW: PCA bounding box corners in world frame
        ])
        if obj_goal_decoupled_world is not None:
            obs_dict["viz_obj_goal_decoupled_pos"] = obj_goal_decoupled_world

        # Return visible_world (before randomization) for IG computation
        return obs, visible_world, obs_dict

def to_plain(x):
    if isinstance(x, OrderedDict):
        return {k: to_plain(v) for k, v in x.items()}
    if isinstance(x, dict):
        return {k: to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(x, np.generic):
        return x.item()
    return x

def save_json_ordered(od, path="out.json"):
    import json
    with open(path, "w") as f:
        json.dump(to_plain(od), f, indent=2, ensure_ascii=False)
