
import argparse
import sys

import os
import time
import numpy as np
import copy
from pathlib import Path
import yaml

# Check if we need EGL for headless rendering before importing mujoco
if '--record_video' in sys.argv:
    os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import mujoco.viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R

from rl_games.algos_torch import torch_ext
from learning import ultra_network_builder_obj_v2, ultra_models
import torch
import trimesh
from utils.obs_vae import MujocoObs, compute_sdf, _gather_dof_slices
import xml.etree.ElementTree as ET
from utils import torch_utils_mujoco

def _randn_like(x, std):
    """Add zero-mean Gaussian noise with per-element std."""
    if std == 0:
        return x
    return x + torch.randn_like(x) * std
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    return quat

def _quat_rotate_np(q_xyzw, v):
    """Rotate vector v by quaternion q (xyzw)."""
    q = np.asarray(q_xyzw, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    q_xyz = q[:3]
    qw = q[3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + qw * t + np.cross(q_xyz, t)

def _normalize_np(v, eps=1e-6):
    n = float(np.linalg.norm(v))
    if n < eps:
        return v, n
    return v / n, n
def _perturb_quat(q, std):
    """Apply small-angle axis-angle noise then renormalise."""
    if std == 0:
        return q
    # random unit axes
    q = q.unsqueeze(0)
    axis = torch.randn_like(q[..., :3])
    axis = axis / torch.norm(axis, dim=-1, keepdim=True).clamp_(min=1e-6)
    angle = torch.randn(q.shape[0], 1, device=q.device) * std  # radians
    half  = 0.5 * angle
    delta = torch.cat([axis * torch.sin(half), torch.cos(half)], dim=-1)
    qn    = quat_mul(delta, q)
    return (qn / torch.norm(qn, dim=-1, keepdim=True)).squeeze(0)


_TRAIN_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "cfg"
    / "train"
    / "rlg"
    / "g1_student_vae.yaml"
)


def _load_network_config_from_yaml(config_path=None):
    """Load the training-time network config so sim2sim matches the deployed policy."""
    path = Path(config_path) if config_path is not None else _TRAIN_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    params = raw_cfg.get("params", {})
    network_cfg = copy.deepcopy(params.get("network"))
    if not network_cfg:
        raise ValueError(f"'network' section missing from {path}")
    if "vae_dim" not in network_cfg:
        vae_dim = params.get("config", {}).get("vae_dim")
        if vae_dim is not None:
            network_cfg["vae_dim"] = vae_dim
    return network_cfg


# _DEFAULT_NOISE_STD = {
#     "root_pos"     : 0.05,  # m
#     "root_rot"     : 0.04,  # rad (axis–angle magnitude)
#     "dof_pos"      : 0.20,  # rad or m
#     "root_vel"     : 0.05,  # m/s
#     "root_ang_vel" : 0.05,  # rad/s
#     "dof_vel"      : 0.05,  # rad/s or m/s
#     "target_pos"   : 0.05,  # m
#     "target_rot"   : 0.05,  # rad
# }

_DEFAULT_NOISE_STD = {
    "root_pos"     : 0.0,  # m
    "root_rot"     : 0.0,  # rad (axis–angle magnitude)
    "dof_pos"      : 0.0,  # rad or m
    "root_vel"     : 0.0,  # m/s
    "root_ang_vel" : 0.0,  # rad/s
    "dof_vel"      : 0.0,  # rad/s or m/s
    "target_pos"   : 0.0,  # m
    "target_rot"   : 0.0,  # rad
}



def to_torch(x, device=None, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.as_tensor(x, dtype=dtype, device=device)

def _first_or_new(parent, tag):
    node = parent.find(tag)
    if node is None:
        node = ET.SubElement(parent, tag)
    return node

def merge_mjcf(humanoid_xml, object_xml, out_xml, viz_sites=0, viz_site_size=0.02):
    """
    - Appends object's <asset> (rewriting mesh 'file' to absolute paths to avoid meshdir surprises)
    - Appends object's top-level <worldbody>/<body> into humanoid <worldbody>
    - Appends object's <default> children (keeps your friction class)
    """
    h_tree, o_tree = ET.parse(humanoid_xml), ET.parse(object_xml)
    h_root, o_root = h_tree.getroot(), o_tree.getroot()

    # --- make the humanoid mesh directory absolute so the merged file can live anywhere
    h_compiler = h_root.find('compiler')
    if h_compiler is not None and 'meshdir' in h_compiler.attrib and not os.path.isabs(h_compiler.attrib['meshdir']):
        h_compiler.set('meshdir', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(humanoid_xml)), h_compiler.attrib['meshdir'])))

    # --- asset merge
    h_asset = _first_or_new(h_root, 'asset')
    o_asset = o_root.find('asset')
    if o_asset is not None:
        base = os.path.dirname(os.path.abspath(object_xml))
        for child in list(o_asset):
            if child.tag == 'mesh' and 'file' in child.attrib:
                fp = child.attrib['file']
                if not os.path.isabs(fp):
                    child.set('file', os.path.normpath(os.path.join(base, fp)))
            h_asset.append(child)

    # --- default merge (preserve object default class for friction, etc.)
    o_default = o_root.find('default')
    if o_default is not None:
        h_default = _first_or_new(h_root, 'default')
        for child in list(o_default):
            h_default.append(child)

    # --- worldbody merge
    h_world = _first_or_new(h_root, 'worldbody')
    o_world = o_root.find('worldbody')
    if o_world is not None:
        # Bring each top-level body over (e.g., <body name="largebox" ...>)
        existing_names = {b.get('name') for b in h_world.findall('body')}
        for body in o_world.findall('body'):
            name = body.get('name', 'object')
            if name in existing_names:
                body.set('name', f"{name}_obj")
            h_world.append(body)

    # --- add visualization sites (worldbody)
    viz_sites = int(viz_sites) if viz_sites is not None else 0
    if viz_sites > 0:
        site_size = float(viz_site_size)
        site_size = max(site_size, 1e-4)
        for prefix, rgba in (
            ("viz_point_g_", "0.2 0.9 0.2 1"),
            ("viz_point_b_", "0.2 0.4 0.95 1"),
        ):
            for i in range(viz_sites):
                ET.SubElement(
                    h_world,
                    "site",
                    {
                        "name": f"{prefix}{i}",
                        "type": "sphere",
                        "size": f"{site_size}",
                        "rgba": rgba,
                        "pos": "0 0 -1000",
                        "group": "0",
                    },
                )
        ET.SubElement(
            h_world,
            "site",
            {
                "name": "viz_goal_obj",
                "type": "sphere",
                "size": f"{site_size * 2.0}",
                "rgba": "0.9 0.6 0.1 1",
                "pos": "0 0 -1000",
                "group": "0",
            },
        )
        ET.SubElement(
            h_world,
            "site",
            {
                "name": "viz_goal_obj_decoupled",
                "type": "sphere",
                "size": f"{site_size * 2.0}",
                "rgba": "0.1 0.9 0.9 1",
                "pos": "0 0 -1000",
                "group": "0",
            },
        )
        ET.SubElement(
            h_world,
            "site",
            {
                "name": "viz_goal_human",
                "type": "sphere",
                "size": f"{site_size * 2.0}",
                "rgba": "0.9 0.1 0.9 1",
                "pos": "0 0 -1000",
                "group": "0",
            },
        )

    # Write combined file
    os.makedirs(os.path.dirname(os.path.abspath(out_xml)), exist_ok=True)
    h_tree.write(out_xml, encoding='utf-8', xml_declaration=True)
    return out_xml


def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="jit"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint


def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0] 
    qx = quat[1] 
    qy = quat[2]
    qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    
    return eulerVec


# -------- helpers --------
def xyzw_to_wxyz(q):
    if isinstance(q, torch.Tensor):
        return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)

def _joint_id_by_name_or_body(model, name: str, require_free: bool = True) -> int:
    """Return joint id by joint name; if not found, try body name → its freejoint."""
    # force plain str (numpy.str_, torch scalar, etc. → str)
    name = str(name)

    # Cast enum to int for pybind
    OBJ_JOINT = int(mujoco.mjtObj.mjOBJ_JOINT)
    OBJ_BODY  = int(mujoco.mjtObj.mjOBJ_BODY)
    JNT_FREE  = int(mujoco.mjtJoint.mjJNT_FREE)

    jid = mujoco.mj_name2id(model, OBJ_JOINT, name)
    if jid >= 0:
        if require_free and model.jnt_type[jid] != JNT_FREE:
            raise ValueError(f"Joint '{name}' exists but is not FREE.")
        return jid

    # Try as a body name: find a FREE joint attached to that body
    bid = mujoco.mj_name2id(model, OBJ_BODY, name)
    if bid >= 0:
        adr = model.body_jntadr[bid]
        n   = model.body_jntnum[bid]
        for k in range(n):
            j = adr + k
            if model.jnt_type[j] == JNT_FREE:
                return j
        raise ValueError(f"Body '{name}' has no FREE joint.")
    raise ValueError(f"Neither joint nor body named '{name}' found.")

def _set_freejoint_state(model, data, joint_or_body_name, pos_xyz, quat_xyzw, lin_vel_xyz, ang_vel_xyz):
    """
    Correctly sets a MuJoCo free joint's state using WORLD-frame inputs.
    """
    jid = _joint_id_by_name_or_body(model, joint_or_body_name, require_free=True)
    qpos_adr = model.jnt_qposadr[jid]
    qvel_adr =  model.jnt_dofadr[jid]

    # Set position and orientation (qpos)
    p = pos_xyz.detach().cpu().numpy().astype(np.float64)
    q = xyzw_to_wxyz(quat_xyzw.detach().cpu().numpy().astype(np.float64))
    data.qpos[qpos_adr:qpos_adr+7] = np.concatenate([p, q])

    # CRITICAL: Run mj_forward to update the model's kinematics (like body orientations)
    # This ensures data.body().xmat is correct for the velocity transformation.
    mujoco.mj_forward(model, data)

    # --- Set Velocity (qvel) ---
    # Linear velocity is already in the world frame and can be set directly.
    data.qvel[qvel_adr:qvel_adr+3] = lin_vel_xyz.detach().cpu().numpy().astype(np.float64)
    
    # Angular velocity from Isaac Gym is in the WORLD frame.
    world_ang_vel = ang_vel_xyz.detach().cpu().numpy().astype(np.float64)

    # Get the body's rotation matrix (which was just updated by mj_forward).
    body_id = model.jnt_bodyid[jid]
    body_rot_mat = data.body(body_id).xmat.reshape(3, 3)

    # Transform WORLD angular velocity to the body's LOCAL frame.
    # v_local = R_transpose * v_world
    local_ang_vel = body_rot_mat.T @ world_ang_vel

    # Set the rotational part of qvel with the correct LOCAL velocity.
    data.qvel[qvel_adr+3:qvel_adr+6] = local_ang_vel
    # print(joint_or_body_name, world_ang_vel, local_ang_vel)
        
# Order for the 29 human hinge joints (matches your XML; tweak if your training order differs)
JOINT_NAMES_29 = [
    # Left leg
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
    "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    # Right leg
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
    "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
    # Waist
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    # Left arm
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    # Right arm
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
]

def _write_hinge_qpos_qvel_by_names(model, data, joint_names, dof_pos, dof_vel, clamp_to_range=False):
    dof_pos = dof_pos.view(-1)  # (29,)
    dof_vel = dof_vel.view(-1)  # (29,)
    assert dof_pos.numel() == len(joint_names) and dof_vel.numel() == len(joint_names), \
        f"Expected {len(joint_names)} dofs, got {dof_pos.numel()}/{dof_vel.numel()}"

    for k, jn in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Joint '{jn}' not found in model.")
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE, f"Joint '{jn}' not hinge."
        qpos_adr = model.jnt_qposadr[jid]
        qvel_adr = model.jnt_dofadr[jid]
        # print(jid, qpos_adr, qvel_adr)
        q = float(dof_pos[k].detach().cpu().item())
        dq = float(dof_vel[k].detach().cpu().item())

        if clamp_to_range and model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q = np.clip(q, lo, hi)

        data.qpos[qpos_adr] = q
        data.qvel[qvel_adr] = dq
        
class HumanoidEnv:
    def __init__(
        self,
        policy_path,
        robot_type="g1_29dof",
        device="cuda",
        record_video=False,
        motion_path=None,
        use_jit=False,
        obj_rot_keep_prob=1.0,
        obj_trans_keep_prob=1.0,
        obj_point_keep_prob=1.0,
        obj_pos_keep_prob=1.0,
        human_move_keep_prob=1.0,
        human_global_keep_prob=None,
        human_local_keep_prob=None,
        human_goal_keep_prob=1.0,
        mask_flip_prob=0.0,
        # Point cloud domain randomization parameters
        point_noise_std=0.0,
        point_dropout_prob=0.0,
        point_outlier_prob=0.0,
        point_outlier_scale=0.5,
        point_depth_noise_scale=0.0,
        point_density_min=1.0,
        point_density_max=1.0,
        point_cluster_noise_std=0.0,
        point_scale_min=1.0,
        point_scale_max=1.0,
        point_translation_noise=0.0,
        point_occlusion_prob=0.0,
        camera_rot_noise=0.0,
        camera_pos_noise=0.0,
        point_fixed_grid_sampling=True,
        point_surface_grid_resolution=None,
        render_camera="tracking",
        object_alpha=0.2,
        point_viz_scale=8.0,
        point_viz_offset=0.0,
        point_viz_sites=64,
        point_viz_site_size=0.02,
        goal_achievement_enabled=False,
        goal_pos_threshold=0.3,
        obj_goal_decouple="off",
        obj_goal_z_threshold=0.15,
        goal_phase_dim=4,
        align_training_dynamics=True,
        match_training_reset=True,
    ):
        self.robot_type = robot_type
        self.device = device
        self.record_video = record_video
        self.motion_path = motion_path
        self.use_jit = use_jit
        self.render_camera = render_camera
        self.object_alpha = object_alpha
        self.point_viz_scale = float(point_viz_scale)
        self.point_viz_offset = float(point_viz_offset)
        self.point_viz_sites = int(point_viz_sites)
        self.point_viz_site_size = float(point_viz_site_size)
        self._use_mjcf_markers = self.point_viz_sites > 0
        self.goal_phase_dim = int(goal_phase_dim)
        self.match_training_reset = match_training_reset
        if self.goal_phase_dim not in (3, 4):
            raise ValueError(f"goal_phase_dim must be 3 or 4, got {self.goal_phase_dim}")
        human_path = f"ultra/data/assets/g1/g1_29dof.xml"
        object_name = motion_path.split('/')[-1].split('_')[1]
        object_path = f"ultra/data/assets/objects/{object_name}.xml"
        self._load_object(object_name, object_path)
        self.object_name = object_name
        self._load_motion(motion_path)
        os.makedirs("ultra/data/assets/merge/", exist_ok=True)
        model_path = f"ultra/data/assets/merge/{object_name}.xml"
        merge_mjcf(
            human_path,
            object_path,
            f"ultra/data/assets/merge/{object_name}.xml",
            viz_sites=self.point_viz_sites,
            viz_site_size=self.point_viz_site_size,
        )

        self.stiffness = np.array([
            150, 150, 200, 200, 20, 20,
            150, 150, 200, 200, 20, 20,
            200, 200, 200,
            40, 40, 40, 40, 20, 20, 20,
            40, 40, 40, 40, 20, 20, 20,
        ], dtype=np.float32)
        self.damping = np.array([
            5, 5, 5, 5, 4, 4,
            5, 5, 5, 5, 4, 4,
            5, 5, 5,
            10, 10, 10, 10, 0.5, 0.5, 0.5,
            10, 10, 10, 10, 0.5, 0.5, 0.5,
        ], dtype=np.float32)

        self.stiffness = np.array([40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
                        40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
                        40.179238, 28.501246, 28.501246,
                        14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
                        14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327])

        # Kd (damping) per DOF
        self.damping = np.array([2.557890, 6.308802, 2.557890, 6.308802, 1.814446, 1.814446,
                2.557890, 6.308802, 2.557890, 6.308802, 1.814446, 1.814446,
                2.557890, 1.814446, 1.814446,
                0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
                0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142])
        # armature per DOF (NOTE: feet + waist roll/pitch are doubled vs ARMATURE_5020)
        armature = [0.010178, 0.025102, 0.010178, 0.025102, 0.007219, 0.007219,
                    0.010178, 0.025102, 0.010178, 0.025102, 0.007219, 0.007219,
                    0.010178, 0.007219, 0.007219,
                    0.003610, 0.003610, 0.003610, 0.003610, 0.003610, 0.004250, 0.004250,
                    0.003610, 0.003610, 0.003610, 0.003610, 0.003610, 0.004250, 0.004250]

        # effort limits per DOF (used for dof_prop["effort"] and for torque_limits)
        effort = [88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                88.0, 50.0, 50.0,
                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0]

        self.control_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]) # all dofs
        self.num_actions = len(self.control_indices)
        self.num_dofs = 29
        self.default_dof_pos = np.array([
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # left leg (6)
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # right leg (6)
            0.0, 0.0, 0.0,  # torso (1)
            0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
        ])
        self.torque_limits = np.array([88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                88.0, 50.0, 50.0,
                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
                25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0])
        self.cycle_time = 0.64

        self.torque_limits = self.torque_limits.astype(np.float32) * (0.8 if align_training_dynamics else 1.0)
        
        self.obs_indices = np.arange(self.num_dofs)
        
        if self.record_video:
            self.sim_duration = 10.0
        else:
            self.sim_duration = 60.0
        self.sim_dt = 1.0 / (60.0 * 17.0)
        self.sim_decimation = 17
        # self.sim_dt = 0.001
        # self.sim_decimation = 20
        self.control_dt = self.sim_dt * self.sim_decimation
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        if align_training_dynamics:
            joint_dofs = np.array([int(self.model.joint(name).dofadr[0]) for name in JOINT_NAMES_29])
            self.model.dof_armature[joint_dofs] = armature
            self.model.dof_damping[joint_dofs] = 0.0
            self.model.dof_frictionloss[joint_dofs] = 0.0
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        if align_training_dynamics:
            mujoco.mj_setConst(self.model, self.data)
        self._set_object_transparency(self.object_alpha)
        self._init_viz_sites()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)
        if self.record_video:
            # Use MuJoCo's native offscreen rendering instead of mujoco_viewer
            self.viewer = None
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

            # IMPORTANT: Enable scene visualization which is needed for custom geometries
            self.renderer.enable_depth_rendering()
            self.renderer.enable_segmentation_rendering()

            # Set initial camera parameters
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = 0  # Track the root body (pelvis)
            self.camera.distance = 5.0
            self.camera.elevation = -20
            self.camera.azimuth = 140

            # Create scene and context for custom geometry rendering (point clouds)
            self.viz_scene = mujoco.MjvScene(self.model, maxgeom=20000)
            self.viz_option = mujoco.MjvOption()

            # Enable rendering of all geometry types
            self.viz_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 1
            self.viz_option.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 1
            self.viz_option.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
            self.viz_option.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 1
            self.viz_option.sitegroup[:] = 1

            # Create viewport and context for low-level rendering
            self.viewport = mujoco.MjrRect(0, 0, 640, 480)

            # Initialize MjrContext for rendering - this is CRITICAL
            self.mjr_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        else:
            # Use MuJoCo's passive viewer for interactive visualization
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 5.0
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 140
            self.viewer.opt.sitegroup[:] = 1

            # CRITICAL: Enable user scene rendering in the viewer
            # The user scene is where we add our custom spheres for visualization
            self.viewer.user_scn.ngeom = 0
            print("Note: Press 'V' in the MuJoCo viewer to toggle visualization flags if spheres don't appear")

        # Point cloud visualization data (similar to Isaac Gym)
        self._visualize_points = True
        self._debug_points_green = None  # Randomized points (what policy sees)
        self._debug_points_blue = None   # Clean visible points (before randomization)
        self._debug_camera_pos = None    # Camera position (red)
        self._debug_camera_rot = None    # Camera rotation (xyzw)
        self._debug_pca_corners = None   # PCA bounding box corners (yellow)
        self._debug_points_center = None  # Point cloud centroid (magenta)
        self._debug_obj_goal_pos = None   # Object translation goal (world)
        self._debug_obj_goal_enabled = False
        self._debug_obj_goal_decoupled_pos = None  # Decoupled object goal (world)
        self._debug_obj_goal_decoupled_enabled = False
        self._debug_human_goal_pos = None  # Human global translation goal (world)
        self._debug_human_goal_enabled = False
        
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_scale = 0.3
        
        self.n_priv = 0

        self.n_proprio = 3 + 2 + 3*self.num_dofs
        self.n_priv_latent = 4 + 1 + 2*self.num_dofs + 3
            
        self.history_len = 10
        
        self.dof_pos_scale = 1.0
        # self.dof_vel_scale = 0.05
        self.dof_vel_scale = 1.0
        # self.ang_vel_scale = 0.25
        self.ang_vel_scale = 1.0
        
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))

        # Command goal: [progress_end, approaching, leaving] (3D)
        self.commands = np.zeros(self.goal_phase_dim)

        self.set_commands()
        self.obs_builder = MujocoObs(
            self.model,
            object_name,
            self.max_episode_length,
            self.hoi_data,
            self.object_points,
            self.object_corners,
            history_step=10,
            obj_rot_keep_prob=obj_rot_keep_prob,
            obj_trans_keep_prob=obj_trans_keep_prob,
            obj_point_keep_prob=obj_point_keep_prob,
            obj_pos_keep_prob=obj_pos_keep_prob,
            human_move_keep_prob=human_move_keep_prob,
            human_global_keep_prob=human_global_keep_prob,
            human_local_keep_prob=human_local_keep_prob,
            human_goal_keep_prob=human_goal_keep_prob,
            mask_flip_prob=mask_flip_prob,
            point_noise_std=point_noise_std,
            point_dropout_prob=point_dropout_prob,
            point_outlier_prob=point_outlier_prob,
            point_outlier_scale=point_outlier_scale,
            point_depth_noise_scale=point_depth_noise_scale,
            point_density_min=point_density_min,
            point_density_max=point_density_max,
            point_cluster_noise_std=point_cluster_noise_std,
            point_scale_min=point_scale_min,
            point_scale_max=point_scale_max,
            point_translation_noise=point_translation_noise,
            point_occlusion_prob=point_occlusion_prob,
            camera_rot_noise=camera_rot_noise,
            camera_pos_noise=camera_pos_noise,
            point_fixed_grid_sampling=point_fixed_grid_sampling,
            point_surface_grid_resolution=point_surface_grid_resolution,
            goal_achievement_enabled=goal_achievement_enabled,
            goal_pos_threshold=goal_pos_threshold,
            goal_phase_dim=self.goal_phase_dim,
            obj_goal_decouple=obj_goal_decouple,
            obj_goal_z_threshold=obj_goal_z_threshold,
        )
        self.obs_builder.set_command_goal(self.commands)

        self.policy_path = policy_path

        # Load policy based on use_jit flag
        if self.use_jit:
            print(f"Loading JIT model from: {policy_path}")
            self.policy_jit = torch.jit.load(policy_path, map_location=self.device)
            self.policy_jit.eval()
            print("✓ JIT model loaded successfully")
            # JIT model doesn't need running_mean/running_var (already included)
            self.policy = None
            self.running_mean = None
            self.running_var = None
        else:
            print(f"Loading checkpoint from: {policy_path}")
            num_envs = 1
            obs_dim = 1494 + 2 * max(0, self.goal_phase_dim - 3)
            config = {
                'actions_num' : 29,
                'input_shape' : (obs_dim, ),
                'num_seqs' : num_envs,
                'value_size': 1,
            }
            ck = torch.load(self.policy_path, map_location="cpu")
            is_rl = any(".rl_finetune_film." in key for key in ck["model"])
            if is_rl:
                from learning import ultra_network_builder_obj_v3
                network = ultra_network_builder_obj_v3.UltraBuilder()
                network_params = _load_network_config_from_yaml(
                    Path(__file__).resolve().parent / "data/cfg/train/rlg/g1_student_finetune.yaml")
            else:
                network = ultra_network_builder_obj_v2.UltraBuilder()
                network_params = _load_network_config_from_yaml()
            network.load(network_params)
            network = ultra_models.ModelUltraContinuous(network)
            policy = network.build(config)
            policy.to(self.device)
            policy.load_state_dict(ck['model'])
            self.policy = policy
            self.running_mean = None
            self.running_var = None
            self.policy_jit = None
            self.vae_dim = self.policy.a2c_network.vae_dim
            self.policy_architecture = "v3_rl" if is_rl else "v2_distill"
            print("✓ Checkpoint loaded successfully")

    def _load_motion(self, data_path):
        loaded_dict = {}
        hoi_data = torch.load(data_path, map_location="cpu")
        # Match training padding; sequence length also controls phase inputs.
        hoi_data = torch.cat([hoi_data, hoi_data[-1:].repeat(20, 1)], dim=0)
        # hoi_data[5:] = hoi_data[4:5]
        loaded_dict['hoi_data'] = hoi_data.detach().to('cpu')

        self.max_episode_length = loaded_dict['hoi_data'].shape[0]
        self.fps_data = new_fps = 60.

        loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
        loaded_dict['root_pos_vel'] = loaded_dict['hoi_data'][:, 7:10].clone()

        loaded_dict['root_rot'] = loaded_dict['hoi_data'][:, 3:7]

        loaded_dict['root_rot_vel'] = loaded_dict['hoi_data'][:, 10:13]

        loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 13:42]

        loaded_dict['dof_pos_vel'] = loaded_dict['hoi_data'][:, 42:71]

        loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 84:201]
        loaded_dict['body_pos_vel'] = loaded_dict['hoi_data'][:, 357:474]

        loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, 71:74]

        loaded_dict['obj_pos_vel'] = loaded_dict['hoi_data'][:, 78:81]

        loaded_dict['obj_rot'] = loaded_dict['hoi_data'][:, 74:78]
        loaded_dict['obj_rot_vel'] = loaded_dict['hoi_data'][:, 81:84]

        obj_rot_extend = loaded_dict['obj_rot'].unsqueeze(1).repeat(1, self.object_points.shape[0], 1).view(-1, 4)
        object_points_extend = self.object_points.unsqueeze(0).repeat(loaded_dict['obj_rot'].shape[0], 1, 1).view(-1, 3)
        obj_points = torch_utils_mujoco.quat_rotate(obj_rot_extend, object_points_extend).view(loaded_dict['obj_rot'].shape[0], self.object_points.shape[0], 3) + loaded_dict['obj_pos'].unsqueeze(1)
        key_body_pose = loaded_dict['body_pos'][:,:].clone()
        ref_ig = compute_sdf(key_body_pose.view(loaded_dict['obj_rot'].shape[0],-1,3), obj_points).view(-1, 3)
        heading_rot = torch_utils_mujoco.calc_heading_quat_inv(loaded_dict['root_rot'])
        heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1] // 3, 1).view(-1, 4)
        ref_ig = torch_utils_mujoco.quat_rotate(heading_rot_extend, ref_ig).view(loaded_dict['obj_rot'].shape[0], -1)    
        loaded_dict['contact_parts'] = loaded_dict['hoi_data'][:, 591:630].clone()

        loaded_dict['human_rot'] = loaded_dict['hoi_data'][:, 201:357]
        loaded_dict['human_rot_vel'] = loaded_dict['hoi_data'][:, 474:591]
        loaded_dict['hoi_data'] = torch.cat([loaded_dict['hoi_data'], ref_ig], dim=-1)

        self.hoi_data_dict = loaded_dict
        self.hoi_data = loaded_dict['hoi_data']
        
        self.hoi_ref = torch.cat((
                            loaded_dict['root_pos'].clone(), # 0:3
                            loaded_dict['root_rot'].clone(), # 3:7
                            loaded_dict['root_pos_vel'].clone(), # 7:10
                            loaded_dict['root_rot_vel'].clone(), # 10:13
                            loaded_dict['dof_pos'].clone(), # 13:42
                            loaded_dict['dof_pos_vel'].clone(), # 42:71
                            loaded_dict['obj_pos'].clone(), # 71:74
                            loaded_dict['obj_rot'].clone(), # 74:78
                            loaded_dict['obj_pos_vel'].clone(), # 78:81
                            loaded_dict['obj_rot_vel'].clone(), # 81:84
                            ),dim=-1)

        return

    def _load_object(self, object_name, object_path): # smplx
        asset_root = "ultra/data/assets/objects/"
        asset_file = object_name + ".xml"
        obj_file = asset_root + 'objects/' + object_name + '/' + object_name + '.obj'
        

        mesh_obj = trimesh.load(obj_file, process=False, force='mesh')
        obb = mesh_obj.bounding_box_oriented
        corners = obb.vertices
        obj_verts = mesh_obj.vertices
        center = np.mean(obj_verts, 0)
        object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=256, seed=2024)

        object_points = to_torch(object_points - center)
        corners = to_torch(corners - center)

        while object_points.shape[0] < 256:
            object_points = torch.cat([object_points, object_points[:256 - object_points.shape[0]]], dim=0)
        self.object_points = to_torch(object_points)
        self.object_corners = to_torch(corners)
        return

    def _set_object_transparency(self, alpha):
        """Make object geoms semi-transparent for easier point-cloud viewing."""
        if alpha is None:
            return
        alpha = float(alpha)
        alpha = max(0.0, min(1.0, alpha))
        if alpha >= 1.0:
            return
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.object_name)
        if body_id < 0:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.object_name}_obj")
        if body_id < 0:
            print(f"Warning: object body '{self.object_name}' not found; transparency not applied")
            return
        root_id = int(self.model.body_rootid[body_id])
        body_ids = np.where(self.model.body_rootid == root_id)[0]
        geom_ids = np.where(np.isin(self.model.geom_bodyid, body_ids))[0]
        if geom_ids.size == 0:
            print("Warning: no object geoms found; transparency not applied")
            return
        for gid in geom_ids:
            self.model.geom_rgba[gid, 3] = alpha
            matid = int(self.model.geom_matid[gid])
            if matid >= 0:
                self.model.mat_rgba[matid, 3] = alpha
        print(f"Applied object transparency: alpha={alpha:.2f} to {geom_ids.size} geoms")

    def _init_viz_sites(self):
        """Cache MJCF site ids for point visualization markers."""
        if not self._use_mjcf_markers:
            self._viz_green_site_ids = None
            self._viz_blue_site_ids = None
            self._viz_obj_goal_site_id = None
            self._viz_obj_goal_decoupled_site_id = None
            self._viz_human_goal_site_id = None
            return
        count = max(0, int(self.point_viz_sites))
        green_ids = []
        blue_ids = []
        for i in range(count):
            green_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"viz_point_g_{i}"))
            blue_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"viz_point_b_{i}"))
        green_ids = np.array([i for i in green_ids if i >= 0], dtype=np.int32)
        blue_ids = np.array([i for i in blue_ids if i >= 0], dtype=np.int32)
        self._viz_green_site_ids = green_ids
        self._viz_blue_site_ids = blue_ids
        self._viz_obj_goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "viz_goal_obj")
        if self._viz_obj_goal_site_id < 0:
            self._viz_obj_goal_site_id = None
        self._viz_obj_goal_decoupled_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "viz_goal_obj_decoupled"
        )
        if self._viz_obj_goal_decoupled_site_id < 0:
            self._viz_obj_goal_decoupled_site_id = None
        self._viz_human_goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "viz_goal_human")
        if self._viz_human_goal_site_id < 0:
            self._viz_human_goal_site_id = None

        site_size = max(self.point_viz_site_size, 1e-4)
        for sid in green_ids:
            self.model.site_size[sid][0] = site_size
            self.model.site_rgba[sid] = np.array([0.2, 0.9, 0.2, 1.0], dtype=np.float32)
        for sid in blue_ids:
            self.model.site_size[sid][0] = site_size
            self.model.site_rgba[sid] = np.array([0.2, 0.4, 0.95, 1.0], dtype=np.float32)
        if self._viz_obj_goal_site_id is not None:
            self.model.site_size[self._viz_obj_goal_site_id][0] = site_size * 2.0
            self.model.site_rgba[self._viz_obj_goal_site_id] = np.array([0.9, 0.6, 0.1, 1.0], dtype=np.float32)
        if self._viz_obj_goal_decoupled_site_id is not None:
            self.model.site_size[self._viz_obj_goal_decoupled_site_id][0] = site_size * 2.0
            self.model.site_rgba[self._viz_obj_goal_decoupled_site_id] = np.array([0.1, 0.9, 0.9, 1.0], dtype=np.float32)
        if self._viz_human_goal_site_id is not None:
            self.model.site_size[self._viz_human_goal_site_id][0] = site_size * 2.0
            self.model.site_rgba[self._viz_human_goal_site_id] = np.array([0.9, 0.1, 0.9, 1.0], dtype=np.float32)

        self._viz_site_hide_pos = np.array([0.0, 0.0, -1000.0], dtype=np.float32)
        self._viz_site_identity_mat = np.eye(3, dtype=np.float32).reshape(9)

    def _update_site_positions(self, site_ids, points):
        if site_ids is None or len(site_ids) == 0:
            return
        count = len(site_ids)
        if points is None:
            self.model.site_pos[site_ids] = self._viz_site_hide_pos
            self.data.site_xpos[site_ids] = self._viz_site_hide_pos
            self.data.site_xmat[site_ids] = self._viz_site_identity_mat
            return
        pts = np.asarray(points, dtype=np.float32)
        n = min(count, pts.shape[0])
        if n > 0:
            self.model.site_pos[site_ids[:n]] = pts[:n]
            self.data.site_xpos[site_ids[:n]] = pts[:n]
            self.data.site_xmat[site_ids[:n]] = self._viz_site_identity_mat
        if n < count:
            self.model.site_pos[site_ids[n:]] = self._viz_site_hide_pos
            self.data.site_xpos[site_ids[n:]] = self._viz_site_hide_pos
            self.data.site_xmat[site_ids[n:]] = self._viz_site_identity_mat

    def _update_viz_sites(self):
        """Update MJCF marker sites with latest point cloud positions."""
        if not self._use_mjcf_markers:
            return
        self._update_site_positions(self._viz_green_site_ids, self._debug_points_green)
        self._update_site_positions(self._viz_blue_site_ids, self._debug_points_blue)
        if self._viz_obj_goal_site_id is not None:
            obj_goal = self._debug_obj_goal_pos if self._debug_obj_goal_enabled else None
            if obj_goal is None:
                self.model.site_pos[self._viz_obj_goal_site_id] = self._viz_site_hide_pos
                self.data.site_xpos[self._viz_obj_goal_site_id] = self._viz_site_hide_pos
                self.data.site_xmat[self._viz_obj_goal_site_id] = self._viz_site_identity_mat
            else:
                self.model.site_pos[self._viz_obj_goal_site_id] = obj_goal
                self.data.site_xpos[self._viz_obj_goal_site_id] = obj_goal
                self.data.site_xmat[self._viz_obj_goal_site_id] = self._viz_site_identity_mat
        if self._viz_obj_goal_decoupled_site_id is not None:
            obj_goal_decoupled = (
                self._debug_obj_goal_decoupled_pos if self._debug_obj_goal_decoupled_enabled else None
            )
            if obj_goal_decoupled is None:
                self.model.site_pos[self._viz_obj_goal_decoupled_site_id] = self._viz_site_hide_pos
                self.data.site_xpos[self._viz_obj_goal_decoupled_site_id] = self._viz_site_hide_pos
                self.data.site_xmat[self._viz_obj_goal_decoupled_site_id] = self._viz_site_identity_mat
            else:
                self.model.site_pos[self._viz_obj_goal_decoupled_site_id] = obj_goal_decoupled
                self.data.site_xpos[self._viz_obj_goal_decoupled_site_id] = obj_goal_decoupled
                self.data.site_xmat[self._viz_obj_goal_decoupled_site_id] = self._viz_site_identity_mat
        if self._viz_human_goal_site_id is not None:
            human_goal = self._debug_human_goal_pos if self._debug_human_goal_enabled else None
            if human_goal is None:
                self.model.site_pos[self._viz_human_goal_site_id] = self._viz_site_hide_pos
                self.data.site_xpos[self._viz_human_goal_site_id] = self._viz_site_hide_pos
                self.data.site_xmat[self._viz_human_goal_site_id] = self._viz_site_identity_mat
            else:
                self.model.site_pos[self._viz_human_goal_site_id] = human_goal
                self.data.site_xpos[self._viz_human_goal_site_id] = human_goal
                self.data.site_xmat[self._viz_human_goal_site_id] = self._viz_site_identity_mat

    def _apply_first_person_camera(self, scene, cam_pos, cam_rot):
        """Override scene camera to match a world-space pose."""
        cam_forward = _quat_rotate_np(cam_rot, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        cam_up = _quat_rotate_np(cam_rot, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        cam_forward, _ = _normalize_np(cam_forward)
        cam_up = cam_up - cam_forward * float(np.dot(cam_up, cam_forward))
        cam_up, _ = _normalize_np(cam_up)
        for i in range(2):
            glcam = scene.camera[i]
            glcam.pos[:] = cam_pos
            glcam.forward[:] = cam_forward
            glcam.up[:] = cam_up

    def visualize_point_clouds(self, scene):
        """
        Visualize point clouds by adding custom geometries to MuJoCo scene.
        - GREEN: Randomized points (what the policy sees)
        - BLUE: Clean visible points (before randomization)
        - RED: Camera position
        - YELLOW: PCA bounding box corners
        - CYAN: Decoupled object goal (vertical-first target)

        Args:
            scene: MjvScene object to add geometries to
        """
        if not self._visualize_points:
            return

        # Helper function to add a sphere geom
        def add_sphere(position, radius, rgba, scale=3.0):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            scene.ngeom += 1

            # Initialize geometry properly
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).reshape(9),
                rgba.astype(np.float32)
            )

            # Mark as decor so MuJoCo treats it as a user geom (not tied to model objects).
            geom.segid = -1
            geom.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
            geom.objid = -1

            # Set size (scaled for visibility)
            geom.size[0] = radius * scale
            geom.size[1] = 0
            geom.size[2] = 0

            # Set position
            geom.pos[0] = float(position[0])
            geom.pos[1] = float(position[1])
            geom.pos[2] = float(position[2])

            # CRITICAL: Set category to mjCAT_DECOR so it gets rendered
            # This is required for custom geometries to be visible
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR

            # Set material properties for better visibility
            geom.emission = 0.5  # Make them glow a bit
            geom.reflectance = 0.5

        geom_count_start = scene.ngeom

        # Draw GREEN spheres: Randomized/noisy points (what policy sees)
        if self._debug_points_green is not None and len(self._debug_points_green) > 0:
            for point in self._debug_points_green:
                add_sphere(point, 0.015, np.array([0.2, 0.9, 0.2, 1.0]), scale=self.point_viz_scale)

        # Draw BLUE spheres: Clean visible points (before randomization)
        if self._debug_points_blue is not None and len(self._debug_points_blue) > 0:
            for point in self._debug_points_blue:
                add_sphere(point, 0.012, np.array([0.2, 0.4, 0.95, 1.0]), scale=self.point_viz_scale)

        # Draw RED sphere: Camera position
        if self._debug_camera_pos is not None:
            add_sphere(self._debug_camera_pos, 0.03, np.array([0.9, 0.2, 0.2, 1.0]))

        # Draw YELLOW spheres: PCA bounding box corners
        if self._debug_pca_corners is not None and len(self._debug_pca_corners) > 0:
            for corner in self._debug_pca_corners:
                add_sphere(corner, 0.02, np.array([0.9, 0.9, 0.2, 1.0]))

        # Draw goal markers (fallback when not using MJCF sites)
        if self._debug_obj_goal_enabled and self._debug_obj_goal_pos is not None:
            add_sphere(self._debug_obj_goal_pos, 0.03, np.array([0.9, 0.6, 0.1, 1.0]))
        if self._debug_obj_goal_decoupled_enabled and self._debug_obj_goal_decoupled_pos is not None:
            add_sphere(self._debug_obj_goal_decoupled_pos, 0.03, np.array([0.1, 0.9, 0.9, 1.0]))
        if self._debug_human_goal_enabled and self._debug_human_goal_pos is not None:
            add_sphere(self._debug_human_goal_pos, 0.03, np.array([0.9, 0.1, 0.9, 1.0]))


        # Ensure the new geoms are included in the render order list.
        if scene.geomorder is not None and scene.ngeom > geom_count_start:
            try:
                scene.geomorder[geom_count_start:scene.ngeom] = np.arange(
                    geom_count_start, scene.ngeom, dtype=np.int32
                )
            except Exception:
                for idx in range(geom_count_start, scene.ngeom):
                    scene.geomorder[idx] = idx

        # Debug output on first call
        if not hasattr(self, '_viz_debug_printed'):
            geom_count_end = scene.ngeom
            print(f"\n=== Point Cloud Visualization Enabled ===")
            print(f"Added {geom_count_end - geom_count_start} geometries to scene")
            print(f"Scene total geoms: {geom_count_end} / {scene.maxgeom}")
            if self._debug_points_green is not None and len(self._debug_points_green) > 0:
                print(f"Sample green point locations (first 5):")
                for i in range(min(5, len(self._debug_points_green))):
                    print(f"  Point {i}: {self._debug_points_green[i]}")
                print(f"Green points range:")
                print(f"  X: [{self._debug_points_green[:, 0].min():.3f}, {self._debug_points_green[:, 0].max():.3f}]")
                print(f"  Y: [{self._debug_points_green[:, 1].min():.3f}, {self._debug_points_green[:, 1].max():.3f}]")
                print(f"  Z: [{self._debug_points_green[:, 2].min():.3f}, {self._debug_points_green[:, 2].max():.3f}]")
            if self._debug_camera_pos is not None:
                print(f"Camera position: {self._debug_camera_pos}")
            print(f"  - Green: Randomized points (what policy sees)")
            print(f"  - Blue: Clean visible points")
            print(f"  - Red: Camera position")
            print(f"  - Yellow: PCA bounding box corners")
            print(f"  - Cyan: Decoupled object goal (vertical-first target)")
            print(f"=========================================\n")
            self._viz_debug_printed = True

    def set_commands(self):
        """Initialize command goal: [progress_end, approaching, leaving] (+ time_to_go)."""
        self.commands.fill(0.0)
        if self.commands.shape[0] >= 3:
            self.commands[0] = 0.0  # progress_end
            self.commands[1] = 0.0  # approaching
            self.commands[2] = 0.0  # leaving
        
    def extract_data(self):
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        quat = self.data.sensor('orientation').data.astype(np.float32)
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        self.dof_vel = torch.from_numpy(dof_vel).float().unsqueeze(0).to(self.device)
        return (dof_pos, dof_vel, quat, ang_vel)

    # -------- main reset (MuJoCo equivalent of your Isaac reset) --------
    def reset_human_and_object_from_ref(self, motion_t=0,
                                            human_freejoint_name="pelvis",
                                            clamp_to_range=False):
            """
            Sets:
            human root pose/vel from hoi_refs[..., 0:13]
            human hinge joints qpos/qvel from hoi_refs[..., 13:42] and 42:71
            object pose/vel from hoi_refs[..., 71:84] (+ noise on pos/rot like your Isaac _reset_target)
            """
            noise = _DEFAULT_NOISE_STD
            object_freejoint_name=f"{self.object_name}_free"
            # 1) Reset to keyframe 'home' (keeps limbs in sane default) or full reset
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if key_id >= 0: mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            else:           mujoco.mj_resetData(self.model, self.data)
            # 2) Slice references (torch on the right device)
            ref = self.hoi_ref[motion_t].clone()
            if not isinstance(ref, torch.Tensor):
                ref = torch.as_tensor(ref, dtype=torch.float32, device=self.device)
            else:
                ref = ref.to(self.device, torch.float32)
            root_pos      = ref[0:3]
            root_rot_xyzw = ref[3:7]
            root_vel      = ref[7:10].clone() if self.match_training_reset else torch.zeros_like(ref[7:10])
            root_ang_vel  = ref[10:13].clone() if self.match_training_reset else torch.zeros_like(ref[10:13])
            dof_pos       = ref[13:42]    # 29
            dof_vel       = ref[42:71].clone() if self.match_training_reset else torch.zeros_like(ref[42:71])
            root_pos_new = _randn_like(root_pos, noise["root_pos"])
            if not self.match_training_reset:
                root_pos_new[..., 2:3] = root_pos[..., 2:3] + 0.01
            root_pos = root_pos_new
            root_rot_xyzw = _perturb_quat(root_rot_xyzw, noise["root_rot"])
            root_vel = _randn_like(root_vel, noise["root_vel"])
            root_ang_vel = _randn_like(root_ang_vel, noise["root_ang_vel"])
            dof_pos = _randn_like(dof_pos, noise["dof_pos"])
            dof_vel = _randn_like(dof_vel, noise["dof_vel"])
            # dof_pos = torch.clamp(dof_pos, min=self.dof_limits_lower, max=self.dof_limits_upper)
            obj_pos       = ref[71:74]
            obj_rot_xyzw  = ref[74:78]
            obj_pos_new  = _randn_like(obj_pos, noise["target_pos"])
            if self.match_training_reset:
                obj_pos_new[..., 2:3] = obj_pos[..., 2:3] + (0.1 if motion_t == 0 else 0.01)
            obj_rot_xyzw  = _perturb_quat(obj_rot_xyzw, noise["target_rot"])
            obj_pos = obj_pos_new
            obj_lin       = ref[78:81].clone() if self.match_training_reset else torch.zeros_like(ref[78:81])
            obj_ang       = ref[81:84].clone() if self.match_training_reset else torch.zeros_like(ref[81:84])
            # 3) Human: set pelvis freejoint, then all hinge joints
            _set_freejoint_state(self.model, self.data, human_freejoint_name,
                                root_pos, root_rot_xyzw, root_vel, root_ang_vel)
            _write_hinge_qpos_qvel_by_names(self.model, self.data, JOINT_NAMES_29, dof_pos, dof_vel,
                                            clamp_to_range=clamp_to_range)
            _set_freejoint_state(self.model, self.data, object_freejoint_name,
                                obj_pos, obj_rot_xyzw, obj_lin, obj_ang)
            # 5) Finalize
            mujoco.mj_forward(self.model, self.data)
        
    def run(self):
        cnt = 0
        self.reset_human_and_object_from_ref()
        
        desired_rtf = getattr(self, "desired_rtf", 1)      # 1.0 = real-time, 0.5 = half-speed (slow motion), 2.0 = 2x speed
        max_render_fps = getattr(self, "max_render_fps", 60)  # limit onscreen draw rate
        start_wall = time.perf_counter()
        last_render = start_wall
        min_render_dt = 1.0 / max_render_fps
        if self.record_video:
            import imageio
            video_name = f"{self.robot_type}_{''.join(os.path.basename(self.policy_path).split('.')[:-1])}.mp4"
            path = f"./logs/mujoco_videos/"
            if not os.path.exists(path):
                os.makedirs(path)
            video_name = os.path.join(path, video_name)
            # mp4_writer = imageio.get_writer(video_name)
            mp4_writer = imageio.get_writer(
                "out.mp4",
                format="FFMPEG",
                mode="I",
                codec="libx264",
                fps=60,                  # pick your sim/video rate
                pixelformat="yuv420p",   # maximum compatibility
                quality=8,
                macro_block_size=None    # keep your resolution, don’t auto-resize
            )

        self.last_action = torch.zeros((1, 29))
        self.last_torque = torch.zeros((1, 29))
        for i in tqdm(range(int(self.sim_duration / self.sim_dt)), desc="Running simulation..."):
            dof_pos, dof_vel = _gather_dof_slices(
                self.data, self.obs_builder.dof_qpos_ids, self.obs_builder.dof_qvel_ids
            )
            dof_pos = dof_pos.squeeze(0).numpy()
            dof_vel = dof_vel.squeeze(0).numpy()
            if i % self.sim_decimation == 0:
                timestep = i // self.sim_decimation
                if timestep >= self.hoi_data.shape[0] - 1:
                    break
                # print(timestep)
                obs_tensor, obs_dict = self.obs_builder._compute_observations(self.data, timestep, self.last_action, self.last_torque, student_obs=True, episode_length=timestep, return_dict=True)
                obs_tensor = obs_tensor.to(self.device)

                # Extract visualization data from obs_dict
                if self._visualize_points:
                    if 'viz_points_clean' in obs_dict:
                        self._debug_points_blue = obs_dict['viz_points_clean'].squeeze(0).detach().cpu().numpy()
                    if 'viz_points_randomized' in obs_dict:
                        self._debug_points_green = obs_dict['viz_points_randomized'].squeeze(0).detach().cpu().numpy()
                    if 'viz_camera_pos' in obs_dict:
                        self._debug_camera_pos = obs_dict['viz_camera_pos'].squeeze(0).detach().cpu().numpy()
                    if 'viz_camera_rot' in obs_dict:
                        self._debug_camera_rot = obs_dict['viz_camera_rot'].squeeze(0).detach().cpu().numpy()
                    if 'viz_pca_corners' in obs_dict:
                        self._debug_pca_corners = obs_dict['viz_pca_corners'].detach().cpu().numpy()
                    if 'viz_obj_goal_pos' in obs_dict:
                        self._debug_obj_goal_pos = obs_dict['viz_obj_goal_pos'].squeeze(0).detach().cpu().numpy()
                    if 'viz_obj_goal_enabled' in obs_dict:
                        self._debug_obj_goal_enabled = bool(obs_dict['viz_obj_goal_enabled'].item())
                    if 'viz_obj_goal_decoupled_pos' in obs_dict:
                        self._debug_obj_goal_decoupled_pos = obs_dict['viz_obj_goal_decoupled_pos'].squeeze(0).detach().cpu().numpy()
                    if 'viz_obj_goal_decoupled_enabled' in obs_dict:
                        self._debug_obj_goal_decoupled_enabled = bool(obs_dict['viz_obj_goal_decoupled_enabled'].item())
                    if 'viz_human_goal_pos' in obs_dict:
                        self._debug_human_goal_pos = obs_dict['viz_human_goal_pos'].squeeze(0).detach().cpu().numpy()
                    if 'viz_human_goal_enabled' in obs_dict:
                        self._debug_human_goal_enabled = bool(obs_dict['viz_human_goal_enabled'].item())

                    if self._debug_points_green is not None and len(self._debug_points_green) > 0:
                        self._debug_points_center = np.mean(self._debug_points_green, axis=0)
                    elif self._debug_points_blue is not None and len(self._debug_points_blue) > 0:
                        self._debug_points_center = np.mean(self._debug_points_blue, axis=0)
                    elif self._debug_pca_corners is not None and len(self._debug_pca_corners) > 0:
                        self._debug_points_center = np.mean(self._debug_pca_corners, axis=0)
                    else:
                        self._debug_points_center = None

                    # Debug: Print visualization data status on first frame
                    if timestep == 0:
                        print(f"\n=== Point Cloud Visualization Debug ===")
                        print(f"Blue points (clean): {self._debug_points_blue.shape if self._debug_points_blue is not None else None}")
                        print(f"Green points (randomized): {self._debug_points_green.shape if self._debug_points_green is not None else None}")
                        print(f"Camera pos: {self._debug_camera_pos.shape if self._debug_camera_pos is not None else None}")
                        print(f"Camera rot: {self._debug_camera_rot.shape if self._debug_camera_rot is not None else None}")
                        print(f"PCA corners: {self._debug_pca_corners.shape if self._debug_pca_corners is not None else None}")
                        if self._debug_points_green is not None:
                            print(f"Sample green point: {self._debug_points_green[0]}")
                        if self._debug_camera_pos is not None:
                            print(f"Camera position: {self._debug_camera_pos}")
                        if self._debug_obj_goal_decoupled_pos is not None:
                            print(f"Decoupled object goal: {self._debug_obj_goal_decoupled_pos}")
                        print(f"======================================\n")
                with torch.no_grad():
                    if self.use_jit:
                        # JIT model: pass raw observation directly (normalization is built-in)
                        teacher_action = self.policy_jit(obs_tensor).squeeze(0)  # (29)
                        raw_action = teacher_action.cpu().numpy()
                    else:
                        self.policy.eval()
                        vae_noise = torch.zeros((1, self.vae_dim), device=self.device, dtype=obs_tensor.dtype)
                        input_dict = {
                            'is_train': False,
                            'prev_actions': None,
                            'obs': obs_tensor,
                            'rnn_states': None,
                            'with_vae': False,
                            'with_encoder': False,
                            'vae_noise': vae_noise,
                            'skip_critic': True,
                        }
                        res_dict = self.policy(input_dict)
                        teacher_action = torch.clamp(res_dict['mus'], min=-1.0, max=1.0).squeeze(0)  # (29)
                        raw_action = teacher_action.cpu().numpy()
                
                self.last_action = to_torch(raw_action).unsqueeze(0)
                # raw_action = np.clip(raw_action, -10, 10)
                # scaled_actions = raw_action * self.action_scale
                scaled_actions = raw_action * 3.0
                step_actions = np.zeros(self.num_dofs)
                step_actions[self.control_indices] = scaled_actions
                # step_actions[[0,6]] /= 1.1
                # step_actions[[1,7]] /= 1.1176
                # step_actions[[3,9]] -= 0.5
                
                pd_target = step_actions + self.default_dof_pos * 0.
                # print("Actions: "   , pd_target)
                if self.record_video:
                    # Update camera for rendering
                    cam_pos = None
                    cam_rot = None
                    if self.render_camera == "head":
                        cam_pos = self._debug_camera_pos
                        cam_rot = self._debug_camera_rot
                        if cam_pos is not None and cam_rot is not None:
                            cam_forward = _quat_rotate_np(
                                cam_rot, np.array([1.0, 0.0, 0.0], dtype=np.float32)
                            )
                            cam_forward, _ = _normalize_np(cam_forward)
                            back_dir = -cam_forward
                            back_xy = float(np.linalg.norm(back_dir[:2]))
                            azim = np.degrees(np.arctan2(back_dir[1], back_dir[0]))
                            elev = np.degrees(np.arctan2(back_dir[2], max(back_xy, 1e-6)))
                            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                            self.camera.lookat[:] = cam_pos + cam_forward
                            self.camera.distance = 1.0
                            self.camera.azimuth = azim
                            self.camera.elevation = elev
                        else:
                            cam_pos = None
                            cam_rot = None
                    elif self.render_camera == "points":
                        target = self._debug_points_center
                        if target is not None:
                            cam_pos = target + np.array([0.0, -2.0, 1.0], dtype=np.float32)
                            delta = cam_pos - target
                            dist = float(np.linalg.norm(delta))
                            if dist < 0.1:
                                dist = 0.1
                                delta, _ = _normalize_np(delta)
                            azim = np.degrees(np.arctan2(delta[1], delta[0]))
                            elev = np.degrees(np.arctan2(delta[2], max(np.linalg.norm(delta[:2]), 1e-6)))
                            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                            self.camera.lookat[:] = target
                            self.camera.distance = dist
                            self.camera.azimuth = azim
                            self.camera.elevation = elev
                        else:
                            cam_pos = None
                            cam_rot = None

                    if cam_pos is None and self.render_camera in ("head", "points"):
                        robot_pos = self.data.qpos[:3]
                        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        self.camera.lookat[:] = robot_pos
                    elif self.render_camera == "tracking":
                        # Update camera to follow robot (pelvis position)
                        robot_pos = self.data.qpos[:3]
                        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        self.camera.lookat[:] = robot_pos

                    # Update MJCF markers before scene build (so positions are captured)
                    if self._visualize_points and self._use_mjcf_markers:
                        self._update_viz_sites()

                    # Build scene with model geometry and physics
                    mujoco.mjv_updateScene(
                        self.model, self.data, self.viz_option,
                        None, self.camera,
                        mujoco.mjtCatBit.mjCAT_ALL,
                        self.viz_scene
                    )

                    # Add custom geometry markers (non-MJCF path)
                    if self._visualize_points and not self._use_mjcf_markers:
                        self.visualize_point_clouds(self.viz_scene)

                    # Override camera for first-person view
                    if (self.render_camera == "head"
                            and cam_pos is not None
                            and cam_rot is not None):
                        self._apply_first_person_camera(self.viz_scene, cam_pos, cam_rot)

                    # Render the scene with our offscreen context
                    mujoco.mjr_render(self.viewport, self.viz_scene, self.mjr_context)

                    # Read pixels from framebuffer
                    img = np.empty((self.viewport.height, self.viewport.width, 3), dtype=np.uint8)
                    mujoco.mjr_readPixels(img, None, self.viewport, self.mjr_context)
                    img = np.flipud(img)
                    img = np.asarray(img)
                    if img.dtype != np.uint8:
                        img = np.clip(img, 0, 255).astype(np.uint8) if img.max() > 1.0 else (img * 255).astype(np.uint8)
                    if img.ndim == 2:
                        img = np.repeat(img[..., None], 3, axis=2)
                    if img.shape[-1] == 4:
                        img = img[..., :3]

                    h, w = img.shape[:2]
                    if (h % 2) or (w % 2):
                        # pad to even dims for yuv420p
                        pad_h = h + (h % 2)
                        pad_w = w + (w % 2)
                        pad = ((0, pad_h - h), (0, pad_w - w), (0, 0))
                        img = np.pad(img, pad, mode="edge")

                    mp4_writer.append_data(np.ascontiguousarray(img))
                else:
                    # --- cap render FPS ---
                    self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]

                    # Draw point clouds inside the viewer
                    if self._visualize_points:
                        if self._use_mjcf_markers:
                            self._update_viz_sites()
                        else:
                            self.viewer.user_scn.ngeom = 0
                            self.visualize_point_clouds(self.viewer.user_scn)

                    now = time.perf_counter()
                    spend = now - last_render
                    if spend < min_render_dt:
                        time.sleep(min_render_dt - spend)
                    last_render = time.perf_counter()
                    self.viewer.sync()
                
            # pd_target[[13,14]] *= 0
            # print("RPY: ", rpy)
            torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
            cnt += np.sum(torque < -self.torque_limits) + np.sum(torque > self.torque_limits)
            # print("Torque: ", torque)
            # print("Torque limits exceeded: ", np.where(torque < -self.torque_limits), np.where(torque > self.torque_limits))
            # print("exceeded torques: ", torque[torque < -self.torque_limits], torque[torque > self.torque_limits])
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            # print(timestep, i, pd_target, dof_pos, dof_vel, torque)
            # if i > 2:
            #     break
            # print("Waist dof target: ", pd_target[[12, 13,14]])
            # print("ACTIONS: ", step_actions[[12,13,14]])
            # print("PD_TARGET: ", (pd_target))
            # print("DOF_POS: ", dof_pos)
            # print("Torque: ", torque)
            # print("Dof vel: ", dof_vel)
            # print("Overall cnt: ", cnt)
            self.data.ctrl = torque
            self.last_torque = to_torch(torque).unsqueeze(0)
            mujoco.mj_step(self.model, self.data)
            if not self.record_video:
                target_wall = start_wall + (self.data.time / max(desired_rtf, 1e-6))
                sleep_s = target_wall - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
        
        if not self.record_video:
            self.viewer.close()
        else:
            mp4_writer.close()
            self.renderer.close()
                 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MuJoCo sim-to-sim deployment with student policy")

    parser.add_argument('--robot', type=str, default="g1_29dof",
                        help="Robot type (default: g1_29dof)")
    parser.add_argument('--ckpt', type=str, required=True,
                        help="Path to checkpoint (.pth) or JIT model (.pt)")
    parser.add_argument('--motion_path', type=str, required=True,
                        help="Path to motion data file")
    parser.add_argument('--record_video', action='store_true',
                        help="Record video of the simulation")
    parser.add_argument('--render_camera', type=str, default="tracking",
                        choices=["tracking", "head", "points"],
                        help="Camera mode for recorded videos")
    parser.add_argument('--object_alpha', type=float, default=0.8,
                        help="Transparency for object geoms (1.0 = opaque)")
    parser.add_argument('--point_viz_scale', type=float, default=8.0,
                        help="Scale factor for point cloud sphere size")
    parser.add_argument('--point_viz_offset', type=float, default=0.0,
                        help="(Deprecated) Offset points toward camera (meters)")
    parser.add_argument('--point_viz_sites', type=int, default=64,
                        help="Number of MJCF site markers to allocate for point visualization (0 disables)")
    parser.add_argument('--point_viz_site_size', type=float, default=0.02,
                        help="Radius for MJCF point marker sites (meters)")
    parser.add_argument('--use_jit', action='store_true',
                        help="Use JIT-compiled model (.pt) instead of checkpoint (.pth)")
    parser.add_argument('--task_mode', type=str, default=None,
                        choices=["full_track", "sparse_track", "object_obs"],
                        help="Preset observation masking mode for sim2sim")
    parser.add_argument('--obj_obs', type=str, default="points",
                        choices=["pos", "points", "none"],
                        help="Object observation source for object_obs mode")
    parser.add_argument('--obs_obj_rot_keep_prob', type=float, default=1.0,
                        help="Keep probability for object rotation observations")
    parser.add_argument('--obs_obj_trans_keep_prob', type=float, default=1.0,
                        help="Keep probability for object translation observations")
    parser.add_argument('--obs_obj_point_keep_prob', type=float, default=1.0,
                        help="Keep probability for object point observations")
    parser.add_argument('--obs_obj_pos_keep_prob', type=float, default=1.0,
                        help="Keep probability for object position observations")
    parser.add_argument('--obs_human_move_keep_prob', type=float, default=1.0,
                        help="Keep probability for human movement observations")
    parser.add_argument('--obs_human_global_keep_prob', type=float, default=None,
                        help="Keep probability for human global goal observations")
    parser.add_argument('--obs_human_local_keep_prob', type=float, default=None,
                        help="Keep probability for human local goal observations")
    parser.add_argument('--obs_human_goal_keep_prob', type=float, default=1.0,
                        help="Keep probability for command goal observations")
    parser.add_argument('--obs_mask_flip_prob', type=float, default=0.0,
                        help="Probability of flipping a keep mask per step")

    # Point cloud domain randomization parameters
    parser.add_argument('--point_noise_std', type=float, default=0.02,
                        help="Standard deviation of Gaussian noise added to points")
    parser.add_argument('--point_dropout_prob', type=float, default=0.15,
                        help="Probability of dropping out individual points")
    parser.add_argument('--point_outlier_prob', type=float, default=0.05,
                        help="Probability of injecting outlier points")
    parser.add_argument('--point_outlier_scale', type=float, default=0.5,
                        help="Scale of outlier displacement (meters)")
    parser.add_argument('--point_depth_noise_scale', type=float, default=0.01,
                        help="Scale of depth-dependent noise")
    parser.add_argument('--point_density_min', type=float, default=0.5,
                        help="Minimum density factor for point sampling")
    parser.add_argument('--point_density_max', type=float, default=1.0,
                        help="Maximum density factor for point sampling")
    parser.add_argument('--point_cluster_noise_std', type=float, default=0.005,
                        help="Standard deviation of cluster-based noise")
    parser.add_argument('--point_scale_min', type=float, default=0.95,
                        help="Minimum scale factor for point cloud")
    parser.add_argument('--point_scale_max', type=float, default=1.05,
                        help="Maximum scale factor for point cloud")
    parser.add_argument('--point_translation_noise', type=float, default=0.02,
                        help="Standard deviation of random translation noise")
    parser.add_argument('--point_occlusion_prob', type=float, default=0.1,
                        help="Probability of random occlusion")
    parser.add_argument('--camera_rot_noise', type=float, default=0.05,
                        help="Camera rotation noise (radians)")
    parser.add_argument('--camera_pos_noise', type=float, default=0.02,
                        help="Camera position noise (meters)")

    # Fixed grid sampling for point cloud (mimics depth camera with fixed pixels)
    parser.add_argument('--no_point_fixed_grid_sampling', action='store_true',
                        help="Disable fixed UV grid for point sampling (default: enabled)")
    parser.add_argument('--point_surface_grid_resolution', type=int, default=None,
                        help="Grid resolution for fixed surface sampling (default: auto-compute)")

    # Goal achievement checking for sparse tracking
    parser.add_argument('--goal_achievement_enabled', action='store_true',
                        help="Enable goal achievement checking for sparse tracking (only progress when goal is reached)")
    parser.add_argument('--goal_pos_threshold', type=float, default=0.2,
                        help="Position threshold (meters) for goal achievement")
    parser.add_argument('--goal_phase_dim', type=int, default=4, choices=[3, 4],
                        help="Command goal dimension: 4 (default, matches numObsStudent=1496 with time-to-go) or 3 (legacy 1494-d checkpoints)")
    parser.add_argument('--obj_goal_decouple', type=str, default="off",
                        choices=["off", "hard", "soft"],
                        help="Decouple object translation goal: vertical first then horizontal (sim2sim only)")
    parser.add_argument('--obj_goal_z_threshold', type=float, default=0.15,
                        help="Z threshold (meters) for object goal decoupling")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect JIT based on file extension if not explicitly specified
    if not args.use_jit and args.ckpt.endswith('.pt'):
        print("Detected .pt file extension, automatically enabling --use_jit")
        args.use_jit = True

    env = HumanoidEnv(
        policy_path=args.ckpt,
        robot_type=args.robot,
        device=device,
        record_video=args.record_video,
        motion_path=args.motion_path,
        use_jit=args.use_jit,
        obj_rot_keep_prob=args.obs_obj_rot_keep_prob,
        obj_trans_keep_prob=args.obs_obj_trans_keep_prob,
        obj_point_keep_prob=args.obs_obj_point_keep_prob,
        obj_pos_keep_prob=args.obs_obj_pos_keep_prob,
        human_move_keep_prob=args.obs_human_move_keep_prob,
        human_global_keep_prob=args.obs_human_global_keep_prob,
        human_local_keep_prob=args.obs_human_local_keep_prob,
        human_goal_keep_prob=args.obs_human_goal_keep_prob,
        mask_flip_prob=args.obs_mask_flip_prob,
        point_noise_std=args.point_noise_std,
        point_dropout_prob=args.point_dropout_prob,
        point_outlier_prob=args.point_outlier_prob,
        point_outlier_scale=args.point_outlier_scale,
        point_depth_noise_scale=args.point_depth_noise_scale,
        point_density_min=args.point_density_min,
        point_density_max=args.point_density_max,
        point_cluster_noise_std=args.point_cluster_noise_std,
        point_scale_min=args.point_scale_min,
        point_scale_max=args.point_scale_max,
        point_translation_noise=args.point_translation_noise,
        point_occlusion_prob=args.point_occlusion_prob,
        camera_rot_noise=args.camera_rot_noise,
        camera_pos_noise=args.camera_pos_noise,
        point_fixed_grid_sampling=not args.no_point_fixed_grid_sampling,
        point_surface_grid_resolution=args.point_surface_grid_resolution,
        render_camera=args.render_camera,
        object_alpha=args.object_alpha,
        point_viz_scale=args.point_viz_scale,
        point_viz_offset=args.point_viz_offset,
        point_viz_sites=args.point_viz_sites,
        point_viz_site_size=args.point_viz_site_size,
        goal_achievement_enabled=args.goal_achievement_enabled,
        goal_pos_threshold=args.goal_pos_threshold,
        obj_goal_decouple=args.obj_goal_decouple,
        obj_goal_z_threshold=args.obj_goal_z_threshold,
        goal_phase_dim=args.goal_phase_dim,
    )
    if args.task_mode is not None:
        env.obs_builder.configure_task_mode(args.task_mode, obj_obs=args.obj_obs)
    env.run()
