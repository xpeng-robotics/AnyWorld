# SPDX-License-Identifier: Apache-2.0
# Modified for the AnyWorld public code release.
import os
import cv2
import sys
import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn

from scipy.spatial.transform import Rotation as R
from typing import List, Callable
from typing_extensions import Literal

# Add parent directory to path to import VITRA modules
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# from libs.models.mano_wrapper import MANO
# from visualization.render_utils import Renderer
# from visualization.video_utils import save_to_video


def euler_traj_to_rotmat_traj(euler_traj, T):
    """Convert Euler angle trajectory to rotation matrix trajectory."""
    hand_pose = euler_traj.reshape(-1, 3)
    pose_matrices = R.from_euler('xyz', hand_pose).as_matrix()
    pose_matrices = pose_matrices.reshape(T, 15, 3, 3)
    return pose_matrices


def state_to_robotic_hand_angles(mano_hand_pose):
    # TODO: Implement conversion if needed
    pass


def state_to_mano_vertices(state, beta, mano, is_left=False):
    """
    Convert hand state to MANO vertices and joints.
    
    Returns:
        verts_worldspace: [778, 3] mesh vertices in world space
        joints_worldspace: [21, 3] joint positions in world space
        global_orient_worldspace: [3, 3] wrist rotation matrix
    """
    transl_worldspace = state[0:3].reshape(1, 3)
    global_orient_euler = state[3:6].reshape(1, 3)
    hand_pose_euler = state[6:51].reshape(1, 45)
    
    global_orient_worldspace = R.from_euler('xyz', global_orient_euler).as_matrix()
    hand_pose = euler_traj_to_rotmat_traj(hand_pose_euler, T=1)
    
    hand_labels = {
        'transl_worldspace': transl_worldspace,
        'global_orient_worldspace': global_orient_worldspace,
        'hand_pose': hand_pose,
        'beta': beta,
    }
    
    T = 1
    hand_mask = np.ones(T, dtype=bool)
    
    wrist_worldspace = hand_labels['transl_worldspace'].reshape(-1, 1, 3)
    wrist_orientation = hand_labels['global_orient_worldspace']
    pose = hand_labels['hand_pose']
    
    beta_torch = torch.from_numpy(beta).float().cuda().unsqueeze(0).repeat(T, 1)
    pose_torch = torch.from_numpy(pose).float().cuda()
    
    global_rot_placeholder = torch.eye(3).float().unsqueeze(0).unsqueeze(0).cuda().repeat(T, 1, 1, 1)
    mano_out = mano(betas=beta_torch, hand_pose=pose_torch, global_orient=global_rot_placeholder)
    
    verts = mano_out.vertices.cpu().numpy()
    joints = mano_out.joints.cpu().numpy()
    
    if is_left:
        verts[:, :, 0] *= -1
        joints[:, :, 0] *= -1
    
    verts_worldspace = (
        wrist_orientation @ 
        (verts - joints[:, 0][:, None]).transpose(0, 2, 1)
    ).transpose(0, 2, 1) + wrist_worldspace
    
    joints_worldspace = (
        wrist_orientation @ 
        (joints - joints[:, 0][:, None]).transpose(0, 2, 1)
    ).transpose(0, 2, 1) + wrist_worldspace
    
    return verts_worldspace[0], joints_worldspace[0], global_orient_worldspace[0], mano.faces
    

def generate_hand_states(
    finger: Literal["thumb", "index", "middle", "ring", "pinky"],
    joint: Literal[0, 1, 2],
    axis: Literal["x", "y", "z"],
    length: int,
    use_left_hand: bool = True,
    speed: float = 1/54,
    origin=[0] * 212
):
    finger_map = {
        'ring': 9,   # 拇指从关节10开始 (3个关节: 10, 11, 12)
        'index': 0,    # 食指从关节1开始 (3个关节: 1, 2, 3)
        'middle': 3,   # 中指从关节4开始 (3个关节: 4, 5, 6)
        'pinky': 6,     # 无名指从关节7开始 (3个关节: 7, 8, 9)
        'thumb': 12,   # 小指从关节13开始 (3个关节: 13, 14, 但只有2个关节可用)
    }
    base_joint = finger_map.get(finger.lower(), 1)
    joint_num = base_joint + joint

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_offset = axis_map.get(axis.lower(), 1)
    
    idx = 6 + joint_num * 3 + axis_offset
    if not use_left_hand: idx += 51

    """Generate hand states for a given finger, joint, and axis."""
    states = [list(origin)]
    while len(states) < length:
        stat = states[-1].copy()
        stat[idx] += speed
        states.append(stat)
    return states


def render_frame(verts, faces, renderer, image_size=(480, 640),
                joints_worldspace=None, global_orient=None,
                draw_axes: bool = True):
    """
    Render a single frame and optionally overlay XYZ axes at joints.
    
    Args:
        verts: [778, 3] mesh vertices
        faces: [1538, 3] face indices
        renderer: Renderer instance
        image_size: (height, width)
        joints_worldspace: [21, 3] joint positions in world space
        global_orient: [3, 3] wrist rotation matrix
    """
    verts_torch = torch.from_numpy(verts).float().cuda()
    faces_torch = torch.from_numpy(faces).long().cuda()

    # Hand color (light blue)
    hand_color = np.array([0.4, 0.5, 0.75])
    colors = np.tile(hand_color, (verts.shape[0], 1))
    colors_torch = torch.from_numpy(colors).float().cuda()

    rend, mask = renderer.render(
        verts_list=[verts_torch],
        faces_list=[faces_torch],
        colors_list=[colors_torch]
    )

    return rend


# def render_animation(
#     mano_path: str, 
#     states: List[np.ndarray],
#     output_path: str,
#     fps: int = 30,
#     image_size: tuple = (480, 640),
#     focal_length: int = 500,
#     is_left_hand: bool = True,
#     beta: np.ndarray = np.zeros(10, dtype=np.float32)
# ) -> List[np.ndarray]:
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     mano = MANO(model_path=mano_path).to(device=device)

#     renderer = Renderer(
#         width=image_size[1],
#         height=image_size[0],
#         focal_length=focal_length,
#         device=device
#     )

#     frames = []        
#     for i, state in enumerate(states):
#         verts, joints, global_orient, faces = state_to_mano_vertices(
#             state, beta, mano, is_left=is_left_hand
#         )

#         frame = render_frame(
#             verts, faces, renderer, image_size,
#             joints_worldspace=joints,
#             global_orient=global_orient,
#         )
#         frames.append(frame)
    
#     os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
#     save_to_video(frames, output_path, fps=fps)

# Copied from VITRA data_utils.py
def read_dataset_statistics(statistics_path: str, default_value=1e-4) -> dict:
    """
    Read dataset statistics from a JSON file.
    Args:
        statistics_path: Path to the JSON file containing dataset statistics.
        default_value: Default value to use if statistics are missing.
    Returns:
        data_statistics: Dictionary with mean and std for state and action of both hands.
    """
    # Check if stats_path is a Hugging Face repo (format: "username/repo-name" or "username/repo-name:filename")
    assert os.path.exists(statistics_path) is True, f"Statistics file not found: {statistics_path}"
        
    # Load dataset statistics
    with open(statistics_path, 'r') as file:
        dataset_stats = json.load(file)
        
        # Assert that right hand statistics must exist
        assert 'state_right' in dataset_stats, "Right hand statistics must exist"
        
        # Get right hand statistics
        state_right_mean = np.array(dataset_stats['state_right']['mean'])
        state_right_std = np.array(dataset_stats['state_right']['std'])
        action_right_mean = np.array(dataset_stats['action_right']['mean'])
        action_right_std = np.array(dataset_stats['action_right']['std'])
        
        # For left hand, use right hand dimensions but fill with default_value if not available
        if 'state_left' in dataset_stats:
            state_left_mean = np.array(dataset_stats['state_left']['mean'])
            state_left_std = np.array(dataset_stats['state_left']['std'])
            action_left_mean = np.array(dataset_stats['action_left']['mean'])
            action_left_std = np.array(dataset_stats['action_left']['std'])
        else:
            state_left_mean = np.full_like(state_right_mean, default_value)
            state_left_std = np.full_like(state_right_std, default_value)
            action_left_mean = np.full_like(action_right_mean, default_value)
            action_left_std = np.full_like(action_right_std, default_value)
        
        data_statistics = {
            'state_right_mean': state_right_mean,
            'state_right_std': state_right_std,
            'action_right_mean': action_right_mean,
            'action_right_std': action_right_std,
            'state_left_mean': state_left_mean,
            'state_left_std': state_left_std,
            'action_left_mean': action_left_mean,
            'action_left_std': action_left_std,
        }
    return data_statistics


# Copied from VITRA data_utils.py
class GaussianNormalizer:
    """
    A class for normalizing and denormalizing state and action arrays.
    Assumes state/action numpy arrays are concatenated as [left, right].
    Accepts pre-loaded data_statistics dictionary.
    """
    def __init__(self, data_statistics: dict):
        """
        Args:
            data_statistics (dict): pre-loaded statistics dictionary with keys:
                'state_left_mean', 'state_left_std', 'state_right_mean', 'state_right_std',
                'action_left_mean', 'action_left_std', 'action_right_mean', 'action_right_std'
                All values are numpy arrays.
        """
        # Concatenate left and right statistics for vectorized operations
        self.state_mean = np.concatenate([data_statistics['state_left_mean'], data_statistics['state_right_mean']])
        self.state_std = np.concatenate([data_statistics['state_left_std'], data_statistics['state_right_std']])
        self.action_mean = np.concatenate([data_statistics['action_left_mean'], data_statistics['action_right_mean']])
        self.action_std = np.concatenate([data_statistics['action_left_std'], data_statistics['action_right_std']])

    # -----------------------------
    # State normalization
    # -----------------------------
    def normalize_state(self, state: np.ndarray, epsilon=1e-7) -> np.ndarray:
        return (state - self.state_mean) / (self.state_std + epsilon)

    def unnormalize_state(self, norm_state: np.ndarray, epsilon=1e-7) -> np.ndarray:
        return norm_state * (self.state_std + epsilon) + self.state_mean
    # -----------------------------
    # Action normalization
    # -----------------------------
    def normalize_action(self, action: np.ndarray, epsilon=1e-7) -> np.ndarray:
        return (action - self.action_mean) / (self.action_std + epsilon)

    def unnormalize_action(self, norm_action: np.ndarray, epsilon=1e-7) -> np.ndarray:
        return norm_action * (self.action_std + epsilon) + self.action_mean


class HandState(object):
    def __init__(self, entry):
        """
        Hand State结构说明（51+51维向量, 左右手）:
        ===========================================
        state[0:3]   - 手腕位置 (translation)
        [0]        - x轴位置（左右，正=右，单位：米）
        [1]        - y轴位置（上下，正=上，单位：米）
        [2]        - z轴位置（前后，正=前/远离相机，单位：米）

        state[3:6]   - 手腕旋转 (global_orient, Euler角)
        [3]        - x轴旋转（俯仰，pitch，单位：弧度）
        [4]        - y轴旋转（偏航，yaw，单位：弧度）
        [5]        - z轴旋转（翻滚，roll，单位：弧度）

        state[6:51]  - 手部关节角度 (hand_pose, 15个关节×3个Euler角)
        每个关节有3个Euler角: [x, y, z]
        关节索引计算: state[6 + joint_num * 3 + axis_offset]
        - joint_num: 关节编号 (0-14)
        - axis_offset: 0=x轴, 1=y轴, 2=z轴
        
        关节映射joint_num:
        - 拇指(thumb): 关节12, 13, 14
        - 食指(index): 关节0, 1, 2
        - 中指(middle): 关节3, 4, 5
        - 无名指(ring): 关节9, 10, 11
        - 小指(pinky): 关节6, 7, 8
        """
        
        self.left_translation = np.array(entry[0:3])
        self.left_global_orient = np.array(entry[3:6])
        self.left_hand_pose = np.array(entry[6:51])
        self.human_left_beta = np.array(entry[51:61]) # MANO shape parameters for left hand

        self.right_translation = np.array(entry[61:64])
        self.right_global_orient = np.array(entry[64:67])
        self.right_hand_pose = np.array(entry[67:112])
        self.human_right_beta = np.array(entry[112:122]) # MANO shape parameters for right hand

        # NOT USED currently
        self.padding = np.array(entry[122:212])


class StateEmbedder(nn.Module):
    def __init__(self, data_statistics_path, state_size, hidden_size):
        super(StateEmbedder, self).__init__()

        # VITRA normalizer
        self.data_statistics = read_dataset_statistics(data_statistics_path)
        self.normalizer = GaussianNormalizer(self.data_statistics)

        # self.linear = nn.Linear(action_size, hidden_size)
        self.projector = nn.Sequential(
            nn.Linear(4*2*state_size, 4*hidden_size, bias=True), # TODO: 4*2*state_size
            nn.GELU(),
            nn.Linear(4*hidden_size, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # Kaiming init helps stabilize training for GELU-activated MLP
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x):
        # x = self.linear(x)
        x = self.projector(x) # (T, hidden_size)
        return x
    
    @staticmethod
    def matrix_to_euler(matrix):
        # matrix shape: (..., 3, 3) -> euler shape: (..., 3)
        r = R.from_matrix(matrix)
        return r.as_euler('xyz', degrees=False)
    
    # NOTE: written by wuf16
    def preprocess_one_hand_states(self, hand_states):
        # This preprocesses raw states loaded from .npy file to tensor

        # 1. Translation (3 dims)
        if 'transl_camspace' in hand_states and not np.all(hand_states['transl_camspace'] == 0):
            transl = hand_states['transl_camspace']
        else:
            transl = hand_states['transl_worldspace']
        
        if transl.ndim == 1: transl = transl[None, :]

        # 2. Global Orientation (3 dims): Matrix (3x3) -> Euler (3)
        global_orient_mat = hand_states['global_orient_camspace'] # (T, 3, 3)
        global_orient = self.matrix_to_euler(global_orient_mat)      # (T, 3)

        # 3. Hand Pose (45 dims): Matrix (15x3x3) -> Euler (15x3) -> Flatten (45)
        hand_pose_mat = hand_states['hand_pose']                  # (T, 15, 3, 3)
        # view (T, 15, 3, 3) -> (T*15, 3, 3)
        T = hand_pose_mat.shape[0]
        hand_pose_euler = self.matrix_to_euler(hand_pose_mat.reshape(-1, 3, 3)) # (T*15, 3)
        hand_pose = hand_pose_euler.reshape(T, -1)              # (T, 45)

        # [Translation(3), GlobalOrient(3), HandPose(45)] -> (T, 51)
        state_vec = np.concatenate([transl, global_orient, hand_pose], axis=1)
        return state_vec

    # NOTE: written by wuf16
    def preprocess_states(self, raw_states):
        left_states = self.preprocess_one_hand_states(raw_states['left'])

        # pad mano dimensions with zeros
        left_states = np.pad(
            left_states,
            ((0, 0), (0, 10)), # pad 10 dims at the end for shape parameters
            mode='constant',
            constant_values=0
        ) # (T, 61)

        right_states = self.preprocess_one_hand_states(raw_states['right'])

        # pad mano dimensions with zeros
        right_states = np.pad(
            right_states,
            ((0, 0), (0, 10)), # pad 10 dims at the end for shape parameters
            mode='constant',
            constant_values=0
        ) # (T, 61)

        # T = left_states.shape[0]

        # Handle masks
        left_mask = raw_states['left']['kept_frames'][:, None]
        left_mask = np.tile(left_mask, (1, 61))

        right_mask = raw_states['right']['kept_frames'][:, None]
        right_mask = np.tile(right_mask, (1, 61))

        # [Left(61), Right(61)] -> (T, 122)
        state_vec = np.concatenate([left_states, right_states], axis=1)
        state_mask = np.concatenate([left_mask, right_mask], axis=1)
        
        norm_state = self.normalizer.normalize_state(state_vec)

        padded_norm_state = np.pad(
            norm_state,
            ((0, 0), (0, 90)), # pad 90 dims at the end for shape parameters
            mode='constant',
            constant_values=0
        ) # (T, 212)

        padded_mask_state = np.pad(
            state_mask,
            ((0, 0), (0, 90)), # pad 90 dims at the end for shape parameters
            mode='constant',
            constant_values=0
        ) # (T, 212)

        return torch.cat([torch.from_numpy(padded_norm_state).float(), torch.from_numpy(padded_mask_state).float()], dim=-1) # (T, 424)

    # NOTE: written by chenc81
    def extract_conditions(self, raw_states):
        """
        从 raw_states 中提取用于视频生成的 Condition。

        返回:
            hand_conditions (dict): 包含左右手在相机空间下的姿态 (T, D)
            camera_conditions (np.ndarray): 相机相对运动 (T, 6) -> [rx, ry, rz, tx, ty, tz]
        """

        # 1. 获取基础数据
        # Extrinsics: World -> Camera transform matrix (T, 4, 4)
        w2c_matrices = raw_states['extrinsics'] 
        num_frames = w2c_matrices.shape[0]

        # -----------------------------
        # A. 提取相机相对运动 (Camera Ego-motion)
        # -----------------------------
        # 计算 T 帧相对于 T-1 帧的变换。
        # Cam_t = Delta_t_prev * Cam_{t-1}  =>  Delta_t_prev = Cam_t * inv(Cam_{t-1})
        # 这里我们通常计算的是相机在世界坐标系下的运动，或者视图矩阵的变化。
        # 为了简化作为 condition，我们计算 Extrinsics 的相对变化往往更直观地对应画面的变化。

        camera_delta_motion = np.zeros((num_frames, 6)) # [rot_vec(3), trans(3)]

        for t in range(1, num_frames):
            curr_w2c = w2c_matrices[t]
            prev_w2c = w2c_matrices[t-1]

            # 计算相对变换矩阵: Rel = Curr * inv(Prev)
            # 这代表了从上一帧相机坐标系转换到当前帧相机坐标系的变换
            rel_mat = curr_w2c @ np.linalg.inv(prev_w2c)

            # 提取旋转 (Rotation Vector) 和 平移
            rel_rot = R.from_matrix(rel_mat[:3, :3]).as_rotvec()
            rel_trans = rel_mat[:3, 3]

            camera_delta_motion[t] = np.concatenate([rel_rot, rel_trans])

        # -----------------------------
        # B. 提取手部 Condition (相机空间)
        # -----------------------------
        hand_conditions = {}
        mask_conditions = {}

        for hand_type in ['left', 'right']:
            hand_data = raw_states[hand_type]

            # 1. 获取世界坐标系下的 Wrist Pose
            # Translation (T, 3)
            transl_world = hand_data['transl_worldspace']
            # Rotation Matrix (T, 3, 3) - MANO global orient
            rot_world = hand_data['global_orient_worldspace']

            # 2. 转换到相机坐标系 (World -> Camera)
            # T_cam = R_ext * T_world + T_ext
            # R_cam = R_ext * R_world

            wrist_transl_cam = np.zeros_like(transl_world)
            wrist_orient_cam = np.zeros((num_frames, 3)) # 存为旋转向量更紧凑

            for t in range(num_frames):
                w2c = w2c_matrices[t]
                r_ext = w2c[:3, :3]
                t_ext = w2c[:3, 3]

                # 位置转换
                p_world = transl_world[t]
                p_cam = r_ext @ p_world + t_ext
                wrist_transl_cam[t] = p_cam

                # 旋转转换
                r_hand_world = rot_world[t]
                r_hand_cam_mat = r_ext @ r_hand_world
                wrist_orient_cam[t] = R.from_matrix(r_hand_cam_mat).as_rotvec()

            # 3. 获取 MANO 15关节参数 (Local Pose)
            # (T, 15, 3, 3) -> 展平或者转为 axis-angle
            hand_pose_local = hand_data['hand_pose'] # 旋转矩阵格式

            # 将局部旋转矩阵转换为旋转向量 (T, 15, 3)
            hand_pose_vec = np.zeros((num_frames, 15, 3))
            for t in range(num_frames):
                for j in range(15):
                    hand_pose_vec[t, j] = R.from_matrix(hand_pose_local[t, j]).as_rotvec()

            # 展平 (T, 45)
            hand_pose_flat = hand_pose_vec.reshape(num_frames, -1)

            # 4. 组合最终的 Hand Condition Vector
            # 包含: [Wrist_Rot(3), Wrist_Trans(3), Finger_Joints(45)] = (T, 51)
            full_hand_cond = np.concatenate([
                wrist_orient_cam, 
                wrist_transl_cam, 
                hand_pose_flat
            ], axis=1)

            # pad mano shape parameters with zeros
            padded_full_hand_cond = np.pad(
                full_hand_cond,
                ((0, 0), (0, 10)), # pad 10 dims at the end for shape parameters
                mode='constant',
                constant_values=0
            ) # (T, 61)

            hand_conditions[hand_type] = padded_full_hand_cond

            hand_mask = hand_data['kept_frames'][:, None]
            hand_mask = np.tile(hand_mask, (1, 61)) # (T, 61)

            mask_conditions[hand_type] = hand_mask


        # [Left(61), Right(61)] -> (T, 122)
        state_vec = np.concatenate([hand_conditions['left'], hand_conditions['right']], axis=1)
        state_mask = np.concatenate([mask_conditions['left'], mask_conditions['right']], axis=1)

        norm_state = self.normalizer.normalize_state(state_vec)

        hand_conditions = torch.cat([torch.from_numpy(norm_state).float(), torch.from_numpy(state_mask).float()], dim=-1) # (T, 244)
        
        # padded_norm_state = np.pad(
        #     norm_state,
        #     ((0, 0), (0, 90)), # pad 90 dims at the end for shape parameters
        #     mode='constant',
        #     constant_values=0
        # ) # (T, 212)

        # padded_mask_state = np.pad(
        #     state_mask,
        #     ((0, 0), (0, 90)), # pad 90 dims at the end for shape parameters
        #     mode='constant',
        #     constant_values=0
        # ) # (T, 212)

        return hand_conditions, camera_delta_motion

    def customize_states(
        self,
        finger: Literal["thumb", "index", "middle", "ring", "pinky"],
        joint: Literal[0, 1, 2],
        axis: Literal["x", "y", "z"],
        length: int,
        speed: float = 1/54,
        state: List[float] | None = None, # [212,]
        mask: List[float] | None = None, # [212,]
    ):
        
        if state is None:
            state = [0] * 212

            state[0:3] = [0.0, 0.0, 0.5]  # x, y, z - hand 50cm in front of camera
    
            # Set global rotation (wrist orientation as Euler angles in radians)
            # Slight rotation to show the hand better
            state[3:6] = [0.1, 0.0, 0.0]  # Small rotation around x-axis
            
            # Set hand pose (15 joints × 3 Euler angles = 45 values)
            # Create a simple "pointing" gesture by bending index finger
            # Joint structure: 5 fingers × 3 joints per finger = 15 joints
            # Each joint has 3 Euler angles (x, y, z)

            # Index finger (joints 0-2): bend middle joint
            state[6 + 0*3 + 1] = 0.5  # Index finger middle joint, y-axis rotation
            
            # Middle finger (joints 3-5): slight bend
            state[6 + 3*3 + 1] = 0.3
            
            # Thumb (joints 10-12): slight bend
            state[6 + 4*3 + 1] = 0.2

        states = generate_hand_states(finger=finger, joint=joint, axis=axis, length=length, origin=state, speed=speed) # [T, 212]

        if mask is None:
            mask = torch.zeros_like(state)

            # Use left hand
            mask[:, :51] = [1]
        else:
            mask = torch.tensor(mask)
            mask = mask.unsqueeze(0).repeat(length, 1)  # [T, 212]

        states = states * mask
        states = torch.cat([states, mask.to(states.dtype)], dim=-1) # [T, 424]
        return states