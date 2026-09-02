# SPDX-License-Identifier: Apache-2.0
# Modified for the AnyWorld public code release.
"""
Camera control module for Wan video models.

This module implements camera control functionality following the pattern
from DiffSynth-Studio, allowing control of camera movement in video generation.
"""

import json
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from typing_extensions import Literal


class ResidualBlock(nn.Module):
    """Residual block for feature extraction in camera adapter."""
    
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        
        # Initialize weights with Kaiming uniform
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.conv2.weight, a=math.sqrt(5))

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


class SimpleAdapter(nn.Module):
    """
    Simple adapter for processing camera control signals.
    
    This adapter processes camera control latents and converts them
    into features that can be added to the video latents.
    """
    
    def __init__(self, in_dim, out_dim, kernel_size, stride, num_residual_blocks=1):
        super(SimpleAdapter, self).__init__()

        # Pixel Unshuffle: reduce spatial dimensions by a factor of 8
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=8)

        # Convolution: reduce spatial dimensions by a factor of 2 (without overlap)
        self.conv = nn.Conv2d(in_dim * 64, out_dim, kernel_size=kernel_size, stride=stride, padding=0)

        # Residual blocks for feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )
        
        # Initialize weights with Kaiming uniform
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

    def forward(self, x):
        # Reshape to merge the frame dimension into batch
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)

        # Pixel Unshuffle operation
        x_unshuffled = self.pixel_unshuffle(x)

        # Convolution operation
        x_conv = self.conv(x_unshuffled)

        # Feature extraction with residual blocks
        out = self.residual_blocks(x_conv)

        # Reshape to restore original bf dimension
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))

        # Permute dimensions to reorder (if needed)
        out = out.permute(0, 2, 1, 3, 4)

        return out
    
    def process_camera_coordinates(
        self,
        direction: Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"],
        length: int,
        height: int,
        width: int,
        speed: float = 1/54,
        origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    ):
        """
        Process camera coordinates to generate plucker embeddings.
        
        Args:
            direction: Camera movement direction
            length: Number of frames
            height: Video height
            width: Video width
            speed: Movement speed
            origin: Origin camera parameters
            
        Returns:
            Plucker embedding tensor
        """
        if origin is None:
            origin = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
        coordinates = generate_camera_coordinates(direction, length, speed, origin)
        plucker_embedding = process_pose_file(coordinates, width, height)
        return plucker_embedding


class Camera(object):
    """Camera parameter representation.
    
    Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def get_relative_pose(cam_params):
    """Get relative camera poses.
    
    Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def custom_meshgrid(*args):
    # torch>=2.0.0 only
    return torch.meshgrid(*args, indexing='ij')


def _mei_unproject(u, v, fx, fy, cx, cy, xi, dist, num_iters=10):
    """Mei unified spherical model back-projection (pixel -> unit ray).

    Derivation from forward model:
        forward: mx = X / (Z + xi*d),  my = Y / (Z + xi*d),  d = ||P||
        On the unit sphere (d=1): r² = (1 - Z²) / (Z + xi)²
        Solving for Z:  (1 + r²)Z² + 2r²xi·Z + (r²xi² - 1) = 0
        => Z = (-r²xi + sqrt(1 + r²(1 - xi²))) / (1 + r²)
        Then X = mx(Z + xi), Y = my(Z + xi).

    Args:
        u, v: pixel coordinates, same-shape tensors  (*, HW)
        fx, fy, cx, cy: intrinsics (broadcastable)
        xi: Mei mirror parameter (scalar or broadcastable)
        dist: distortion coeffs [k1, k2, k3, p1, p2] tensor of shape (5,)
              (same order as the rendering script NPY, NOT OpenCV order)
        num_iters: Newton iterations for iterative undistortion

    Returns:
        directions: (*, HW, 3) unit ray directions in camera frame
    """
    mx = (u - cx) / fx
    my = (v - cy) / fy

    if dist is not None and dist.numel() >= 5:
        k1, k2, k3, p1, p2 = dist[0], dist[1], dist[2], dist[3], dist[4]
        mx_u, my_u = mx.clone(), my.clone()
        for _ in range(num_iters):
            r2 = mx_u * mx_u + my_u * my_u
            r4 = r2 * r2
            r6 = r4 * r2
            radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            dx = 2.0 * p1 * mx_u * my_u + p2 * (r2 + 2.0 * mx_u * mx_u)
            dy = p1 * (r2 + 2.0 * my_u * my_u) + 2.0 * p2 * mx_u * my_u
            mx_u = (mx - dx) / radial
            my_u = (my - dy) / radial
        mx, my = mx_u, my_u

    r2 = mx * mx + my * my
    alpha = 1.0 - xi * xi
    disc = torch.sqrt(torch.clamp(1.0 + r2 * alpha, min=1e-10))
    zs = (-r2 * xi + disc) / (1.0 + r2)
    lam = zs + xi
    xs = lam * mx
    ys = lam * my

    directions = torch.stack([xs, ys, zs], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    return directions


def ray_condition(K, c2w, H, W, device, camera_xi=None, camera_dist=None):
    """Compute ray conditions for plucker embedding.

    When camera_xi is provided (> 0), uses Mei unified spherical model for
    back-projection (fisheye). Otherwise falls back to pinhole model.
    """
    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    use_mei = (camera_xi is not None
               and (isinstance(camera_xi, (int, float)) and camera_xi > 0
                    or isinstance(camera_xi, torch.Tensor) and camera_xi.item() > 0))

    if use_mei:
        xi_val = float(camera_xi) if not isinstance(camera_xi, torch.Tensor) else camera_xi
        xi_t = torch.tensor(xi_val, device=device, dtype=c2w.dtype)
        dist_t = None
        if camera_dist is not None:
            if isinstance(camera_dist, (list, tuple)):
                dist_t = torch.tensor(camera_dist, device=device, dtype=c2w.dtype)
            elif isinstance(camera_dist, torch.Tensor):
                dist_t = camera_dist.to(device=device, dtype=c2w.dtype)
        directions = _mei_unproject(i, j, fx, fy, cx, cy, xi_t, dist_t)
    else:
        zs = torch.ones_like(i)
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        zs = zs.expand_as(ys)
        directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
        directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    return plucker


def process_pose_file(cam_params, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu', return_poses=False, camera_xi=None, camera_dist=None):
    """Process camera pose file to generate plucker embeddings."""
    if return_poses:
        return cam_params
    else:
        cam_params = [Camera(cam_param) for cam_param in cam_params]

        sample_wh_ratio = width / height
        pose_wh_ratio = original_pose_width / original_pose_height

        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = height * pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_ori_w * cam_param.fx / width
        else:
            resized_ori_h = width / pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_ori_h * cam_param.fy / height

        intrinsic = np.asarray([[cam_param.fx * width,
                                cam_param.fy * height,
                                cam_param.cx * width,
                                cam_param.cy * height]
                                for cam_param in cam_params], dtype=np.float32)

        K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
        c2ws = get_relative_pose(cam_params)
        c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
        plucker_embedding = ray_condition(K, c2ws, height, width, device=device, camera_xi=camera_xi, camera_dist=camera_dist)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
        plucker_embedding = plucker_embedding[None]
        plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
        return plucker_embedding


# NOTE: origin from DiffSynth
def generate_camera_coordinates_diffsynth(
    direction: Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown", "In", "Out"],
    length: int,
    speed: float = 1/54,
    origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
):
    """Generate camera coordinates for a given direction and length."""
    coordinates = [list(origin)]
    while len(coordinates) < length:
        coor = coordinates[-1].copy()
        if "Left" in direction:
            coor[9] += speed
        if "Right" in direction:
            coor[9] -= speed
        if "Up" in direction:
            coor[13] += speed
        if "Down" in direction:
            coor[13] -= speed
        if "In" in direction:
            coor[18] -= speed
        if "Out" in direction:
            coor[18] += speed
        coordinates.append(coor)
    return coordinates

def get_rotation_matrix(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif axis == "y":
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    elif axis == "z":
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    else:
        return np.eye(3)

def generate_camera_coordinates(
    direction,
    length: int,
    speed: float = 1/54,
    origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
):
    if not isinstance(direction, list):
        direction = [direction]
    
    direction = np.repeat(list(direction), [length // len(direction)] * len(direction)).tolist() + [direction[-1]] * (length % len(direction))

    coordinates = [list(origin)]
    # while len(coordinates) < length:
    for i in range(length - 1):
        coor = coordinates[-1].copy()
        # origin
        if "Left" in direction:
            coor[9] += speed
        if "Right" in direction:
            coor[9] -= speed
        if "Up" in direction:
            coor[13] += speed
        if "Down" in direction:
            coor[13] -= speed
        if "In" in direction:
            coor[18] -= speed
        if "Out" in direction:
            coor[18] += speed

        w2c_mat = np.array(coor[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        position = w2c_mat_4x4[:3, 3]

        forward = w2c_mat_4x4[2, :3]
        right = w2c_mat_4x4[0, :3]

        if "w" in direction[i]:
            w2c_mat_4x4[:3, 3] = position - forward * speed
        if "s" in direction[i]:
            w2c_mat_4x4[:3, 3] = position + forward * speed
        if "a" in direction[i]:
            w2c_mat_4x4[:3, 3] = position + right * speed
        if "d" in direction[i]:
            w2c_mat_4x4[:3, 3] = position - right * speed

        # View rotation
        angle = np.deg2rad(1)
        rotation = np.eye(3)
        if "8" in direction[i]:
            rotation = get_rotation_matrix("x", -angle)
        if "2" in direction[i]:
            rotation = get_rotation_matrix("x", angle)
        if "6" in direction[i]:
            rotation = get_rotation_matrix("y", -angle)
        if "4" in direction[i]:
            rotation = get_rotation_matrix("y", angle)

        # Apply rotation
        w2c_mat_4x4[:3, :3] = rotation @ w2c_mat_4x4[:3, :3]
        coor[7:] = w2c_mat_4x4[:3, :].flatten()

        coordinates.append(coor)
    return coordinates


def rot6d_to_matrix(rot6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.

    6D rotation = first two columns of rotation matrix (flattened).
    Args:
        rot6d: (6,) array - [col1_x, col1_y, col1_z, col2_x, col2_y, col2_z]
    Returns:
        R: (3, 3) rotation matrix
    """
    # Extract first two columns
    col1 = rot6d[:3]
    col2 = rot6d[3:6]

    # Gram-Schmidt orthonormalization
    col1 = col1 / (np.linalg.norm(col1) + 1e-8)
    col2 = col2 - np.dot(col1, col2) * col1
    col2 = col2 / (np.linalg.norm(col2) + 1e-8)

    # Third column is cross product
    col3 = np.cross(col1, col2)

    return np.stack([col1, col2, col3], axis=1)


def camera_extrinsic_to_matrix(extrinsic_9d):
    """
    Convert 9-dim camera extrinsic to 4x4 homogeneous matrix.

    Args:
        extrinsic_9d: (9,) array - [pos_x, pos_y, pos_z, rot6d[0:6]]
                      or (T, 9) array for multiple frames
    Returns:
        T_cam: (4, 4) or (T, 4, 4) homogeneous transformation matrix
               Transforms points from world frame to camera frame
    """
    if extrinsic_9d.ndim == 1:
        pos = extrinsic_9d[:3]
        rot6d = extrinsic_9d[3:9]

        R = rot6d_to_matrix(rot6d)

        T_cam = np.eye(4)
        T_cam[:3, :3] = R
        T_cam[:3, 3] = pos
        return T_cam
    else:
        # Batch version for (T, 9)
        T = extrinsic_9d.shape[0]
        matrices = np.zeros((T, 4, 4))
        for i in range(T):
            matrices[i] = camera_extrinsic_to_matrix(extrinsic_9d[i])
        return matrices


def adjust_intrinsics_for_crop_resize(intrinsics, orig_h, orig_w, target_h, target_w, top_crop=False):
    """
    根据 crop 和 resize 变换调整相机内参。
    
    Args:
        intrinsics: 原始内参矩阵 (3, 3)
        orig_h: 原始图像高度
        orig_w: 原始图像宽度
        target_h: 目标图像高度
        target_w: 目标图像宽度
        top_crop: 是否使用 top crop (False 表示 center crop)
    
    Returns:
        调整后的内参矩阵 (3, 3)
    """
    # 计算 crop 参数（与 CenterCropResizeVideo 中的逻辑一致）
    tr = target_h / target_w
    if orig_h / orig_w > tr:
        new_h = int(orig_w * tr)
        new_w = orig_w
    else:
        new_h = orig_h
        new_w = int(orig_h / tr)
    
    crop_i = 0 if top_crop else int(round((orig_h - new_h) / 2.0))
    crop_j = int(round((orig_w - new_w) / 2.0))
    
    # 计算缩放因子
    scale_h = target_h / new_h
    scale_w = target_w / new_w
    
    # 调整内参
    # fx, fy: 需要乘以对应的缩放因子
    # cx, cy: 先减去 crop 偏移，再乘以缩放因子
    fx = intrinsics[0, 0] * scale_w
    fy = intrinsics[1, 1] * scale_h
    cx = (intrinsics[0, 2] - crop_j) * scale_w
    cy = (intrinsics[1, 2] - crop_i) * scale_h
    
    # 构建新的内参矩阵
    adjusted_intrinsics = intrinsics.copy()
    adjusted_intrinsics[0, 0] = fx
    adjusted_intrinsics[1, 1] = fy
    adjusted_intrinsics[0, 2] = cx
    adjusted_intrinsics[1, 2] = cy
    
    return adjusted_intrinsics


def adjust_intrinsics_for_crop_resize_iron(camera_fx, camera_fy, camera_cx, camera_cy, orig_h, orig_w, target_h, target_w, top_crop=False):
    """
    根据 crop 和 resize 变换调整相机内参。
    
    Args:
        camera_fx: 原始焦距
        camera_fy: 原始焦距
        camera_cx: 原始主点
        camera_cy: 原始主点
        orig_h: 原始图像高度
        orig_w: 原始图像宽度
        target_h: 目标图像高度
        target_w: 目标图像宽度
        top_crop: 是否使用 top crop (False 表示 center crop)
    
    Returns:
        调整后的内参矩阵 (3, 3)
    """
    # 计算 crop 参数（与 CenterCropResizeVideo 中的逻辑一致）
    tr = target_h / target_w
    if orig_h / orig_w > tr:
        new_h = int(orig_w * tr)
        new_w = orig_w
    else:
        new_h = orig_h
        new_w = int(orig_h / tr)
    
    crop_i = 0 if top_crop else int(round((orig_h - new_h) / 2.0))
    crop_j = int(round((orig_w - new_w) / 2.0))
    
    # 计算缩放因子
    scale_h = target_h / new_h
    scale_w = target_w / new_w
    
    # 调整内参
    # fx, fy: 需要乘以对应的缩放因子
    # cx, cy: 先减去 crop 偏移，再乘以缩放因子
    fx = camera_fx * scale_w
    fy = camera_fy * scale_h
    cx = (camera_cx - crop_j) * scale_w
    cy = (camera_cy - crop_i) * scale_h
    
    # 构建新的内参矩阵
    return fx, fy, cx, cy


def process_camera_coordinates_from_json(
    json_data,
    height: int,
    width: int,
    original_pose_width: int = 1280,
    original_pose_height: int = 720,
    origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
):
    """
    Process camera coordinates from DL3DV JSON format data (OpenGL).
    
    Args:
        height: Target height for the output
        width: Target width for the output
        original_pose_width: Original width used in COLMAP reconstruction (auto-detected if None)
        original_pose_height: Original height used in COLMAP reconstruction (auto-detected if None)
        
    Returns:
        plucker_embedding: Camera pose embeddings in plucker coordinates
    """
    if isinstance(json_data, str):
        with open(json_data, 'r') as f:
            json_data = json.load(f)

    # Extract camera parameters
    camera_params = {
        'w': json_data.get('w', 1280),
        'h': json_data.get('h', 720),
        'fl_x': json_data.get('fl_x', 1000.0),
        'fl_y': json_data.get('fl_y', 1000.0),
        'cx': json_data.get('cx', 0.5),
        'cy': json_data.get('cy', 0.5),
        'camera_model': json_data.get('camera_model', 'OPENCV')
    }

    # Use provided dimensions or auto-detect from data
    if original_pose_width is None:
        original_pose_width = camera_params['w']
    if original_pose_height is None:
        original_pose_height = camera_params['h']
    
    cam_params = []
    sorted_frames = sorted(json_data['frames'], key=lambda x: x.get('colmap_im_id', 0))

    for i, frame in enumerate(sorted_frames):
        timestamp = i * 33333

        # Extract camera intrinsics
        fx = camera_params['fl_x'] / camera_params['w']
        fy = camera_params['fl_y'] / camera_params['h']
        cx = camera_params['cx'] / camera_params['w']  # Normalize to [0, 1]
        cy = camera_params['cy'] / camera_params['h']  # Normalize to [0, 1]
        
        cam = np.array(frame["transform_matrix"], dtype=np.float64)

        if cam.shape[0] == 3:
            cam = np.vstack((cam, np.array([[0, 0, 0, 1]])))
        cam[2, :] *= -1
        cam = cam[np.array([0, 2, 1, 3]), :]
        cam[0:3, 1:3] *= -1 # NOTE: from https://github.com/DL3DV-10K/Dataset/issues/4

        c2w = cam
        w2c = np.linalg.inv(c2w)

        pose = w2c[:3, :]  # shape (3,4)
        cam_param = [timestamp, fx, fy, cx, cy, 0, 0] + pose.flatten().tolist()

        assert len(cam_param) == 19
        
        cam_params.append(cam_param)
    
    # Process the camera parameters using the existing pipeline
    plucker_embedding = process_pose_file(
        cam_params, 
        width, 
        height, 
        original_pose_width, 
        original_pose_height, 
    )
    
    return plucker_embedding






# def process_camera_coordinates_from_npy(
#     extrinsics,
#     height: int,
#     width: int,
#     original_pose_width: int,
#     original_pose_height: int,
#     camera_fx: float,
#     camera_fy: float,
#     camera_cx: float,
#     camera_cy: float,
#     origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
# ):
#     """
#     Process camera coordinates from NPY format data.
    
#     The NPY file should contain:
#     - 'extrinsics': array of w2c matrices (world-to-camera transformation matrices)
#     - 'intrinsics': camera intrinsic matrix (3x3)
    
#     Args:
#         npy_path: Path to the NPY file or loaded NPY data dict
#         height: Target height for the output
#         width: Target width for the output
#         original_pose_width: Original width used in camera calibration
#         original_pose_height: Original height used in camera calibration
#         origin: Unused parameter (kept for API compatibility)
        
#     Returns:
#         plucker_embedding: Camera pose embeddings in plucker coordinates
#     """
#     # if isinstance(npy_path, str):
#     #     npy_data = np.load(npy_path, allow_pickle=True).item()
#     # else:
#     #     npy_data = npy_path

#     # w2c_matrices = npy_data['extrinsics']
#     w2c_matrices = extrinsics
#     # intrinsics = npy_data['intrinsics']

#     # original_pose_width = original_pose_width if original_pose_width is not None else npy_data['original_pose_width']
#     # original_pose_height = original_pose_height if original_pose_height is not None else npy_data['original_pose_height']

#     # Adjust intrinsics for crop and resize transformation

#     fx, fy, cx, cy = adjust_intrinsics_for_crop_resize_iron(
#         camera_fx, camera_fy, camera_cx, camera_cy, original_pose_height, original_pose_width, height, width, top_crop=False
#     )


#     # intrinsics = adjust_intrinsics_for_crop_resize(
#     #     intrinsics, 
#     #     orig_h=original_pose_height, 
#     #     orig_w=original_pose_width, 
#     #     target_h=height, 
#     #     target_w=width, 
#     #     top_crop=False
#     # )

#     # Extract pixel coordinates from adjusted intrinsics
#     # fx_pixel = intrinsics[0, 0]
#     # fy_pixel = intrinsics[1, 1]
#     # cx_pixel = intrinsics[0, 2]
#     # cy_pixel = intrinsics[1, 2]

#     # Normalize intrinsics to [0, 1] range (required by process_pose_file)
#     # process_pose_file expects normalized intrinsics and will multiply by width/height
#     fx = fx / width
#     fy = fy / height
#     cx = cx / width
#     cy = cy / height
    
#     cam_params = []
#     for i, w2c in enumerate(w2c_matrices):
#         timestamp = i
        
#         pose = w2c[:3, :]  # shape (3,4)
#         cam_param = [timestamp, fx, fy, cx, cy, 0, 0] + pose.flatten().tolist()

#         assert len(cam_param) == 19
#         cam_params.append(cam_param)
    
#     # Process the camera parameters using the existing pipeline
#     plucker_embedding = process_pose_file(
#         cam_params, 
#         width, 
#         height, 
#         original_pose_width=width, 
#         original_pose_height=height, # NOTE: already adjusted for crop and resize
#     )
    
#     return plucker_embedding





def process_camera_coordinates_from_npy(
    extrinsics,
    height: int,
    width: int,
    original_pose_width: int,
    original_pose_height: int,
    camera_fx: float,
    camera_fy: float,
    camera_cx: float,
    camera_cy: float,
    origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
    camera_xi=None,
    camera_dist=None,
):
    """
    Process camera coordinates from NPY format data.
    """
    # w2c_matrices 是一个 batch 的外参
    w2c_matrices = extrinsics

    # 1. 调整内参 (这部分保持你原有的逻辑不变，看起来是没问题的)
    fx, fy, cx, cy = adjust_intrinsics_for_crop_resize_iron(
        camera_fx, camera_fy, camera_cx, camera_cy, original_pose_height, original_pose_width, height, width, top_crop=False
    )

    # 2. 归一化内参 (这是 process_pose_file 要求的格式)
    fx = fx / width
    fy = fy / height
    cx = cx / width
    cy = cy / height
    
    cam_params = []
    
    # 3. 遍历每一帧的外参
    for i, w2c in enumerate(w2c_matrices):
        timestamp = i
        
        #Params: w2c 可能是一个 tensor 或 numpy array
        # === [核心修复开始] ===
        # 如果是 Tensor 且是一维的，说明被压扁了，需要 Reshape 回来
        if isinstance(w2c, torch.Tensor):
            if w2c.dim() == 1:
                if w2c.numel() == 16:
                    w2c = w2c.view(4, 4)
                elif w2c.numel() == 12:
                    w2c = w2c.view(3, 4)
                else:
                    raise ValueError(f"Unexpected w2c shape: {w2c.shape}, expected 12 or 16 elements.")
        
        # 如果是 Numpy Array (以防万一数据源变了)
        elif isinstance(w2c, np.ndarray):
            if w2c.ndim == 1:
                if w2c.size == 16:
                    w2c = w2c.reshape(4, 4)
                elif w2c.size == 12:
                    w2c = w2c.reshape(3, 4)
        # === [核心修复结束] ===

        # 现在 w2c 肯定是 2D 矩阵了，可以安全切片
        pose = w2c[:3, :]  # shape (3,4)
        
        # 拼接参数：[时间戳, 内参4项, 0, 0, 外参12项] = 总共 19 项
        cam_param = [timestamp, fx, fy, cx, cy, 0, 0] + pose.flatten().tolist()

        assert len(cam_param) == 19
        cam_params.append(cam_param)
    
    # Process the camera parameters using the existing pipeline
    plucker_embedding = process_pose_file(
        cam_params, 
        width, 
        height, 
        original_pose_width=width, 
        original_pose_height=height, # NOTE: already adjusted for crop and resize
        camera_xi=camera_xi,
        camera_dist=camera_dist,
    )
    
    return plucker_embedding