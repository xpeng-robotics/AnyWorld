# SPDX-License-Identifier: Apache-2.0
# Modified for the AnyWorld public code release.
"""
Camera control stage for Wan video pipelines.

This stage processes camera control parameters and generates camera control latents
that can be used to control camera movement in video generation.
"""

import PIL
import torch
import numpy as np
from einops import repeat
from typing import Optional
from typing_extensions import Literal
from PIL import Image

from fastvideo.distributed import get_local_torch_device

from fastvideo.models.controllers.camera_controller import SimpleAdapter, process_camera_coordinates_from_npy, camera_extrinsic_to_matrix
from fastvideo.models.vaes.common import ParallelTiledVAE
from fastvideo.models.vision_utils import (get_default_height_width, normalize,
                                           pil_to_numpy, resize, numpy_to_pt)

from fastvideo.fastvideo_args import ExecutionMode, FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE


logger = init_logger(__name__)


class CameraControlStage(PipelineStage):
    """
    Stage for processing camera control parameters.
    
    This stage generates camera control latents from camera control direction,
    speed, and origin parameters, following the pattern from DiffSynth-Studio.
    """
    def __init__(self, vae, transformer=None):
        self.transformer = transformer
        self.vae: ParallelTiledVAE = vae
        if transformer is not None:
            self.adapter: SimpleAdapter = transformer.control_adapter

    def preprocess(
            self,
            image: torch.Tensor | PIL.Image.Image,
            vae_scale_factor: int,
            height: int | None = None,
            width: int | None = None,
            resize_mode: str = "default",  # "default", "fill", "crop"
    ) -> torch.Tensor:

        if isinstance(image, PIL.Image.Image):
            height, width = get_default_height_width(image, vae_scale_factor,
                                                     height, width)
            image = resize(image, height, width, resize_mode=resize_mode)
            image = pil_to_numpy(image)  # to np
            image = numpy_to_pt(image)  # to pt

        do_normalize = True
        if image.min() < 0:
            do_normalize = False
        if do_normalize:
            image = normalize(image)

        return image

    def retrieve_latents(self,
                         encoder_output: torch.Tensor,
                         generator: torch.Generator | None = None,
                         sample_mode: str = "sample"):
        if sample_mode == "sample":
            return encoder_output.sample(generator)
        elif sample_mode == "argmax":
            return encoder_output.mode()
        else:
            raise AttributeError(
                "Could not access latents of provided encoder_output")

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Process camera control parameters and generate control latents.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with camera control latents added.
        """
        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        # TODO: clean up this logic
        num_frames = batch.num_frames if isinstance(batch.num_frames, int) else batch.num_frames[0]
        height = batch.height if isinstance(batch.height, int) else batch.height[0]
        width = batch.width if isinstance(batch.width, int) else batch.width[0]


        if batch.state_camera_extrinsic_path is not None:
            state_camera_extrinsic_path = getattr(batch, 'state_camera_extrinsic_path', None)
            # Convert single str to list[str] for unified processing
            if isinstance(state_camera_extrinsic_path, str):
                state_camera_extrinsic_paths = [state_camera_extrinsic_path]
            else:
                state_camera_extrinsic_paths = state_camera_extrinsic_path

            # NOTE: info is stored in batch, iron
            if batch.camera_fx is not None:
                state_camera_extrinsic = [np.load(scep, allow_pickle=True) for scep in state_camera_extrinsic_paths]
                if state_camera_extrinsic[0].shape[-1] == 9:
                    state_camera_extrinsic = [camera_extrinsic_to_matrix(extrinsic) for extrinsic in state_camera_extrinsic]
                camera_fx = getattr(batch, 'camera_fx', None)
                camera_fy = getattr(batch, 'camera_fy', None)
                camera_cx = getattr(batch, 'camera_cx', None)
                camera_cy = getattr(batch, 'camera_cy', None)
                camera_orig_width = getattr(batch, 'camera_orig_width', None)
                camera_orig_height = getattr(batch, 'camera_orig_height', None)

                if not isinstance(camera_fx, list):
                    camera_fx = [camera_fx]
                    camera_fy = [camera_fy]
                    camera_cx = [camera_cx]
                    camera_cy = [camera_cy]
                    camera_orig_width = [camera_orig_width]
                    camera_orig_height = [camera_orig_height]

            # VITRA
            else:
                state_camera_extrinsic = [np.load(scep, allow_pickle=True).item() for scep in state_camera_extrinsic_paths]
                camera_fx = [s["intrinsics"][0, 0] for s in state_camera_extrinsic]
                camera_fy = [s["intrinsics"][1, 1] for s in state_camera_extrinsic]
                camera_cx = [s["intrinsics"][0, 2] for s in state_camera_extrinsic]
                camera_cy = [s["intrinsics"][1, 2] for s in state_camera_extrinsic]
                camera_orig_width = [s["original_pose_width"] for s in state_camera_extrinsic]
                camera_orig_height = [s["original_pose_height"] for s in state_camera_extrinsic]
                state_camera_extrinsic = [s["extrinsics"] for s in state_camera_extrinsic]

            # Fisheye (Mei) parameters — None for pinhole data
            batch_camera_xi = getattr(batch, 'camera_xi', None)
            batch_camera_dist = getattr(batch, 'camera_dist', None)

            def _get_per_sample_camera_value(camera_value, sample_idx):
                if camera_value is None:
                    return None
                if not isinstance(camera_value, list):
                    return camera_value
                if len(camera_value) == len(state_camera_extrinsic):
                    return camera_value[sample_idx]
                return camera_value

            camera_control_plucker_embedding = []
            for i in range(len(state_camera_extrinsic)):
                per_sample_xi = _get_per_sample_camera_value(batch_camera_xi, i)
                per_sample_dist = _get_per_sample_camera_value(batch_camera_dist, i)

                camera_control_plucker_embedding.append(
                    process_camera_coordinates_from_npy(
                        state_camera_extrinsic[i], 
                        height, 
                        width, 
                        original_pose_width=camera_orig_width[i], 
                        original_pose_height=camera_orig_height[i], 
                        camera_fx=camera_fx[i], 
                        camera_fy=camera_fy[i], 
                        camera_cx=camera_cx[i], 
                        camera_cy=camera_cy[i],
                        camera_xi=per_sample_xi,
                        camera_dist=per_sample_dist,
                    )[:num_frames]
                )
            camera_control_plucker_embedding = torch.stack(camera_control_plucker_embedding, dim=0)
            # Convert plucker embedding to control camera latents
            # Shape: (B, num_frames, height, width, 6) -> (B, 6, num_frames, height, width)
            control_camera_video = camera_control_plucker_embedding.permute([0, 4, 1, 2, 3])
        else:
            # Get camera control parameters from batch
            camera_control_direction = getattr(batch, 'camera_control_direction', None)
            
            if camera_control_direction is None:
                return batch
            
            # Get required parameters
            camera_control_speed = getattr(batch, 'camera_control_speed', 1/54)
            camera_control_origin = getattr(
                batch, 'camera_control_origin',
                (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
            )
            
            camera_control_plucker_embedding = self.adapter.process_camera_coordinates(
                camera_control_direction,
                num_frames,
                height,
                width,
                camera_control_speed,
                camera_control_origin
            )
        
            # Convert plucker embedding to control camera latents
            # Shape: (num_frames, height, width, 6) -> (1, 6, num_frames, height, width)
            control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        
        # Process latents similar to DiffSynth-Studio
        # Repeat first frame 4 times to match VAE frame structure
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        
        # Ensure f is divisible by 4 to match latent time dimension
        b, f, c, h, w = control_camera_latents.shape
        target_latent_frames = (num_frames - 1) // 4 + 1
        target_f = target_latent_frames * 4
        if f > target_f:
            control_camera_latents = control_camera_latents[:, :target_f]
        elif f < target_f:
            pad = control_camera_latents[:, -1:].repeat(1, target_f - f, 1, 1, 1)
            control_camera_latents = torch.cat([control_camera_latents, pad], dim=1)
        f = control_camera_latents.shape[1]

        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        
        # Convert to appropriate device and dtype
        # Try to get device from batch, fallback to cuda
        # if hasattr(batch, 'latents') and batch.latents is not None:
        #     device = batch.latents.device
        #     dtype = batch.latents.dtype
        # else:
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #     dtype = torch.bfloat16
        
        control_camera_latents_input = control_camera_latents.to(device=get_local_torch_device(), dtype=vae_dtype)
        
        # Store in batch
        batch.control_camera_latents_input = control_camera_latents_input

        # --- Image latent (match DiffSynth-Studio pattern: encode image first, then concat zeros) ---
        self.vae = self.vae.to(get_local_torch_device())

        # Process single image for I2V
        latent_height = height // self.vae.spatial_compression_ratio
        latent_width = width // self.vae.spatial_compression_ratio

        image = batch.pil_image # torch.Size([B, 3, H, W])

        # image = image.resize((width, height))

        video_condition = self.preprocess(
            image,
            vae_scale_factor=self.vae.spatial_compression_ratio,
            height=height,
            width=width).to(get_local_torch_device(), dtype=torch.float32)

        video_condition = video_condition.unsqueeze(2)

        # image = self.preprocess_video([image], torch_dtype=torch.bfloat16, device=get_local_torch_device())
        # video_condition = image
        # video_condition = torch.cat([
        #     image,
        #     image.new_zeros(image.shape[0], image.shape[1], num_frames - 1,
        #                     image.shape[3], image.shape[4])
        # ],
        #                             dim=2)
        # video_condition = video_condition.to(device=get_local_torch_device(),
        #                                      dtype=torch.f\\loat32)

        # Encode Image
        with torch.autocast(device_type="cuda",
                            dtype=vae_dtype,
                            enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)
            encoder_output = self.vae.encode(video_condition)

        if fastvideo_args.mode == ExecutionMode.PREPROCESS:
            latent_condition = encoder_output.mean
        else:
            generator = batch.generator
            if generator is None:
                raise ValueError("Generator must be provided")
            latent_condition = self.retrieve_latents(encoder_output, generator)

        # Apply shifting if needed
        if (hasattr(self.vae, "shift_factor")
                and self.vae.shift_factor is not None):
            if isinstance(self.vae.shift_factor, torch.Tensor):
                latent_condition -= self.vae.shift_factor.to(
                    latent_condition.device, latent_condition.dtype)
            else:
                latent_condition -= self.vae.shift_factor

        if isinstance(self.vae.scaling_factor, torch.Tensor):
            latent_condition = latent_condition * self.vae.scaling_factor.to(
                latent_condition.device, latent_condition.dtype)
        else:
            latent_condition = latent_condition * self.vae.scaling_factor

        y = torch.zeros(batch.latents.shape).to(device=get_local_torch_device(), dtype=vae_dtype)
        y[:, :, :1] = latent_condition

        batch.image_latent = y

        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        self.vae.to("cpu")
        
        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify camera control stage inputs."""
        result = VerificationResult()
        camera_control_direction = getattr(batch, 'camera_control_direction', None)
        
        # Camera control is optional
        if camera_control_direction is not None:
            result.add_check("camera_control_direction", camera_control_direction, V.not_none)
            result.add_check("num_frames", batch.num_frames, V.positive_int)
            result.add_check("height", batch.height, V.positive_int)
            result.add_check("width", batch.width, V.positive_int)
        
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify camera control stage outputs."""
        result = VerificationResult()
        camera_control_direction = getattr(batch, 'camera_control_direction', None)
        
        # If camera control was requested, verify output exists
        if camera_control_direction is not None:
            result.add_check("control_camera_latents_input",
                           getattr(batch, 'control_camera_latents_input', None),
                           V.not_none)
        
        return result
