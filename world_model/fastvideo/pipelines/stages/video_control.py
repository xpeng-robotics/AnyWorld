# SPDX-License-Identifier: Apache-2.0
# Modified for the AnyWorld public code release.
"""
Camera control stage for Wan video pipelines.

This stage processes camera control parameters and generates camera control latents
that can be used to control camera movement in video generation.
"""

import os
import PIL
import torch
import numpy as np
import imageio.v2 as imageio

from einops import repeat
from typing import Optional
from typing_extensions import Literal
from PIL import Image
import torchvision
import torch.nn.functional as nnF
import torchvision.transforms.functional as F

from fastvideo.distributed import get_local_torch_device

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


class VideoControlStage(PipelineStage):
    """
    Stage for processing video control parameters.
    
    This stage generates video control latents from video control direction,
    speed, and origin parameters, following the pattern from DiffSynth-Studio.
    """
    def __init__(self, vae):
        self.vae: ParallelTiledVAE = vae

    @staticmethod
    def get_center_crop_params(orig_h: int, orig_w: int, target_h: int,
                               target_w: int) -> tuple[int, int, int, int]:
        """Compute center-crop params to match target aspect ratio."""
        target_ratio = target_h / target_w
        if (orig_h / orig_w) > target_ratio:
            new_h = int(orig_w * target_ratio)
            new_w = orig_w
        else:
            new_h = orig_h
            new_w = int(orig_h / target_ratio)
        crop_i = int(round((orig_h - new_h) / 2.0))
        crop_j = int(round((orig_w - new_w) / 2.0))
        return crop_i, crop_j, new_h, new_w

    @staticmethod
    def mp4_to_pil_list(video_path):
        reader = imageio.get_reader(video_path, format="ffmpeg")
        frames = []

        for frame in reader:
            frames.append(Image.fromarray(frame))

        reader.close()
        return frames

    @classmethod
    def center_crop_resize_tensor(cls, video: torch.Tensor, target_h: int,
                                  target_w: int) -> torch.Tensor:
        """Match the model's center-crop + resize geometry for TCHW videos."""
        if video.ndim != 4:
            raise ValueError(
                f"Expected video tensor in TCHW format, got shape {tuple(video.shape)}"
            )
        crop_i, crop_j, crop_h, crop_w = cls.get_center_crop_params(
            video.shape[-2], video.shape[-1], target_h, target_w)
        video = video[:, :, crop_i:crop_i + crop_h, crop_j:crop_j + crop_w]
        if video.shape[-2:] != (target_h, target_w):
            video = nnF.interpolate(video,
                                    size=(target_h, target_w),
                                    mode="bilinear",
                                    align_corners=True,
                                    antialias=True)
        return video

    @classmethod
    def center_crop_resize_image(cls, image: torch.Tensor | PIL.Image.Image,
                                 target_h: int,
                                 target_w: int) -> torch.Tensor | PIL.Image.Image:
        """Match the model's center-crop + resize geometry for reference images."""
        orig_h, orig_w = image.shape[-2:] if isinstance(
            image, torch.Tensor) else (image.height, image.width)
        crop_i, crop_j, crop_h, crop_w = cls.get_center_crop_params(
            orig_h, orig_w, target_h, target_w)
        image = F.crop(image, crop_i, crop_j, crop_h, crop_w)
        if (crop_h, crop_w) != (target_h, target_w):
            image = F.resize(image,
                             [target_h, target_w],
                             interpolation=F.InterpolationMode.BILINEAR,
                             antialias=True)
        return image

    def preprocess(
            self,
            image: torch.Tensor | PIL.Image.Image,
            vae_scale_factor: int,
            height: int | None = None,
            width: int | None = None,
            resize_mode: str = "default",  # "default", "fill", "crop"
    ) -> torch.Tensor:

        if isinstance(image, PIL.Image.Image):
            # height, width = get_default_height_width(image, vae_scale_factor,
            #                                          height, width)
            # image = resize(image, height, width, resize_mode=resize_mode)
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
        Process video control parameters following DiffSynth-Studio FunControl and FunReference patterns.
        
        This implements:
        - FunControl: Encodes control video and concatenates with y latents
        - FunReference: Encodes reference image for reference latents
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with control latents and reference latents added.
        """
        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[
            fastvideo_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        # Get control video from batch (FunControl)
        control_video_path = getattr(batch, 'control_video_path', None)
        
        # Convert single str to list[str] for unified processing
        if isinstance(control_video_path, str):
            control_video_paths = [control_video_path]
        else:
            control_video_paths = control_video_path
        
        # Get required parameters - handle both single values and lists
        # TODO: clean up this logic
        num_frames = batch.num_frames if isinstance(batch.num_frames, int) else batch.num_frames[0]
        height = batch.height if isinstance(batch.height, int) else batch.height[0]
        width = batch.width if isinstance(batch.width, int) else batch.width[0]
        spatial_preprocess = getattr(batch, 'spatial_preprocess',
                                     'direct_resize')

        latents = batch.latents
        # clip_feature = batch.image_embeds
        # y = batch.image_latent
        batch_size = len(control_video_paths)

        # Move VAE to device (matching DiffSynth-Studio pattern)
        self.vae = self.vae.to(get_local_torch_device())

        # ========== FunControl: Process control video for each item in batch ==========
        # TODO: optimize batch processing
        control_video_conditions = []
        
        for i in range(batch_size):
            control_video_path = control_video_paths[i]
            
            # Load control video using torchvision (similar to VideoTransformStage)
            control_video, _, _ = torchvision.io.read_video(control_video_path, output_format="TCHW")
            control_video = control_video[:num_frames]  # (T, C, H, W)


            control_video = control_video.to(torch.float32) / 255.0
            if spatial_preprocess == "center_crop_resize":
                control_video = self.center_crop_resize_tensor(
                    control_video, height, width)
            elif control_video.shape[2] != height or control_video.shape[3] != width:
                control_video = nnF.interpolate(
                    control_video,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )

            # Save crop+resize result for debugging
            # save_dir = os.path.dirname(control_video_path)
            # save_name = os.path.splitext(os.path.basename(control_video_path))[0]
            # save_path = os.path.join(save_dir, f"{save_name}_crop_resize.mp4")
            # video_uint8 = (control_video.clamp(0, 1) * 255.0).to(torch.uint8)
            # video_uint8 = video_uint8.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
            # imageio.mimsave(save_path, video_uint8, fps=16)
            
            
            # Convert to (C, T, H, W) format for preprocessing
            control_video = control_video.permute(1, 0, 2, 3)  # (C, T, height, width)
            
            # Apply preprocessing (normalization, etc.)
            control_video = self.preprocess(
                control_video,
                vae_scale_factor=self.vae.spatial_compression_ratio,
                height=height,
                width=width
            )  # (C, T, H, W)
            # Preprocess each frame and stack into C T H W format
            # Resize frames to target dimensions first (matching FunReference pattern)
            # processed_frames = []
            # for frame in control_frames:
            #     # Resize frame to target dimensions
            #     frame = frame.resize((width, height), Image.Resampling.LANCZOS)
            #     # Preprocess frame to tensor (returns (c, h, w))
            #     processed_frame = self.preprocess(
            #         frame,
            #         vae_scale_factor=self.vae.spatial_compression_ratio,
            #         height=height,
            #         width=width
            #     )

            #     # Remove batch dimension if present: (1, c, h, w) -> (c, h, w)
            #     processed_frames.append(processed_frame.squeeze(0))
            
            # Stack frames along time dimension: stack frames (list of (c, h, w)) -> (c, f, h, w)
            # control_video_condition = control_video # (c, f, h, w)
            control_video_conditions.append(control_video)

        
        # Stack all batch items: list of (c, f, h, w) -> (B, c, f, h, w)
        control_video_conditions = torch.stack(control_video_conditions, dim=0)  # (B, c, f, h, w)
        control_video_conditions = control_video_conditions.to(
            device=get_local_torch_device(), dtype=torch.float32)

        # Encode Control Video (matching FunControl: pipe.vae.encode)
        with torch.autocast(device_type="cuda",
                            dtype=vae_dtype,
                            enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                control_video_conditions = control_video_conditions.to(vae_dtype)
            control_video_encoder_output = self.vae.encode(control_video_conditions)

        # Match the expected control-video latent preprocessing:
        # are stored as deterministic VAE posterior means without additional
        # shift/scaling normalization.
        control_latents = control_video_encoder_output.mean

        batch.control_video_latent = control_latents # NOTE: important for preprocess

        # Convert to target dtype and device (matching FunControl: .to(dtype=pipe.torch_dtype, device=pipe.device))
        # control_latents = control_latents.to(
        #     dtype=control_latents.dtype, device=control_latents.device)

        # Calculate y_dim and concatenate (matching FunControl exactly)
        # y_dim = pipe.dit.in_dim - control_latents.shape[1] - latents.shape[1]
        y_dim = 48 - control_latents.shape[1] - latents.shape[1] # TODO: hard code for Wan2.1-Fun-Control-14B

        # if clip_feature is None or y is None:
        # clip_feature = torch.zeros(
        #     (1, 257, 1280), 
        #     dtype=control_latents.dtype, 
        #     device=control_latents.device
        # )
        y = torch.zeros(
            (batch_size, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), 
            dtype=control_latents.dtype, 
            device=control_latents.device
        )
        # else:
        # y = y[:, -y_dim:]
        
        # Concatenate control latents with y (matching FunControl: torch.concat([control_latents, y], dim=1))
        # control_latents: (B, c, t, h, w), y: (B, y_dim, t, h, w) -> (B, c+y_dim, t, h, w)
        y = torch.concat([control_latents, y], dim=1)

        batch.image_latent = y
        # batch.image_embeds = clip_feature

        # ========== FunReference: Process reference image ==========
        # Resize reference image first (matching FunReference: reference_image.resize((width, height)))
        reference_image = batch.pil_image
        assert reference_image is not None, "Reference image must be provided in batch.pil_image"
        # TODO: clean up this code
        if isinstance(reference_image, torch.Tensor):
            reference_height = reference_image.shape[2]
            reference_width = reference_image.shape[3]
        else:
            reference_height = reference_image.height
            reference_width = reference_image.width
            
        if spatial_preprocess == "center_crop_resize":
            reference_image = self.center_crop_resize_image(
                reference_image, height, width)
        else:
            reference_image = F.resize(
                reference_image,
                (height, width),
                interpolation=F.InterpolationMode.BILINEAR,
                antialias=True,
            )

        # Save crop+resize result for debugging
        # if control_video_paths:
        #     ref_save_dir = os.path.dirname(control_video_paths[0])
        #     ref_base = os.path.splitext(os.path.basename(control_video_paths[0]))[0]
        # else:
        #     ref_save_dir = os.getcwd()
        #     ref_base = "reference_image"
        # ref_save_path = os.path.join(ref_save_dir,
        #                              f"{ref_base}_reference_crop_resize.png")
        # reference_image.save(ref_save_path)
        # Preprocess as video (list of one image) - matching FunReference: pipe.preprocess_video([reference_image])
        video_condition = self.preprocess(
            reference_image,
            vae_scale_factor=self.vae.spatial_compression_ratio,
            height=height,
            width=width
        ).to(get_local_torch_device(), dtype=torch.float32)

        # Add time dimension: (B, c, h, w) -> (B, c, 1, h, w)
        video_condition = video_condition.unsqueeze(2)

        # Encode reference image (matching FunReference: pipe.vae.encode)
        with torch.autocast(device_type="cuda",
                            dtype=vae_dtype,
                            enabled=vae_autocast_enabled):
            if fastvideo_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)
            encoder_output = self.vae.encode(video_condition)

        # Reference latents come from the VAE posterior:
        # means without an extra shift/scaling normalization pass.
        reference_latents = encoder_output.mean

        batch.reference_latents = reference_latents

        # Cleanup
        if hasattr(self, 'maybe_free_model_hooks'):
            self.maybe_free_model_hooks()

        self.vae.to("cpu")
        
        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify control stage inputs."""
        result = VerificationResult()
        control_video = getattr(batch, 'control_video', None)
        
        if control_video is not None:
            result.add_check("control_video", control_video, V.not_none)
            result.add_check("num_frames", batch.num_frames, V.positive_int)
            result.add_check("height", batch.height, V.positive_int)
            result.add_check("width", batch.width, V.positive_int)
        
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify control stage outputs."""
        result = VerificationResult()
        control_video = getattr(batch, 'control_video', None)
        
        # If control video was requested, verify output exists
        if control_video is not None:
            result.add_check("reference_latents",
                           getattr(batch, 'reference_latents', None),
                           V.not_none)
        
        return result
