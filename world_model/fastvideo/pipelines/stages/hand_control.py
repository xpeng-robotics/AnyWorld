# SPDX-License-Identifier: Apache-2.0
# Modified for the AnyWorld public code release.
import torch
import numpy as np
from einops import repeat
from typing import Optional
from typing_extensions import Literal

from fastvideo.distributed import get_local_torch_device

from fastvideo.models.controllers.hand_controller import StateEmbedder
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


class HandControlStage(PipelineStage):
    """
    Stage for processing hand control parameters.
    
    This stage generates hand control latents from hand control parameters
    (finger, joint, axis), following the pattern from camera control stage.
    """
    def __init__(self, vae, transformer):
        self.transformer = transformer
        self.vae: ParallelTiledVAE = vae
        self.hand_control_adapter: StateEmbedder = transformer.hand_control_adapter

    def preprocess(
        self,
        image: torch.Tensor,
        vae_scale_factor: int,
        height: int | None = None,
        width: int | None = None,
        resize_mode: str = "default",  # "default", "fill", "crop"
    ) -> torch.Tensor:

        if isinstance(image, torch.Tensor):
            return image

        if isinstance(image, type(None)):
            return None

        # Handle PIL Image if needed
        from PIL import Image
        if isinstance(image, Image.Image):
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
        Process hand control parameters and generate control latents.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with hand control latents added.
        """        
        if batch.control_hand_state_path is None:
            return batch

        # Get required parameters
        num_frames = batch.num_frames

        control_hand_state_path = batch.control_hand_state_path
        control_hand_state = np.load(control_hand_state_path, allow_pickle=True)
        control_hand_state = control_hand_state.item()   
        
        # Generate hand states using the hand controller
        # NOTE: camera condition not used now
        hand_states, camera_delta_motion = self.hand_control_adapter.extract_conditions(
            control_hand_state
        )

        hand_states = hand_states[:num_frames].to(device=get_local_torch_device(), dtype=torch.float32)

        # TODO: batch size 1
        hand_states = hand_states.unsqueeze(0)  # (B, T, state_dim)

        hand_states = torch.concat(
            [
                torch.repeat_interleave(hand_states[:, 0:1], repeats=4, dim=1),
                hand_states[:, 1:]
            ], dim=1
        ) # (B, T+3, state_dim)
        
        # Reshape to match expected format
        b, f, c = hand_states.shape
        hand_states = hand_states.contiguous().view(b, f // 4, 4, c).transpose(2, 3)
        hand_states = hand_states.contiguous().view(b, f // 4, c * 4)  # (B, T//4, c*4)

        batch.hand_states = hand_states
        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify hand control stage inputs."""
        result = VerificationResult()
        control_hand_state_path = getattr(batch, 'control_hand_state_path', None)
        
        # Hand control is optional
        if control_hand_state_path is not None:
            result.add_check("control_hand_state_path", control_hand_state_path,
                           lambda x: x is not None and isinstance(x, str))
        
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify hand control stage outputs."""
        result = VerificationResult()
        hand_states = getattr(batch, 'hand_states', None)
        
        # If hand control was requested, verify output exists
        if hand_states is not None:
            result.add_check("hand_states", getattr(batch, 'hand_states', None), V.not_none)
        
        return result

