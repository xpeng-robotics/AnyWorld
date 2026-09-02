# SPDX-License-Identifier: Apache-2.0
# Modified for the AnyWorld public code release.
"""
Wan video diffusion pipeline implementation with both video control and camera control.

This module contains an implementation of the Wan video diffusion pipeline
using the modular pipeline architecture, supporting both video control and camera control.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline

# isort: off
from fastvideo.pipelines.stages import (
    ConditioningStage, DecodingStage, DenoisingStage,
    ImageVAEEncodingStage, InputValidationStage, LatentPreparationStage,
    TextEncodingStage, TimestepPreparationStage, CameraControlStage, RefImageEncodingStage, VideoControlStage)
# isort: on
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)

logger = init_logger(__name__)


class WanImageToVideoCombinedControlPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    Wan I2V pipeline with both video control and camera control support.
    
    This pipeline combines:
    - Video control: Uses control video to guide video generation (via ref_conv)
    - Camera control: Uses camera control parameters to control camera movement (via control_adapter)
    """

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler", \
        "image_encoder", "image_processor"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        if (self.get_module("image_encoder") is not None
                and self.get_module("image_processor") is not None):
            self.add_stage(
                stage_name="reference_image_encoding_stage",
                stage=RefImageEncodingStage(
                    image_encoder=self.get_module("image_encoder"),
                    image_processor=self.get_module("image_processor"),
                ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer")))

        # Add image latent preparation stage for camera control
        # This is needed when camera control is used
        # if self.get_module("transformer").control_adapter is not None:
        #     self.add_stage(stage_name="image_latent_preparation_stage",
        #                    stage=ImageVAEEncodingStage(vae=self.get_module("vae")))

        # Add camera control stage if control_adapter is available
        self.add_stage(stage_name="camera_control_stage",
                        stage=CameraControlStage(
                            vae=self.get_module("vae"), 
                            transformer=self.get_module("transformer")))

        # Add video control stage if ref_conv is available
        self.add_stage(stage_name="video_control_stage",
                        stage=VideoControlStage(
                            vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           transformer_2=self.get_module("transformer_2"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = WanImageToVideoCombinedControlPipeline

