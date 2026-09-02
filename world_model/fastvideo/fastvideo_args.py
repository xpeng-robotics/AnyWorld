# SPDX-License-Identifier: Apache-2.0
# Modified for the AnyWorld public code release.
# Inspired by SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py
"""The arguments of FastVideo Inference."""
import argparse
import dataclasses
import json
from contextlib import contextmanager
from dataclasses import field
from enum import Enum
from typing import Any, TYPE_CHECKING

from fastvideo.configs.pipelines.base import PipelineConfig, STA_Mode
from fastvideo.configs.utils import clean_cli_args
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser, StoreBoolean

if TYPE_CHECKING:
    from ray.runtime_env import RuntimeEnv
    from ray.util.placement_group import PlacementGroup
else:
    RuntimeEnv = Any
    PlacementGroup = Any

logger = init_logger(__name__)


class ExecutionMode(str, Enum):
    """
    Enumeration for different pipeline modes.
    
    Inherits from str to allow string comparison for backward compatibility.
    """
    INFERENCE = "inference"

    @classmethod
    def from_string(cls, value: str) -> "ExecutionMode":
        """Convert string to ExecutionMode enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid mode: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [mode.value for mode in cls]


class WorkloadType(str, Enum):
    """
    Enumeration for different workload types.
    
    Inherits from str to allow string comparison for backward compatibility.
    """
    I2V = "i2v"  # Image to Video
    T2V = "t2v"  # Text to Video
    T2I = "t2i"  # Text to Image
    I2I = "i2i"  # Image to Image
    I2V_COMBINED_CONTROL = "i2v_combined_control"  # Image to Video with Combined Control

    @classmethod
    def from_string(cls, value: str) -> "WorkloadType":
        """Convert string to WorkloadType enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid workload type: {value}. Must be one of: {', '.join([m.value for m in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings for argparse."""
        return [workload.value for workload in cls]


# args for fastvideo framework
@dataclasses.dataclass
class FastVideoArgs:
    # Model and path configuration (for convenience)
    model_path: str

    # Running mode
    mode: ExecutionMode = ExecutionMode.INFERENCE

    # Workload type
    workload_type: WorkloadType = WorkloadType.T2V

    # Cache strategy
    cache_strategy: str = "none"

    # Distributed executor backend
    distributed_executor_backend: str = "mp"

    # a few attributes for ray related
    ray_placement_group: PlacementGroup | None = None

    ray_runtime_env: RuntimeEnv | None = None

    inference_mode: bool = True

    # HuggingFace specific parameters
    trust_remote_code: bool = False
    revision: str | None = None

    # Parallelism
    num_gpus: int = 1
    tp_size: int = -1
    sp_size: int = -1
    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1
    dist_timeout: int | None = None  # timeout for torch.distributed

    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    # LoRA parameters
    # (Wenxuan) prefer to keep it here instead of in pipeline config to not make it complicated.
    lora_path: str | None = None
    lora_nickname: str = "default"  # for swapping adapters in the pipeline
    # can restrict layers to adapt, e.g. ["q_proj"]
    # Will adapt only q, k, v, o by default.
    lora_target_modules: list[str] | None = None

    output_type: str = "pil"

    # CPU offload parameters
    dit_cpu_offload: bool = True
    use_fsdp_inference: bool = True
    text_encoder_cpu_offload: bool = True
    image_encoder_cpu_offload: bool = True
    vae_cpu_offload: bool = True
    pin_cpu_memory: bool = True

    # STA (Sliding Tile Attention) parameters
    mask_strategy_file_path: str | None = None
    STA_mode: STA_Mode = STA_Mode.STA_INFERENCE
    skip_time_steps: int = 15

    # Compilation
    enable_torch_compile: bool = False
    torch_compile_kwargs: dict[str, Any] = field(default_factory=dict)

    disable_autocast: bool = False

    # VSA parameters
    VSA_sparsity: float = 0.0  # inference/validation sparsity

    # V-MoBA parameters
    moba_config_path: str | None = None
    moba_config: dict[str, Any] = field(default_factory=dict)

    # Master port for distributed inference
    master_port: int | None = None

    # Stage verification
    enable_stage_verification: bool = True

    # Prompt text file for batch processing
    prompt_txt: str | None = None

    # model paths for correct deallocation
    model_paths: dict[str, str] = field(default_factory=dict)
    model_loaded: dict[str, bool] = field(default_factory=lambda: {
        "transformer": True,
        "vae": True,
    })
    override_transformer_cls_name: str | None = None
    init_weights_from_safetensors: str = ""  # path to safetensors file for initial weight loading
    init_weights_from_safetensors_2: str = ""  # path to safetensors file for initial weight loading for transformer_2

    # # DMD parameters
    # dmd_denoising_steps: List[int] | None = field(default=None)

    # MoE parameters used by Wan2.2
    boundary_ratio: float | None = 0.875

    @property
    def training_mode(self) -> bool:
        return not self.inference_mode

    def __post_init__(self):
        if self.moba_config_path:
            try:
                with open(self.moba_config_path) as f:
                    self.moba_config = json.load(f)
                logger.info("Loaded V-MoBA config from %s",
                            self.moba_config_path)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error("Failed to load V-MoBA config from %s: %s",
                             self.moba_config_path, e)
                raise
        self.check_fastvideo_args()

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        # Model and path configuration
        parser.add_argument(
            "--model-path",
            type=str,
            help=
            "The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--model-dir",
            type=str,
            help="Directory containing StepVideo model",
        )

        # Running mode
        parser.add_argument(
            "--mode",
            type=str,
            choices=ExecutionMode.choices(),
            default=FastVideoArgs.mode.value,
            help="The mode to run FastVideo",
        )

        # Workload type
        parser.add_argument(
            "--workload-type",
            type=str,
            choices=WorkloadType.choices(),
            default=FastVideoArgs.workload_type.value,
            help="The workload type",
        )

        # distributed_executor_backend
        parser.add_argument(
            "--distributed-executor-backend",
            type=str,
            choices=["mp"],
            default=FastVideoArgs.distributed_executor_backend,
            help="The distributed executor backend to use",
        )

        parser.add_argument(
            "--inference-mode",
            action=StoreBoolean,
            default=FastVideoArgs.inference_mode,
            help="Whether to use inference mode",
        )

        # HuggingFace specific parameters
        parser.add_argument(
            "--trust-remote-code",
            action=StoreBoolean,
            default=FastVideoArgs.trust_remote_code,
            help="Trust remote code when loading HuggingFace models",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=FastVideoArgs.revision,
            help=
            "The specific model version to use (can be a branch name, tag name, or commit id)",
        )

        # Parallelism
        parser.add_argument(
            "--num-gpus",
            type=int,
            default=FastVideoArgs.num_gpus,
            help="The number of GPUs to use.",
        )
        parser.add_argument(
            "--tp-size",
            type=int,
            default=FastVideoArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--sp-size",
            type=int,
            default=FastVideoArgs.sp_size,
            help="The sequence parallelism size.",
        )
        parser.add_argument(
            "--hsdp-replicate-dim",
            type=int,
            default=FastVideoArgs.hsdp_replicate_dim,
            help="The data parallelism size.",
        )
        parser.add_argument(
            "--hsdp-shard-dim",
            type=int,
            default=FastVideoArgs.hsdp_shard_dim,
            help="The data parallelism shards.",
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=FastVideoArgs.dist_timeout,
            help="Set timeout for torch.distributed initialization.",
        )

        # Output type
        parser.add_argument(
            "--output-type",
            type=str,
            default=FastVideoArgs.output_type,
            choices=["pil"],
            help="Output type for the generated video",
        )

        # Prompt text file for batch processing
        parser.add_argument(
            "--prompt-txt",
            type=str,
            default=FastVideoArgs.prompt_txt,
            help=
            "Path to a text file containing prompts (one per line) for batch processing",
        )

        # STA (Sliding Tile Attention) parameters
        parser.add_argument(
            "--STA-mode",
            type=str,
            default=FastVideoArgs.STA_mode.value,
            choices=[mode.value for mode in STA_Mode],
            help=
            "STA mode contains STA_inference, STA_searching, STA_tuning, STA_tuning_cfg, None",
        )
        parser.add_argument(
            "--skip-time-steps",
            type=int,
            default=FastVideoArgs.skip_time_steps,
            help="Number of time steps to warmup (full attention) for STA",
        )
        parser.add_argument(
            "--mask-strategy-file-path",
            type=str,
            help="Path to mask strategy JSON file for STA",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action=StoreBoolean,
            default=FastVideoArgs.enable_torch_compile,
            help="Use torch.compile to speed up DiT inference." +
            "However, will likely cause precision drifts. See (https://github.com/pytorch/pytorch/issues/145213)",
        )
        parser.add_argument(
            "--torch-compile-kwargs",
            type=str,
            default=None,
            help=
            "JSON string of kwargs to pass to torch.compile. Example: '{\"backend\":\"inductor\",\"mode\":\"reduce-overhead\"}'",
        )

        parser.add_argument(
            "--dit-cpu-offload",
            action=StoreBoolean,
            help=
            "Use CPU offload for DiT inference. Enable if run out of memory with FSDP.",
        )
        parser.add_argument(
            "--use-fsdp-inference",
            action=StoreBoolean,
            help=
            "Use FSDP for inference by sharding the model weights. Latency is very low due to prefetch--enable if run out of memory.",
        )
        parser.add_argument(
            "--text-encoder-cpu-offload",
            action=StoreBoolean,
            help=
            "Use CPU offload for text encoder. Enable if run out of memory.",
        )
        parser.add_argument(
            "--image-encoder-cpu-offload",
            action=StoreBoolean,
            help=
            "Use CPU offload for image encoder. Enable if run out of memory.",
        )
        parser.add_argument(
            "--vae-cpu-offload",
            action=StoreBoolean,
            help="Use CPU offload for VAE. Enable if run out of memory.",
        )
        parser.add_argument(
            "--pin-cpu-memory",
            action=StoreBoolean,
            help=
            "Pin memory for CPU offload. Only added as a temp workaround if it throws \"CUDA error: invalid argument\". "
            "Should be enabled in almost all cases",
        )
        parser.add_argument(
            "--disable-autocast",
            action=StoreBoolean,
            help=
            "Disable autocast for denoising loop and vae decoding in pipeline sampling",
        )

        # VSA parameters
        parser.add_argument(
            "--VSA-sparsity",
            type=float,
            default=FastVideoArgs.VSA_sparsity,
            help="Validation sparsity for VSA",
        )

        # Master port for distributed inference
        parser.add_argument(
            "--master-port",
            type=int,
            default=FastVideoArgs.master_port,
            help="Master port for distributed inference",
        )

        # Stage verification
        parser.add_argument(
            "--enable-stage-verification",
            action=StoreBoolean,
            default=FastVideoArgs.enable_stage_verification,
            help="Enable input/output verification for pipeline stages",
        )
        parser.add_argument(
            "--override-transformer-cls-name",
            type=str,
            default=FastVideoArgs.override_transformer_cls_name,
            help="Override transformer cls name",
        )
        parser.add_argument(
            "--init-weights-from-safetensors",
            type=str,
            help="Path to safetensors file for initial weight loading")
        parser.add_argument(
            "--init-weights-from-safetensors-2",
            type=str,
            help="Path to safetensors file for initial weight loading")

        # Add pipeline configuration arguments
        PipelineConfig.add_cli_args(parser)

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "FastVideoArgs":
        provided_args = clean_cli_args(args)
        # Get all fields from the dataclass
        attrs = [attr.name for attr in dataclasses.fields(cls)]

        # Create a dictionary of attribute values, with defaults for missing attributes
        kwargs: dict[str, Any] = {}
        for attr in attrs:
            if attr == 'pipeline_config':
                pipeline_config = PipelineConfig.from_kwargs(provided_args)
                kwargs['pipeline_config'] = pipeline_config
            elif attr == 'mode':
                # Convert string to ExecutionMode enum
                mode_value = getattr(args, attr, FastVideoArgs.mode.value)
                kwargs['mode'] = ExecutionMode.from_string(
                    mode_value) if isinstance(mode_value, str) else mode_value
            elif attr == 'torch_compile_kwargs':
                # Parse JSON string for torch.compile kwargs
                torch_compile_kwargs_str = getattr(args, 'torch_compile_kwargs',
                                                   None)
                if torch_compile_kwargs_str:
                    try:
                        import json
                        kwargs['torch_compile_kwargs'] = json.loads(
                            torch_compile_kwargs_str)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON for torch_compile_kwargs: {e}"
                        ) from e
                else:
                    kwargs['torch_compile_kwargs'] = {}
            elif attr == 'workload_type':
                # Convert string to WorkloadType enum
                workload_type_value = getattr(args, 'workload_type',
                                              FastVideoArgs.workload_type.value)
                kwargs['workload_type'] = WorkloadType.from_string(
                    workload_type_value) if isinstance(
                        workload_type_value, str) else workload_type_value
            # Use getattr with default value from the dataclass for potentially missing attributes
            else:
                # Get the field to check if it has a default_factory
                field = dataclasses.fields(cls)[next(
                    i for i, f in enumerate(dataclasses.fields(cls))
                    if f.name == attr)]
                if field.default_factory is not dataclasses.MISSING:
                    # Use the default_factory to create the default value
                    default_value = field.default_factory()
                else:
                    default_value = getattr(cls, attr, None)
                value = getattr(args, attr, default_value)
                kwargs[attr] = value  # type: ignore

        return cls(**kwargs)  # type: ignore

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "FastVideoArgs":
        # Convert mode string to enum if necessary
        if 'mode' in kwargs and isinstance(kwargs['mode'], str):
            kwargs['mode'] = ExecutionMode.from_string(kwargs['mode'])

        # Convert workload_type string to enum if necessary
        if 'workload_type' in kwargs and isinstance(kwargs['workload_type'],
                                                    str):
            kwargs['workload_type'] = WorkloadType.from_string(
                kwargs['workload_type'])

        kwargs['pipeline_config'] = PipelineConfig.from_kwargs(kwargs)
        return cls(**kwargs)

    def check_fastvideo_args(self) -> None:
        """Validate inference arguments for consistency"""
        from fastvideo.platforms import current_platform

        if current_platform.is_mps():
            self.use_fsdp_inference = False

        if self.mode is not ExecutionMode.INFERENCE or not self.inference_mode:
            raise ValueError("Unsupported configuration for the public inference interface")

        if self.tp_size == -1:
            self.tp_size = 1
        if self.sp_size == -1:
            self.sp_size = self.num_gpus
        if self.hsdp_shard_dim == -1:
            self.hsdp_shard_dim = self.num_gpus

        assert self.sp_size <= self.num_gpus and self.num_gpus % self.sp_size == 0, "num_gpus must >= and be divisible by sp_size"
        assert self.hsdp_replicate_dim <= self.num_gpus and self.num_gpus % self.hsdp_replicate_dim == 0, "num_gpus must >= and be divisible by hsdp_replicate_dim"
        assert self.hsdp_shard_dim <= self.num_gpus and self.num_gpus % self.hsdp_shard_dim == 0, "num_gpus must >= and be divisible by hsdp_shard_dim"

        if self.num_gpus < max(self.tp_size, self.sp_size):
            self.num_gpus = max(self.tp_size, self.sp_size)

        if self.pipeline_config is None:
            raise ValueError("pipeline_config is not set in FastVideoArgs")

        self.pipeline_config.check_pipeline_config()



_current_fastvideo_args = None


def prepare_fastvideo_args(argv: list[str]) -> FastVideoArgs:
    """
    Prepare the inference arguments from the command line arguments.

    Args:
        argv: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The inference arguments.
    """
    parser = FlexibleArgumentParser()
    FastVideoArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    fastvideo_args = FastVideoArgs.from_cli_args(raw_args)
    global _current_fastvideo_args
    _current_fastvideo_args = fastvideo_args
    return fastvideo_args


@contextmanager
def set_current_fastvideo_args(fastvideo_args: FastVideoArgs):
    """
    Temporarily set the current fastvideo config.
    Used during model initialization.
    We save the current fastvideo config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the fastvideo config to determine how to dispatch.
    """
    global _current_fastvideo_args
    old_fastvideo_args = _current_fastvideo_args
    try:
        _current_fastvideo_args = fastvideo_args
        yield
    finally:
        _current_fastvideo_args = old_fastvideo_args


def get_current_fastvideo_args() -> FastVideoArgs:
    if _current_fastvideo_args is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the fastvideo config. In that case, we set a default
        # config.
        # TODO(will): may need to handle this for CI.
        raise ValueError("Current fastvideo args is not set.")
    return _current_fastvideo_args



def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",")]
