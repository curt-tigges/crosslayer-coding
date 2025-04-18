from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any


@dataclass
class ActivationConfig:
    """Configuration for generating or locating activation datasets."""

    # --- Model Source Identification ---
    model_name: str  # Name or path of the Hugging Face transformer model
    mlp_input_module_path_template: str  # NNsight path template for MLP inputs
    mlp_output_module_path_template: str  # NNsight path template for MLP outputs
    # --- Dataset Source Identification ---
    dataset_path: str  # Path or name of the Hugging Face dataset
    # --- Fields with Defaults --- #
    model_dtype: Optional[str] = (
        None  # Optional dtype for the model ('float16', 'bfloat16')
    )
    activation_dtype: Literal["bfloat16", "float16", "float32"] = (
        "bfloat16"  # Precision for storing activations
    )
    dataset_split: str = "train"  # Dataset split to use
    dataset_text_column: str = "text"  # Column containing text data

    # --- Generation Parameters ---
    context_size: int = 128  # Max sequence length for tokenization/inference
    inference_batch_size: int = 512  # Batch size for model inference during generation
    exclude_special_tokens: bool = True  # Exclude special tokens during generation
    prepend_bos: bool = False  # Prepend BOS token during generation

    # --- Dataset Handling Parameters (for generation) ---
    streaming: bool = True  # Use HF dataset streaming during generation
    dataset_trust_remote_code: bool = False  # Trust remote code for HF dataset
    cache_path: Optional[str] = (
        None  # Optional cache path for HF dataset (if not streaming)
    )

    # --- Generation Output Control ---
    target_total_tokens: Optional[int] = (
        None  # Target num tokens to generate (approximate)
    )

    # --- Storage Parameters (for generation output) ---
    activation_dir: str = "./activations"  # Base directory to save activation datasets
    output_format: Literal["hdf5", "npz"] = "hdf5"  # Format to save activations
    compression: Optional[str] = (
        "gzip"  # Compression for saved files ('lz4', 'gzip', None)
    )
    chunk_token_threshold: int = (
        1_000_000  # Minimum tokens to accumulate before saving a chunk
    )
    # Note: 'storage_type' (local/remote) is handled by the generator script/workflow,
    # not stored intrinsically here, as the generated data itself is local.

    # --- Normalization Computation (during generation) ---
    compute_norm_stats: bool = (
        True  # Compute mean/std during generation and save to norm_stats.json
    )

    # --- Remote Storage Parameters ---
    remote_server_url: Optional[str] = None  # Base URL of the remote activation server
    delete_after_upload: bool = False  # Delete local chunk after successful upload

    # --- NNsight Parameters (Optional) ---
    # Use field to allow mutable default dict
    nnsight_tracer_kwargs: Dict[str, Any] = field(default_factory=dict)
    nnsight_invoker_args: Dict[str, Any] = field(default_factory=dict)

    # --- Device Parameter ---
    # While generation happens on a device, this config is more about the data itself.
    # The device used for generation can be passed separately to the generator instance.
    # device: Optional[str] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.context_size > 0, "Context size must be positive"
        assert self.inference_batch_size > 0, "Inference batch size must be positive"
        assert self.chunk_token_threshold > 0, "Chunk token threshold must be positive"
        if self.output_format == "hdf5":
            try:
                import h5py  # noqa: F401 - Check if h5py is available if format is hdf5
            except ImportError:
                raise ImportError(
                    "h5py is required for HDF5 output format. Install with: pip install h5py"
                )
        if self.compression not in ["lz4", "gzip", None, False]:
            print(
                f"Warning: Unsupported compression '{self.compression}'. Will attempt without compression for {self.output_format}."
            )
            # Allow generator to handle disabling if format doesn't support it.
