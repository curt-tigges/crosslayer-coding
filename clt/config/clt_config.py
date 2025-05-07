import json
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, TypeVar, Type


# Generic type for Config subclasses
C = TypeVar("C", bound="CLTConfig")


@dataclass
class CLTConfig:
    """Configuration for a Cross-Layer Transcoder."""

    num_features: int  # Number of features per layer
    num_layers: int  # Number of transformer layers
    d_model: int  # Dimension of model's hidden state
    model_name: Optional[str] = None  # Optional name for the underlying model
    normalization_method: Literal["auto", "estimated_mean_std", "none"] = (
        "none"  # How activations were normalized during training
    )
    activation_fn: Literal["jumprelu", "relu", "batchtopk"] = "jumprelu"
    jumprelu_threshold: float = 0.03  # Threshold for JumpReLU activation
    # BatchTopK parameters
    batchtopk_k: Optional[int] = None  # Absolute k for BatchTopK
    batchtopk_frac: Optional[float] = None  # Fraction of features to keep for BatchTopK
    batchtopk_straight_through: bool = True  # Whether to use straight-through estimator for BatchTopK
    clt_dtype: Optional[str] = None  # Optional dtype for the CLT model itself (e.g., "float16")
    expected_input_dtype: Optional[str] = None  # Expected dtype of input activations
    mlp_input_template: Optional[str] = None  # Module path template for MLP input activations
    mlp_output_template: Optional[str] = None  # Module path template for MLP output activations
    tl_input_template: Optional[str] = None  # TransformerLens hook point pattern before MLP
    tl_output_template: Optional[str] = None  # TransformerLens hook point pattern after MLP

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.num_features > 0, "Number of features must be positive"
        assert self.num_layers > 0, "Number of layers must be positive"
        assert self.d_model > 0, "Model dimension must be positive"
        assert self.jumprelu_threshold > 0, "JumpReLU threshold must be positive"
        valid_norm_methods = ["auto", "estimated_mean_std", "none"]
        assert (
            self.normalization_method in valid_norm_methods
        ), f"Invalid normalization_method: {self.normalization_method}. Must be one of {valid_norm_methods}"
        valid_activation_fns = ["jumprelu", "relu", "batchtopk"]
        assert (
            self.activation_fn in valid_activation_fns
        ), f"Invalid activation_fn: {self.activation_fn}. Must be one of {valid_activation_fns}"

        if self.activation_fn == "batchtopk":
            if self.batchtopk_k is not None and self.batchtopk_frac is not None:
                raise ValueError("Only one of batchtopk_k or batchtopk_frac can be specified.")
            if self.batchtopk_k is None and self.batchtopk_frac is None:
                raise ValueError("One of batchtopk_k or batchtopk_frac must be specified for BatchTopK.")
            if self.batchtopk_k is not None and self.batchtopk_k <= 0:
                raise ValueError("batchtopk_k must be positive.")
            if self.batchtopk_frac is not None and not (0 < self.batchtopk_frac <= 1):
                raise ValueError("batchtopk_frac must be between 0 (exclusive) and 1 (inclusive).")

    @classmethod
    def from_json(cls: Type[C], json_path: str) -> C:
        """Load configuration from a JSON file.

        Args:
            json_path: Path to the JSON configuration file.

        Returns:
            An instance of the configuration class.
        """
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_json(self, json_path: str) -> None:
        """Save configuration to a JSON file.

        Args:
            json_path: Path to save the JSON configuration file.
        """
        config_dict = self.__dict__
        with open(json_path, "w") as f:
            json.dump(config_dict, f, indent=4)


@dataclass
class TrainingConfig:
    """Configuration for training a Cross-Layer Transcoder."""

    # Basic training parameters
    learning_rate: float  # Learning rate for optimizer
    training_steps: int  # Total number of training steps
    seed: int = 42
    gradient_clip_val: Optional[float] = None  # Gradient clipping value
    # Training batch size (tokens)
    train_batch_size_tokens: int = 4096  # Number of tokens per training step batch
    # Buffer size for streaming store
    n_batches_in_buffer: int = 16  # Number of extraction batches in buffer

    # Normalization parameters
    normalization_method: Literal["auto", "estimated_mean_std", "none"] = "auto"
    # 'auto': Use pre-calculated from mapped store, or estimate for streaming store.
    # 'estimated_mean_std': Always estimate for streaming store (ignored for mapped).
    # 'none': Disable normalization.
    normalization_estimation_batches: int = 50  # Batches for normalization estimation

    # --- Activation Store Source --- #
    activation_source: Literal["generate", "local_manifest", "remote"] = "generate"
    activation_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    # Config for "generate" source (on-the-fly)
    generation_config: Optional[Dict[str, Any]] = None  # Dict matching ActivationConfig fields needed for extractor
    dataset_params: Optional[Dict[str, Any]] = None  # Dict matching dataset fields for stream_activations
    # Config for "local_manifest" source (pre-generated with manifest)
    activation_path: Optional[str] = None  # Path to pre-generated activation dataset directory (containing index.bin)
    # Config for "remote" source
    remote_config: Optional[Dict[str, Any]] = None  # Dict with server_url, dataset_id, etc.
    # --- End Activation Store Source --- #

    # Sampling strategy for manifest-based stores
    sampling_strategy: Literal["sequential", "random_chunk"] = "sequential"

    # Loss function coefficients
    sparsity_lambda: float = 1e-3  # Coefficient for sparsity penalty
    # Sparsity schedule: \'linear\' scales lambda from 0 to max over all steps.
    # \'delayed_linear\' keeps lambda at 0 for `delay_frac` steps, then scales linearly.
    sparsity_lambda_schedule: Literal["linear", "delayed_linear"] = "linear"
    sparsity_lambda_delay_frac: float = (
        0.1  # Fraction of steps to delay lambda increase (if schedule is delayed_linear)
    )
    sparsity_c: float = 1.0  # Parameter affecting sparsity penalty shape
    preactivation_coef: float = 3e-6  # Coefficient for pre-activation loss
    aux_loss_factor: Optional[float] = None  # Coefficient for the auxiliary reconstruction loss (e.g. for dead latents)
    apply_sparsity_penalty_to_batchtopk: bool = True  # Whether to apply sparsity penalty when using BatchTopK

    # Optimizer parameters
    optimizer: Literal["adam", "adamw"] = "adamw"
    optimizer_beta1: Optional[float] = None  # Beta1 for Adam/AdamW (default: 0.9)
    optimizer_beta2: Optional[float] = None  # Beta2 for Adam/AdamW (default: 0.999)
    # Learning rate scheduler type. "linear_final20" keeps LR constant for the first 80% of
    # training and then linearly decays it to 0 for the final 20% (configurable via lr_scheduler_params).
    lr_scheduler: Optional[Literal["linear", "cosine", "linear_final20"]] = "linear"
    lr_scheduler_params: Optional[Dict[str, Any]] = None

    # Logging parameters
    log_interval: int = 100  # How often to log metrics
    eval_interval: int = 1000  # How often to run evaluation
    checkpoint_interval: int = 1000  # How often to save checkpoints

    # Optional diagnostic metrics (can be slow)
    compute_sparsity_diagnostics: bool = False  # Whether to compute detailed sparsity diagnostics during eval

    # Dead feature tracking
    dead_feature_window: int = 1000  # Steps until a feature is considered dead

    # WandB logging configuration
    enable_wandb: bool = False  # Whether to use Weights & Biases logging
    wandb_project: Optional[str] = None  # WandB project name
    wandb_entity: Optional[str] = None  # WandB entity/organization name
    wandb_run_name: Optional[str] = None  # WandB run name, defaults to timestamp if None
    wandb_tags: Optional[list] = field(default_factory=list)

    def __post_init__(self):
        """Validate training parameters."""
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.training_steps > 0, "Training steps must be positive"
        assert self.train_batch_size_tokens > 0, "Training batch size (tokens) must be positive"
        assert self.n_batches_in_buffer > 0, "Buffer size must be positive"
        assert self.sparsity_lambda >= 0, "Sparsity lambda must be non-negative"
        assert self.dead_feature_window > 0, "Dead feature window must be positive"

        # Validate activation source configuration
        if self.activation_source == "generate":
            assert (
                self.generation_config is not None
            ), "generation_config dict must be provided when activation_source is 'generate'"
            assert (
                self.dataset_params is not None
            ), "dataset_params dict must be provided when activation_source is 'generate'"
            # Basic check for essential keys in the dicts (can be expanded)
            assert "model_name" in self.generation_config, "generation_config missing 'model_name'"
            assert "dataset_path" in self.dataset_params, "dataset_params missing 'dataset_path'"
        elif self.activation_source == "local_manifest":
            assert (
                self.activation_path is not None
            ), "activation_path must be specified when activation_source is 'local_manifest'"
        elif self.activation_source == "remote":
            assert (
                self.remote_config is not None
            ), "remote_config dict must be provided when activation_source is 'remote'"
            assert (
                "server_url" in self.remote_config and "dataset_id" in self.remote_config
            ), "remote_config must contain 'server_url' and 'dataset_id'"

        # Validate sampling strategy
        assert self.sampling_strategy in [
            "sequential",
            "random_chunk",
        ], "sampling_strategy must be 'sequential' or 'random_chunk'"

        # Validate sparsity schedule params
        assert self.sparsity_lambda_schedule in ["linear", "delayed_linear"], "Invalid sparsity_lambda_schedule"
        if self.sparsity_lambda_schedule == "delayed_linear":
            assert (
                0.0 <= self.sparsity_lambda_delay_frac < 1.0
            ), "sparsity_lambda_delay_frac must be between 0.0 (inclusive) and 1.0 (exclusive)"
