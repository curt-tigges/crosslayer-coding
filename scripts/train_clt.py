#!/usr/bin/env python3
"""
Script to train a Cross-Layer Transcoder (CLT) using activations from
either a local manifest or a remote server.
Handles configuration parsing from command-line arguments and initiates training.
"""

import argparse
import torch
from pathlib import Path
from typing import Literal, Optional, Dict, Any
import logging
import time
import json

# Attempt to import transformers for model dimension detection
try:
    from transformers import AutoConfig
except ImportError:
    AutoConfig = None

# Import necessary CLT components
try:
    from clt.config import CLTConfig, TrainingConfig
    from clt.training.trainer import CLTTrainer
except ImportError as e:
    print(
        f"FATAL: ImportError: {e}. Please ensure the 'clt' library is installed or "
        "the project root is in your PYTHONPATH."
    )
    raise

# Setup basic logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_model_dimensions(model_name: str) -> tuple[Optional[int], Optional[int]]:
    """Attempt to dynamically get num_layers and d_model from model_name."""
    if AutoConfig is None:
        logger.warning(
            "Transformers library not found. Cannot dynamically detect model dimensions."
            " Falling back to gpt2 defaults (12 layers, 768 hidden size) if not otherwise specified."
            " Install transformers (`pip install transformers`) for auto-detection."
        )
        return None, None  # Indicate failure to auto-detect

    try:
        config = AutoConfig.from_pretrained(model_name)
        num_layers = getattr(config, "num-hidden-layers", None) or getattr(config, "n-layer", None)
        d_model = getattr(config, "hidden-size", None) or getattr(config, "n-embd", None)

        if num_layers is None or d_model is None:
            logger.warning(
                f"Could not automatically determine num_layers or d_model for {model_name}. "
                "Will rely on defaults or error out if not sufficient."
            )
            return None, None
        logger.info(f"Detected model dimensions for {model_name}: {num_layers} layers, {d_model} hidden size.")
        return num_layers, d_model
    except Exception as e:
        logger.warning(
            f"Failed to get model dimensions for {model_name}: {e}. "
            "Will rely on defaults or error out if not sufficient."
        )
        return None, None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Cross-Layer Transcoder (CLT) from local or remote activations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Core Training Parameters ---
    core_group = parser.add_argument_group("Core Training Parameters")
    core_group.add_argument(
        "--activation-source",
        type=str,
        choices=["local_manifest", "remote"],
        required=True,
        help="Source of activations: 'local_manifest' or 'remote' server.",
    )
    core_group.add_argument(
        "--output-dir",
        type=str,
        default=f"clt_train_{int(time.time())}",
        help="Directory to save logs, checkpoints, and final model. If resuming, this might be overridden by --resume-from-checkpoint-dir.",
    )
    core_group.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Base model name or path (e.g., 'gpt2', 'EleutherAI/pythia-70m'). Used for activation generation context and CLT dimension inference.",
    )
    core_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cpu', 'mps'). Auto-detected if None.",
    )
    core_group.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training (requires torchrun/appropriate launcher).",
    )
    core_group.add_argument(
        "--resume-from-checkpoint-dir",
        type=str,
        default=None,
        help="Path to the output directory of a previous run to resume from. Will attempt to load 'latest' or a specific step if --resume-step is also given.",
    )
    core_group.add_argument(
        "--resume-step",
        type=int,
        default=None,
        help="Optional specific step to resume from. Used in conjunction with --resume-from-checkpoint-dir.",
    )

    # --- Local Activation Source Parameters ---
    local_group = parser.add_argument_group(
        "Local Activation Source Parameters (if --activation-source=local_manifest)"
    )
    local_group.add_argument(
        "--activation-path",
        type=str,
        default=None,  # Required if local_manifest, checked in main
        help="Path to the directory containing pre-generated activations (e.g., .../index.bin, metadata.json).",
    )

    # --- Remote Activation Server Parameters ---
    remote_group = parser.add_argument_group("Remote Activation Server Parameters (if --activation-source=remote)")
    remote_group.add_argument(
        "--server-url",
        type=str,
        default=None,  # Required if remote, checked in main
        help="URL of the remote activation storage server (e.g., 'http://localhost:8000').",
    )
    remote_group.add_argument(
        "--dataset-id",
        type=str,
        default=None,  # Required if remote, checked in main
        help="Unique identifier for the dataset on the remote server (e.g., 'gpt2/pile-10k_train').",
    )
    remote_group.add_argument(
        "--remote-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for fetching batches from the remote server.",
    )
    remote_group.add_argument(
        "--remote-max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed batch fetch requests.",
    )
    remote_group.add_argument(
        "--remote-prefetch-batches",
        type=int,
        default=16,  # Default from train_clt_remote
        help="Number of batches to prefetch from the server.",
    )

    # --- CLT Model Architecture (CLTConfig) ---
    clt_group = parser.add_argument_group("CLT Model Architecture (CLTConfig)")
    clt_group.add_argument(
        "--num-features",
        type=int,
        required=True,
        help="Number of features per layer in the CLT.",
    )
    # num_layers and d_model are derived from the base model if not explicitly set
    clt_group.add_argument(
        "--activation-fn",
        type=str,
        choices=["jumprelu", "relu", "batchtopk", "topk"],
        default="jumprelu",
        help="Activation function for the CLT.",
    )
    clt_group.add_argument(
        "--jumprelu-threshold",
        type=float,
        default=0.03,
        help="Threshold for JumpReLU activation (if used).",
    )
    clt_group.add_argument(
        "--batchtopk-k",
        type=int,
        default=None,
        help="Absolute k for BatchTopK activation (if used). Only one of k or frac.",
    )
    clt_group.add_argument(
        "--batchtopk-frac",
        type=float,
        default=None,
        help="Fraction of features to keep for BatchTopK (if used). Only one of k or frac.",
    )
    clt_group.add_argument(
        "--disable-batchtopk-straight-through",
        action="store_true",  # If flag is present, disable is true. Default behavior is enabled.
        help="Disable straight-through estimator for BatchTopK. (BatchTopK default is True).",
    )
    clt_group.add_argument(
        "--topk-k",
        type=float,  # As per CLTConfig, topk_k can be a float (fraction) or int (count)
        default=None,
        help="Number or fraction of features to keep for TopK activation (if used). If < 1, treated as fraction; if >= 1, treated as int count.",
    )
    clt_group.add_argument(
        "--disable-topk-straight-through",
        action="store_true",
        help="Disable straight-through estimator for TopK. (TopK default is True).",
    )
    clt_group.add_argument(
        "--clt-dtype",
        type=str,
        default=None,
        help="Optional data type for the CLT model parameters (e.g., 'float16', 'bfloat16').",
    )

    # --- Training Hyperparameters (TrainingConfig) ---
    train_group = parser.add_argument_group("Training Hyperparameters (TrainingConfig)")
    train_group.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    train_group.add_argument(
        "--training-steps",
        type=int,
        default=50000,
        help="Total number of training steps.",
    )
    train_group.add_argument(
        "--train-batch-size-tokens",
        type=int,
        default=4096,
        help="Target number of tokens per training batch.",
    )
    train_group.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",  # Default to fp32 as per TrainingConfig
        help="Training precision: 'fp32', 'fp16' (mixed precision with AMP), or 'bf16' (mixed precision with AMP).",
    )
    train_group.add_argument(
        "--fp16-convert-weights",
        action="store_true",
        help="If --precision is fp16, also convert model weights to fp16. Saves memory but model parameters remain fp32 by default with AMP. Default is False.",
    )
    train_group.add_argument(
        "--debug-anomaly",
        action="store_true",
        help="Enable PyTorch autograd anomaly detection for debugging NaN issues. Default is False.",
    )
    train_group.add_argument(
        "--normalization-method",
        type=str,
        choices=["auto", "none", "estimated_mean_std"],  # Added estimated_mean_std from TrainingConfig
        default="auto",
        help=(
            "Normalization for activation store. 'auto' expects server/local store to provide stats. "
            "'estimated_mean_std' forces estimation (if store supports it). 'none' disables."
        ),
    )
    train_group.add_argument(
        "--sparsity-lambda",
        type=float,
        default=1e-3,
        help="Coefficient for the L1 sparsity penalty.",
    )
    train_group.add_argument(
        "--sparsity-lambda-schedule",
        type=str,
        choices=["linear", "delayed_linear"],
        default="linear",
        help="Schedule for applying sparsity lambda.",
    )
    train_group.add_argument(
        "--sparsity-lambda-delay-frac",
        type=float,
        default=0.1,
        help="Fraction of steps to delay lambda increase for 'delayed_linear' schedule.",
    )
    train_group.add_argument(
        "--sparsity-c",
        type=float,
        default=1.0,
        help="Constant shaping the sparsity penalty (typically 1.0).",
    )
    train_group.add_argument(
        "--preactivation-coef",
        type=float,
        default=3e-6,
        help="Coefficient for the pre-activation MSE loss term.",
    )
    train_group.add_argument(
        "--aux-loss-factor",
        type=float,
        default=None,
        help="Coefficient for the auxiliary reconstruction loss (e.g., for dead latents). If None, loss is not applied.",
    )
    train_group.add_argument(
        "--apply-sparsity-penalty-to-batchtopk",
        action=argparse.BooleanOptionalAction,  # Allows --apply-sparsity-penalty-to-batchtopk or --no-apply-sparsity-penalty-to-batchtopk
        default=True,  # Matches TrainingConfig default
        help="Apply standard L1 sparsity penalty to BatchTopK activations. Default is True. Use --no-apply-sparsity-penalty-to-batchtopk to disable.",
    )
    train_group.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw"],
        default="adamw",
        help="Optimizer algorithm.",
    )
    train_group.add_argument(
        "--optimizer-beta1",
        type=float,
        default=None,
        help="Optimizer beta1 value (if using Adam/AdamW).",
    )
    train_group.add_argument(
        "--optimizer-beta2",
        type=float,
        default=None,
        help="Optimizer beta2 value (if using Adam/AdamW).",
    )
    train_group.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["linear", "cosine", "linear_final20", "none"],
        default="linear",
        help=(
            "Learning rate scheduler type. 'linear_final20' keeps LR constant until the last 20% "
            "of steps then decays linearly to 0 ('none' to disable)."
        ),
    )
    train_group.add_argument(
        "--gradient-clip-val",
        type=float,
        default=None,
        help="Value for gradient clipping. If None, no clipping.",
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    train_group.add_argument(
        "--activation-dtype",
        type=str,
        default="float32",  # Consistent default
        help="Data type to process/load activations as (e.g., 'float32', 'bfloat16').",
    )
    train_group.add_argument(
        "--dead-feature-window",
        type=int,
        default=1000,
        help="Number of steps of inactivity before a feature is considered 'dead' for evaluation.",
    )
    train_group.add_argument(
        "--compute-sparsity-diagnostics",
        action="store_true",
        help="Enable computation of detailed sparsity diagnostics during evaluation.",
    )

    # --- Sampling Strategy ---
    sampling_group = parser.add_argument_group("Sampling Strategy (TrainingConfig)")
    sampling_group.add_argument(
        "--sampling-strategy",
        type=str,
        choices=["sequential", "random_chunk"],
        default="sequential",
        help="Sampling strategy for manifest-based stores: 'sequential' or 'random_chunk'.",
    )

    # --- Logging & Checkpointing ---
    log_group = parser.add_argument_group("Logging & Checkpointing (TrainingConfig)")
    log_group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log training metrics every N steps.",
    )
    log_group.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Run evaluation metrics computation every N steps.",
    )
    log_group.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save a training checkpoint every N steps.",
    )
    log_group.add_argument("--enable-wandb", action="store_true", help="Enable Weights & Biases logging.")
    log_group.add_argument("--wandb-project", type=str, default=None, help="WandB project name.")
    log_group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity (username or team).",
    )
    log_group.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Custom name for the WandB run. Auto-generated if None.",
    )
    log_group.add_argument("--wandb-tags", nargs="+", default=None, help="List of tags for the WandB run.")

    args = parser.parse_args()

    # --- Validate conditional arguments ---
    if args.activation_source == "remote":
        if not args.server_url:
            parser.error("--server-url is required when --activation-source is 'remote'")
        if not args.dataset_id:
            parser.error("--dataset-id is required when --activation-source is 'remote'")
    elif args.activation_source == "local_manifest":
        if not args.activation_path:
            # Allow activation_path to be None if resuming, as it will be loaded from cli_args.json
            if not args.resume_from_checkpoint_dir:
                parser.error(
                    "--activation-path is required when --activation-source is 'local_manifest' and not resuming."
                )

    return args


def main():
    """Main function to configure and run the CLTTrainer."""
    args = parse_args()

    output_dir_for_trainer_str = args.output_dir
    actual_checkpoint_path_to_load: Optional[str] = None
    resuming_run = False

    if args.resume_from_checkpoint_dir:
        resuming_run = True
        resume_base_dir = Path(args.resume_from_checkpoint_dir)
        logger.info(f"Attempting to resume training from directory: {resume_base_dir}")

        # Override output_dir to be the resume directory
        output_dir_for_trainer_str = str(resume_base_dir.resolve())

        # Load original CLI args from the run being resumed
        original_cli_args_path = resume_base_dir / "cli_args.json"
        if original_cli_args_path.exists():
            logger.info(f"Loading original CLI arguments from {original_cli_args_path}")
            with open(original_cli_args_path, "r") as f:
                original_cli_vars = json.load(f)

            # Create a new argparse.Namespace from the loaded dict
            # Update this new namespace with the original args, then override with any current CLI args
            # that are relevant for resuming (like resume_step, or if user wants to change e.g. training_steps for the resumed run)

            # Start with current args (which include resume_from_checkpoint_dir, resume_step)
            # Then load original args, but current resume-specific args should take precedence if they were specified.
            # Also, things like output_dir might change if we allow resuming to a NEW directory (not supported yet, logs to original)

            current_args_dict = vars(args).copy()
            # args_from_file = argparse.Namespace(**original_cli_vars) # Unused / Can be removed

            # Update args_from_file with any overriding CLI args from current invocation
            # For most params, we want the original run's params. But some (e.g. training_steps) user might want to extend.
            # For now, let's prioritize original CLI args for most things, except for resume flags and potentially output_dir.

            # Convert original_cli_vars to Namespace and then update it with relevant current args.
            # The `args` variable will be rebuilt from original_cli_vars, with care for resume flags.
            temp_args_dict = original_cli_vars.copy()

            # Keep current resume flags and potentially new output_dir if we decide to support it
            # For now, output_dir is forced to be the resume_from_checkpoint_dir
            temp_args_dict["resume_from_checkpoint_dir"] = args.resume_from_checkpoint_dir
            temp_args_dict["resume_step"] = args.resume_step
            temp_args_dict["output_dir"] = (
                output_dir_for_trainer_str  # Ensure output_dir is the one we are resuming into
            )
            # If user wants to override training_steps for a resumed run, they can pass it.
            if current_args_dict.get("training_steps") != original_cli_vars.get("training_steps"):
                logger.info(
                    f"Overriding training_steps from {original_cli_vars.get('training_steps')} to {current_args_dict.get('training_steps')}"
                )
                temp_args_dict["training_steps"] = current_args_dict.get("training_steps")
            # Potentially other overridable args like learning_rate, wandb settings etc.

            args = argparse.Namespace(**temp_args_dict)  # Re-assign args with merged values
            logger.info(f"Effective arguments for resumed run: {vars(args)}")

        else:
            logger.warning(
                f"Original cli_args.json not found at {original_cli_args_path}. "
                f"Configuration will be based on the currently provided command-line arguments. "
                f"Ensure all necessary configuration parameters are supplied."
            )
            # In this case, `args` remains as parsed from the current command line, which is desired.

        # Determine the specific checkpoint path to load (model file or distributed dir)
        # This logic assumes CLTTrainer's load_checkpoint handles whether it's a file or dir based on distributed status
        if args.distributed:
            if args.resume_step is not None:
                actual_checkpoint_path_to_load = str(resume_base_dir / f"step_{args.resume_step}")
            else:
                actual_checkpoint_path_to_load = str(resume_base_dir / "latest")
        else:  # Non-distributed
            if args.resume_step is not None:
                actual_checkpoint_path_to_load = str(resume_base_dir / f"clt_checkpoint_{args.resume_step}.safetensors")
            else:
                actual_checkpoint_path_to_load = str(resume_base_dir / "clt_checkpoint_latest.safetensors")

        if not Path(actual_checkpoint_path_to_load).exists():
            logger.error(f"Checkpoint to load does not exist: {actual_checkpoint_path_to_load}")
            if args.distributed and (
                actual_checkpoint_path_to_load.endswith("latest")
                or actual_checkpoint_path_to_load.endswith(f"step_{args.resume_step}")
            ):
                logger.error("For distributed runs, ensure the directory exists.")
            elif not args.distributed and actual_checkpoint_path_to_load.endswith(".safetensors"):
                logger.error("For non-distributed runs, ensure the .safetensors file exists.")
            return

        logger.info(f"Will attempt to load checkpoint state from: {actual_checkpoint_path_to_load}")

    # --- Setup Output Directory (now based on output_dir_for_trainer_str) ---
    output_dir_path = Path(output_dir_for_trainer_str)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Using output directory: {output_dir_path.resolve()}")

    # Save command-line arguments (only if not resuming, or save effective if resuming?)
    # For now, let's only save if it's a new run, to avoid overwriting original if resuming.
    if not resuming_run:
        try:
            with open(output_dir_path / "cli_args.json", "w") as f:
                json.dump(vars(args), f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save command-line args: {e}")

    # --- Determine Device ---
    if args.device:
        device_str = args.device
    else:
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():  # For Apple Silicon
            device_str = "mps"
        else:
            device_str = "cpu"
    logger.info(f"Using device: {device_str}")
    # Trainer will handle torch.device object creation

    # --- Determine Base Model Dimensions ---
    base_model_name = args.model_name
    num_layers_auto, d_model_auto = get_model_dimensions(base_model_name)
    if num_layers_auto is None or d_model_auto is None:
        # This case implies get_model_dimensions failed or returned Nones.
        # CLTConfig requires num_layers and d_model.
        # If they couldn't be auto-detected, it's a fatal error.
        logger.error(
            f"Could not determine dimensions (num_layers, d_model) for model '{base_model_name}'. "
            "These are required for CLTConfig. Please ensure the model name is correct and visible "
            "to the Hugging Face AutoConfig, or that the CLT library can derive them."
        )
        return  # Exit if dimensions are critical and not found

    # --- Create CLT Configuration ---
    clt_config = CLTConfig(
        num_features=args.num_features,
        num_layers=num_layers_auto,  # d_model and num_layers are now from auto-detection
        d_model=d_model_auto,
        model_name=base_model_name,  # Store for reference
        activation_fn=args.activation_fn,
        jumprelu_threshold=args.jumprelu_threshold,
        batchtopk_k=args.batchtopk_k,
        batchtopk_straight_through=(not args.disable_batchtopk_straight_through),
        clt_dtype=args.clt_dtype,
        topk_k=args.topk_k,
        topk_straight_through=(not args.disable_topk_straight_through),
    )
    logger.info(f"CLT Config: {clt_config}")

    # --- Create Training Configuration ---
    lr_scheduler_arg: Optional[Literal["linear", "cosine", "linear_final20"]] = (
        args.lr_scheduler if args.lr_scheduler != "none" else None
    )

    activation_path_arg: Optional[str] = None
    remote_config_dict: Optional[Dict[str, Any]] = None

    if args.activation_source == "local_manifest":
        activation_path_arg = args.activation_path
        logger.info(f"Using local activation source: {activation_path_arg}")
    elif args.activation_source == "remote":
        remote_config_dict = {
            "server_url": args.server_url,
            "dataset_id": args.dataset_id,
            "timeout": args.remote_timeout,
            "max_retries": args.remote_max_retries,
            "prefetch_batches": args.remote_prefetch_batches,
        }
        logger.info(f"Using remote activation source: {args.server_url}, dataset: {args.dataset_id}")

    # --- Determine WandB Run Name ---
    wandb_run_name_for_config: Optional[str] = None  # Initialize to None

    if resuming_run:
        # If resuming, we prioritize the ID from the checkpoint.
        # WandBLogger will use the ID and resume="must".
        # Explicitly set run name to None to avoid conflicts.
        wandb_run_name_for_config = None
        logger.info("Resuming run: wandb_run_name will be None; WandB ID from checkpoint will be used.")
    else:
        # This is a new run (not resuming)
        wandb_run_name_for_config = args.wandb_run_name
        if (
            not wandb_run_name_for_config and args.enable_wandb
        ):  # Auto-generate if not provided and wandb is enabled for a new run
            name_parts = [f"{args.num_features}-width"]
            if args.activation_fn == "batchtopk":
                name_parts.append("batchtopk")
                if args.batchtopk_k is not None:
                    name_parts.append(f"k{args.batchtopk_k}")
            elif args.activation_fn == "topk":
                name_parts.append("topk")
                if args.topk_k is not None:
                    name_parts.append(f"k{args.topk_k}")
            else:  # jumprelu or relu
                name_parts.append(args.activation_fn)
                name_parts.append(f"{args.sparsity_lambda:.1e}-slambda")
                name_parts.append(f"{args.sparsity_c:.1f}-sc")

            name_parts.append(f"{args.train_batch_size_tokens}-batch")
            name_parts.append(f"{args.learning_rate:.1e}-lr")
            if args.activation_source == "remote" and args.dataset_id:
                # Sanitize dataset_id for use in filename/run name
                sanitized_dataset_id = args.dataset_id.replace("/", "_")
                name_parts.append(f"ds_{sanitized_dataset_id[:20]}")  # Truncate if too long
            elif args.activation_source == "local_manifest" and args.activation_path:
                path_basename = Path(args.activation_path).name
                name_parts.append(f"path_{path_basename[:20]}")

            wandb_run_name_for_config = "-".join(name_parts)
            logger.info(f"Generated WandB run name for new run: {wandb_run_name_for_config}")
    # --- End Determine WandB Run Name ---

    training_config = TrainingConfig(
        # Core Training
        learning_rate=args.learning_rate,
        training_steps=args.training_steps,
        seed=args.seed,
        gradient_clip_val=args.gradient_clip_val,
        train_batch_size_tokens=args.train_batch_size_tokens,
        # Activation Source
        activation_source=args.activation_source,  # Directly from args
        activation_path=activation_path_arg,  # Populated if local
        remote_config=remote_config_dict,  # Populated if remote
        activation_dtype=args.activation_dtype,
        # Normalization
        normalization_method=args.normalization_method,
        # Sampling Strategy
        sampling_strategy=args.sampling_strategy,
        # Loss Coeffs
        sparsity_lambda=args.sparsity_lambda,
        sparsity_lambda_schedule=args.sparsity_lambda_schedule,
        sparsity_lambda_delay_frac=args.sparsity_lambda_delay_frac,
        sparsity_c=args.sparsity_c,
        preactivation_coef=args.preactivation_coef,
        aux_loss_factor=args.aux_loss_factor,
        apply_sparsity_penalty_to_batchtopk=args.apply_sparsity_penalty_to_batchtopk,
        # Optimizer & Scheduler
        optimizer=args.optimizer,
        optimizer_beta1=args.optimizer_beta1,
        optimizer_beta2=args.optimizer_beta2,
        lr_scheduler=lr_scheduler_arg,
        # Logging & Checkpointing
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        # Dead Features & Diagnostics
        dead_feature_window=args.dead_feature_window,
        compute_sparsity_diagnostics=args.compute_sparsity_diagnostics,
        # WandB
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=wandb_run_name_for_config,  # Use the decision from above
        wandb_tags=args.wandb_tags,
        # Precision & Debugging
        precision=args.precision,
        debug_anomaly=args.debug_anomaly,
        fp16_convert_weights=args.fp16_convert_weights,
    )
    logger.info(f"Training Config: {training_config}")

    # --- Initialize Trainer ---
    logger.info(f"Initializing CLTTrainer for {args.activation_source} training...")
    try:
        trainer = CLTTrainer(
            clt_config=clt_config,
            training_config=training_config,
            log_dir=str(output_dir_path),  # Use the resolved output_dir_path
            device=device_str,
            distributed=args.distributed,
            resume_from_checkpoint_path=actual_checkpoint_path_to_load if resuming_run else None,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize CLTTrainer: {e}")
        raise

    # --- Start Training ---
    if args.activation_source == "remote":
        logger.info(f"Starting training from remote server {args.server_url} using dataset {args.dataset_id}...")
    else:  # local_manifest
        logger.info(f"Starting training from local activations at {args.activation_path}...")

    try:
        trainer.train()  # eval_every is handled by eval_interval in TrainingConfig
        logger.info("Training complete!")
        logger.info(f"Final model and logs saved in: {output_dir_path.resolve()}")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
