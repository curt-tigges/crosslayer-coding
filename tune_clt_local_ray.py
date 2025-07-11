#!/usr/bin/env python3
"""
Script to train a Cross-Layer Transcoder (CLT) using pre-generated local activations.
Handles configuration parsing from command-line arguments and initiates training.
"""

import argparse
import torch
from pathlib import Path
from typing import Literal, Optional
import logging
import time
import json
import ray, ray.tune

from clt.training.trainer_ray import CLTTrainerRay

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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_model_dimensions(model_name: str) -> tuple[int, int]:
    """Attempt to dynamically get num_layers and d_model from model_name."""
    if AutoConfig is None:
        logger.warning(
            "Transformers library not found. Cannot dynamically detect model dimensions."
            " Falling back to gpt2 defaults (12 layers, 768 hidden size)."
            " Install transformers (`pip install transformers`) for auto-detection."
        )
        return 12, 768  # Default to gpt2 small

    try:
        config = AutoConfig.from_pretrained(model_name)
        num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
        d_model = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)

        if num_layers is None or d_model is None:
            raise ValueError(f"Could not automatically determine num_layers or d_model for {model_name}")
        logger.info(f"Detected model dimensions for {model_name}: {num_layers} layers, {d_model} hidden size.")
        return num_layers, d_model
    except Exception as e:
        logger.warning(
            f"Failed to get model dimensions for {model_name}: {e}. "
            f"Falling back to gpt2 defaults (12 layers, 768 hidden size)."
        )
        return 12, 768


def train_loop_per_worker(cfg):

    # --- Determine Device ---
    if cfg['device']:
        device = cfg['device']
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")

    # --- Determine Base Model Dimensions ---
    # Use the provided --model-name to get dimensions for the CLT config.
    # This ensures the CLT matches the architecture activations were generated from.
    base_model_name = cfg['model_name']
    num_layers, d_model = get_model_dimensions(base_model_name)
    if num_layers is None or d_model is None:
        # Added error handling if dimensions couldn't be determined
        logger.error(f"Could not determine dimensions for model '{base_model_name}'. Exiting.")
        return

    # --- Create CLT Configuration ---
    clt_config = CLTConfig(
        num_features=cfg['num_features'],
        num_layers=num_layers,
        d_model=d_model,
        activation_fn=cfg['activation_fn'],
        jumprelu_threshold=cfg['jumprelu_threshold'],
        clt_dtype=cfg['clt_dtype'],
    )
    logger.info(f"CLT Config: {clt_config}")

    # --- Create Training Configuration ---
    # Handle 'none' scheduler case
    lr_scheduler_arg: Optional[Literal["linear", "cosine", "linear_final20"]] = (
        cfg['lr_scheduler'] if cfg['lr_scheduler'] != "none" else None
    )

    # Simplified TrainingConfig instantiation for local source only
    training_config = TrainingConfig(
        # Core Training
        learning_rate=cfg['learning_rate'],
        training_steps=cfg['training_steps'],
        seed=cfg['seed'],
        train_batch_size_tokens=cfg['train_batch_size_tokens'],
        # Activation Source (hardcoded to local_manifest)
        activation_source="local_manifest",
        activation_path=cfg['activation_path'],
        activation_dtype=cfg['activation_dtype'],
        # Normalization
        normalization_method=cfg['normalization_method'],
        # Sampling Strategy
        sampling_strategy=cfg['sampling_strategy'],
        # Loss Coeffs
        sparsity_lambda=cfg['sparsity_lambda'],
        sparsity_c=cfg['sparsity_c'],
        preactivation_coef=cfg['preactivation_coef'],
        # Optimizer & Scheduler
        optimizer=cfg['optimizer'],
        optimizer_beta1=cfg['optimizer_beta1'],
        optimizer_beta2=cfg['optimizer_beta2'],
        lr_scheduler=lr_scheduler_arg,
        # Logging & Checkpointing
        log_interval=cfg['log_interval'],
        eval_interval=cfg['eval_interval'],
        checkpoint_interval=cfg['checkpoint_interval'],
        # Dead Features
        dead_feature_window=cfg['dead_feature_window'],
        # WandB
        enable_wandb=cfg['enable_wandb'],
        wandb_project=cfg['wandb_project'],
        wandb_entity=cfg['wandb_entity'],
        wandb_run_name=cfg['wandb_run_name'],
        wandb_tags=cfg['wandb_tags'],
        # Remote config is not handled by this script
        remote_config=None,
    )
    logger.info(f"Training Config: {training_config}")

    trainer = CLTTrainerRay(
        clt_config=clt_config,
        training_config=training_config,
        log_dir=cfg['log_dir'],
        device=device,
        distributed=False,  # distributing model across gpus within a ray tune worker not implemented
        use_ray='tune'
    )

    trained_clt_model = trainer.train(eval_every=training_config.eval_interval)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Cross-Layer Transcoder (CLT) using pre-generated local activations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Core Training Parameters ---
    core_group = parser.add_argument_group("Core Training Parameters")
    core_group.add_argument(
        "--activation-path",
        type=str,
        required=True,
        help="Path to the directory containing pre-generated activations (including index.bin, metadata.json, etc.).",
    )
    core_group.add_argument(
        "--output-dir",
        type=str,
        default=f"clt_train_local_{int(time.time())}",
        help="Directory to save logs, checkpoints, and final model.",
    )
    core_group.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Base model name or path (e.g., 'gpt2', 'gpt2-medium'). Must match the model used for activation generation.",
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

    # --- CLT Model Architecture ---
    clt_group = parser.add_argument_group("CLT Model Architecture (CLTConfig)")
    clt_group.add_argument(
        "--num-features",
        type=int,
        required=True,
        help="Number of features per layer in the CLT.",
    )
    # num_layers and d_model are derived from the base model
    clt_group.add_argument(
        "--activation-fn",
        type=str,
        choices=["jumprelu", "relu"],
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
        "--clt-dtype",
        type=str,
        default=None,
        help="Optional data type for the CLT model parameters (e.g., 'float16', 'bfloat16').",
    )

    # --- Training Hyperparameters ---
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
        "--normalization-method",
        type=str,
        choices=["auto", "none"],
        default="auto",
        help=(
            "Normalization for activation store. 'auto' uses pre-calculated stats "
            "(norm_stats.json) from the activation_path. 'none' disables normalization."
        ),
    )
    train_group.add_argument(
        "--sparsity-lambda",
        type=float,
        default=1e-3,
        help="Coefficient for the L1 sparsity penalty.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    train_group.add_argument(
        "--activation-dtype",
        type=str,
        default="float32",
        help="Data type to load activations as (e.g., 'float32', 'bfloat16'). Should match storage or be compatible.",
    )
    train_group.add_argument(
        "--dead-feature-window",
        type=int,
        default=1000,
        help="Number of steps of inactivity before a feature is considered 'dead' for evaluation.",
    )

    tune_group = parser.add_argument_group("Hyperparameter Tuning Parameters")
    tune_group.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="Number of GPUs available for hyperparameter tuning."
    )
    tune_group.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of trainers to run in parallel for hyperparameter tuning."
    )

    # --- Sampling Strategy --- Added Group
    sampling_group = parser.add_argument_group("Sampling Strategy (TrainingConfig)")
    sampling_group.add_argument(
        "--sampling-strategy",
        type=str,
        choices=["sequential", "random_chunk"],
        default="sequential",
        help="Sampling strategy: 'sequential' processes chunks in order per epoch, 'random_chunk' picks a random valid chunk each step.",
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
    # WandB arguments
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
        help="Custom name for the WandB run (defaults to a timestamp).",
    )
    log_group.add_argument("--wandb-tags", nargs="+", default=None, help="List of tags for the WandB run.")

    args = parser.parse_args()

    # --- Validation ---
    # Simplified validation: activation_path is required by argparse
    # No need to check for generation args

    return args


def main():
    """Main function to configure and run the CLTTrainer for local activations."""
    args = parse_args()

    # --- Setup Output Directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {output_dir.resolve()}")

    # Save command-line arguments
    try:
        with open(output_dir / "cli_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save command-line args: {e}")


    # --- Start Tuning ---
    # config to pass to workers
    hps = {
        'log_dir': str(output_dir.resolve()),
        **vars(args)
    }
    # add actual hyperparameters
    hps |= {
        'sparsity_c': ray.tune.grid_search([0.01, 0.03, 0.09, 0.27, 0.81, 2.43]),
        'sparsity_lambda': ray.tune.grid_search([1e-5, 3e-5, 9e-5, 2.7e-4, 8.1e-4, 2.43e-3]),
    }

    logger.info("Starting hyperparameter tuning from local activations...")
    try:
        tuner = ray.tune.Tuner(
            ray.tune.with_resources(train_loop_per_worker, {'cpu': 1, 'gpu': args.n_gpus / args.n_workers}),
            param_space=hps,
            tune_config=ray.tune.TuneConfig(max_concurrent_trials=args.n_workers),
            run_config=ray.tune.RunConfig(storage_path=str(output_dir.resolve()))
        )
        results = tuner.fit()
        logger.info("Tuning complete!")
        logger.info(f"Final model and logs saved in: {output_dir.resolve()}")
    except Exception as e:
        logger.exception(f"Tuning failed: {e}")  # Use logger.exception
        raise


if __name__ == "__main__":
    main()
