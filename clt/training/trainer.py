import torch
import torch.optim as optim
from typing import Dict, Optional, Union, Any
from tqdm import tqdm  # type: ignore
import os
import json
import time
import sys
import logging  # Add logging import
import torch.distributed as dist  # Import torch.distributed
from torch.distributed import ProcessGroup  # Import ProcessGroup
from dataclasses import asdict  # Add dataclasses import
import numpy as np  # For numpy RNG state
import random  # For python RNG state

from clt.config import CLTConfig, TrainingConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.data.base_store import BaseActivationStore

# Import the new manifest-based stores
from clt.training.data.local_activation_store import LocalActivationStore
from clt.training.data.remote_activation_store import RemoteActivationStore

from clt.training.losses import LossManager

from .evaluator import CLTEvaluator  # Import the new evaluator
from .wandb_logger import WandBLogger, DummyWandBLogger
from .checkpointing import CheckpointManager
from .distributed_utils import average_shared_parameter_grads  # Add this import
from clt.training.data.activation_store_factory import create_activation_store  # Add this import
from .metric_utils import MetricLogger  # Add this import
from .diagnostics import compute_sparsity_diagnostics  # Add this import
from .profiler import TrainingProfiler, CUDAMemoryProfiler, DistributedProfiler  # Add profiler imports

# Get logger for this module
logger = logging.getLogger(__name__)


class CLTTrainer:
    """Trainer for Cross-Layer Transcoder models."""

    # Add type hint for the activation store attribute
    activation_store: BaseActivationStore
    # Model type hint
    model: CrossLayerTranscoder
    # WandB logger can be real or dummy
    wandb_logger: Union[WandBLogger, DummyWandBLogger]

    def __init__(
        self,
        clt_config: CLTConfig,
        training_config: TrainingConfig,
        log_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        distributed: bool = False,  # Add distributed flag
        resume_from_checkpoint_path: Optional[str] = None,  # For resuming
    ):
        """Initialize the CLT trainer.

        Args:
            clt_config: Configuration for the CLT model
            training_config: Configuration for training
            log_dir: Directory to save logs and checkpoints
            device: Device to use for training (ignored if distributed)
            distributed: Whether to use distributed training
            resume_from_checkpoint_path: Path to a checkpoint file to resume from.
                                         For non-distributed, path to .safetensors model file.
                                         For distributed, path to checkpoint directory (e.g. step_XXX or latest).
        """
        self.clt_config = clt_config
        self.training_config = training_config
        self.distributed = distributed
        self.loaded_trainer_state: Optional[Dict[str, Any]] = None  # Store loaded state here

        # Initialize distributed training if enabled
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.process_group: Optional[ProcessGroup] = None  # For tensor parallelism

        if self.distributed:
            if not dist.is_initialized():
                # Default backend, consider NCCL for NVIDIA GPUs
                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))  # Get local rank if available

            # Set device based on local_rank when distributed
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{self.local_rank}")
                torch.cuda.set_device(self.device)
            else:
                # Fallback for CPU distributed testing (not typical)
                self.device = torch.device("cpu")
                logger.warning("Distributed training requested but CUDA not available. Using CPU.")
            # Set the process group for tensor parallelism (using WORLD for now)
            self.process_group = dist.group.WORLD
        else:
            # Original device handling for non-distributed case
            _device_input = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            self.device = torch.device(_device_input) if isinstance(_device_input, str) else _device_input
            # Process group is None when not distributed
            self.process_group = None
            # world_size is 1 if not distributed, rank is 0 (already initialized)

        # --- Mixed Precision Setup ---
        self.mixed_precision = self.training_config.precision.lower()  # fp32, fp16, bf16

        self.autocast_enabled = False
        self.autocast_dtype = torch.float32  # Default for fp32

        if self.mixed_precision == "fp16":
            if torch.cuda.is_available():
                self.autocast_enabled = True
                self.autocast_dtype = torch.float16
                logger.info(f"Rank {self.rank}: Enabling CUDA autocast with float16.")
            elif self.device.type == "mps":
                self.autocast_enabled = True
                self.autocast_dtype = torch.float16  # MPS supports float16
                logger.info(f"Rank {self.rank}: Enabling MPS autocast with float16.")
        elif self.mixed_precision == "bf16":
            if torch.cuda.is_available():
                self.autocast_enabled = True
                self.autocast_dtype = torch.bfloat16
                logger.info(f"Rank {self.rank}: Enabling CUDA autocast with bfloat16.")
            elif self.device.type == "mps":
                self.autocast_enabled = True
                self.autocast_dtype = torch.bfloat16  # MPS supports bfloat16
                logger.info(f"Rank {self.rank}: Enabling MPS autocast with bfloat16.")
        # If self.mixed_precision is "fp32", autocast_enabled remains False, autocast_dtype remains float32

        # Initialize GradScaler (enabled only for fp16 on CUDA)
        # MPS doesn't use GradScaler in the same way, typically.
        # The warning is about autocast itself, not scaler.
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.mixed_precision == "fp16" and torch.cuda.is_available()))

        logger.info(
            f"Rank {self.rank}: Mixed precision mode: {self.mixed_precision}, autocast_enabled: {self.autocast_enabled}, autocast_dtype: {self.autocast_dtype}"
        )
        logger.info(f"Rank {self.rank}: GradScaler enabled: {self.scaler.is_enabled()}")

        # Set up log directory - only rank 0 creates it
        self.log_dir = log_dir or f"clt_train_{int(time.time())}"
        if not self.distributed or self.rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)

        # Record start time
        self.start_time = time.time()

        # Initialize model, passing device and process group for direct initialization
        # self.process_group is correctly set to None if not distributed

        # --- Set Rank-Specific Seed --- #
        if self.training_config.seed is not None:
            torch.manual_seed(self.training_config.seed + self.rank)
            logger.info(f"Rank {self.rank}: Set manual seed to {self.training_config.seed + self.rank}")
        else:
            logger.warning(f"Rank {self.rank}: No seed provided in TrainingConfig. Using default torch seeding.")

        # Initialize profilers early (before model creation)
        self.profiler = TrainingProfiler(
            enabled=self.training_config.enable_profiling, log_interval=self.training_config.log_interval
        )
        self.memory_profiler = CUDAMemoryProfiler(
            enabled=self.training_config.enable_profiling and torch.cuda.is_available()
        )
        self.dist_profiler = DistributedProfiler(
            enabled=self.training_config.enable_profiling and self.distributed, rank=self.rank
        )

        self.model = CrossLayerTranscoder(
            clt_config,
            process_group=self.process_group,
            device=self.device,
            profiler=self.profiler if self.training_config.enable_profiling else None,
        )

        # --- Optionally convert model to FP16 (Step 8) ---
        # If precision is "fp16", GradScaler is used, which expects FP32 optimizer parameters.
        # Therefore, if precision is "fp16", we do not convert model to .half() before optimizer init,
        # regardless of fp16_convert_weights. Autocast handles FP16 computations.
        # fp16_convert_weights will apply if precision is not "fp16" (e.g., "bf16" or "fp32" training
        # where user still wants to store/use fp16 weights directly without GradScaler for fp16).
        if self.training_config.fp16_convert_weights:
            if self.mixed_precision == "fp16":
                logger.warning(
                    f"Rank {self.rank}: 'fp16_convert_weights=True' is set with 'precision=fp16'. "
                    "GradScaler expects FP32 optimizer parameters. Model weights will NOT be converted to FP16 "
                    "before optimizer initialization to ensure GradScaler compatibility. "
                    "Autocast will still use FP16 for computations."
                )
            else:  # mixed_precision is 'bf16' or 'fp32' (autocast_dtype is bfloat16 or float32, GradScaler is disabled for fp16)
                logger.info(
                    f"Rank {self.rank}: Converting model weights and buffers to FP16 (fp16_convert_weights=True, precision={self.mixed_precision})."
                )
                self.model.half()  # Converts all parameters and buffers
                # Keep LayerNorm buffers in FP32 for stability
                for name, buf in self.model.named_buffers():
                    if buf.dtype == torch.float16 and "norm" in name.lower():
                        logger.info(f"Rank {self.rank}: Converting buffer '{name}' from FP16 back to FP32.")
                        buf.data = buf.data.float()

        # Initialize optimizer - works on local parameters
        # Explicitly type the kwargs dict for clarity and linting
        optimizer_kwargs: Dict[str, Any] = {"lr": training_config.learning_rate}
        beta1 = training_config.optimizer_beta1  # Could be None
        beta2 = training_config.optimizer_beta2  # Could be None

        # Only add 'betas' if at least one is specified
        if beta1 is not None or beta2 is not None:
            # Get defaults if one is None
            # Default Adam/AdamW betas are (0.9, 0.999)
            final_beta1 = beta1 if beta1 is not None else 0.9
            final_beta2 = beta2 if beta2 is not None else 0.999
            optimizer_kwargs["betas"] = (final_beta1, final_beta2)
            logger.info(f"Rank {self.rank}: Using optimizer betas: ({final_beta1}, {final_beta2})")

        if training_config.optimizer == "adam":
            self.optimizer: Any = optim.Adam(self.model.parameters(), **optimizer_kwargs)
        else:  # "adamw"
            self.optimizer = optim.AdamW(self.model.parameters(), **optimizer_kwargs)

        # Initialize scheduler
        self.scheduler: Optional[Any] = None
        scheduler_type = training_config.lr_scheduler
        # Get scheduler params from config, default to empty dict if None
        scheduler_params = training_config.lr_scheduler_params or {}

        if scheduler_type == "linear":
            # Default params for LinearLR
            default_linear_params = {
                "start_factor": 1.0,
                "end_factor": 0.1,
                # total_iters is always training_steps for this setup
            }
            # Update defaults with user-provided params
            final_params = {**default_linear_params, **scheduler_params}
            # Ensure total_iters is not overridden by user params
            final_params.pop("total_iters", None)

            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                total_iters=training_config.training_steps,
                **final_params,  # Pass start_factor, end_factor, etc.
            )
            logger.info(
                f"Rank {self.rank}: Using LinearLR scheduler with params: {final_params}, total_iters={training_config.training_steps}"
            )

        elif scheduler_type == "cosine":
            # Default params for CosineAnnealingLR
            default_cosine_params = {
                # T_max defaults to training_steps
                "eta_min": 0,  # Default minimum LR
            }
            # Update defaults with user-provided params
            final_params = {**default_cosine_params, **scheduler_params}
            # Set T_max explicitly, allowing override but defaulting to training_steps
            t_max = final_params.pop("T_max", training_config.training_steps)

            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max, **final_params  # Pass eta_min, etc.
            )
            logger.info(
                f"Rank {self.rank}: Using CosineAnnealingLR scheduler with params: {final_params}, T_max={t_max}"
            )

        elif scheduler_type == "linear_final20":
            # This scheduler keeps LR constant for the initial fraction of training
            # and then linearly decays it to 0 over the remaining steps (default 20%).
            # The fraction can be customized via lr_scheduler_params["decay_start_frac"].
            decay_start_frac = scheduler_params.get("decay_start_frac", 0.8)  # 0.8 means last 20% decays
            assert 0.0 < decay_start_frac < 1.0, "decay_start_frac must be between 0 and 1"
            total_steps = training_config.training_steps
            decay_start_step = int(decay_start_frac * total_steps)

            def lr_lambda(current_step: int):
                if current_step < decay_start_step:
                    return 1.0  # Keep LR constant
                # Linearly decay from 1 -> 0 over the remaining steps
                remaining = total_steps - current_step
                decay_steps = total_steps - decay_start_step
                return max(remaining / decay_steps, 0.0)

            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            logger.info(
                "Rank %d: Using linear_final20 LR scheduler with decay_start_frac=%s (start step %d of %d)",
                self.rank,
                decay_start_frac,
                decay_start_step,
                total_steps,
            )

        # Add elif blocks here for other potential schedulers

        # Initialize activation store.
        # For tensor parallelism we need every rank to process the *same* batch because
        # activations are sharded **across the feature dimension**, not across tokens.
        # Passing the per-rank/world sharded parameters here would make each rank see a
        # different subset of tokens and lead to inconsistent batch sizes which breaks
        # collective ops such as all_gather in ColumnParallelLinear.

        # Pass the actual rank and world size from the distributed setup
        activation_store_rank = self.rank
        activation_store_world = self.world_size

        self.activation_store = create_activation_store(
            training_config=self.training_config,
            clt_config=self.clt_config,
            device=self.device,
            rank=activation_store_rank,  # Pass the actual rank
            world_size=activation_store_world,  # Pass the actual world size
            start_time=self.start_time,
            shard_data=not self.distributed,  # ---> ADDED: False if distributed (TP), True otherwise <----
        )

        # Pass normalisation statistics (if available) so the loss can be computed in
        # the *original* scale even when inputs/targets are stored normalised.
        mean_tg_stats = getattr(self.activation_store, "mean_tg", {})  # type: ignore[arg-type]
        std_tg_stats = getattr(self.activation_store, "std_tg", {})  # type: ignore[arg-type]

        self.loss_manager = LossManager(
            training_config,
            mean_tg=mean_tg_stats,
            std_tg=std_tg_stats,
            clt_config=clt_config,
        )

        # Initialize Evaluator - Pass norm stats here too
        self.evaluator = CLTEvaluator(
            model=self.model,
            device=self.device,
            start_time=self.start_time,
            mean_tg=mean_tg_stats,  # Pass the same stats
            std_tg=std_tg_stats,  # Pass the same stats
            normalization_method=training_config.normalization_method,
            d_model=clt_config.d_model,
        )

        # Initialize dead neuron counters (replicated for now, consider sharding later if needed)
        self.n_forward_passes_since_fired = torch.zeros(
            (clt_config.num_layers, clt_config.num_features),
            device=self.device,
            dtype=torch.long,
            requires_grad=False,  # Explicitly disable gradient tracking
        )

        # ------------------------------------------------------------------
        #  CheckpointManager EARLY init (use Dummy logger placeholder)
        # ------------------------------------------------------------------
        # We create a placeholder DummyWandBLogger so that we can construct the
        # CheckpointManager and load the checkpoint *before* creating the real
        # WandBLogger.  This lets us obtain the stored `wandb_run_id` from the
        # checkpoint and feed it into WandBLogger for a proper resume.
        placeholder_wandb_logger: Union[WandBLogger, DummyWandBLogger]
        placeholder_wandb_logger = DummyWandBLogger(
            training_config=training_config, clt_config=clt_config, log_dir=self.log_dir, resume_wandb_id=None
        )

        self.checkpoint_manager = CheckpointManager(
            model=self.model,
            activation_store=self.activation_store,
            wandb_logger=placeholder_wandb_logger,
            log_dir=self.log_dir,
            distributed=self.distributed,
            rank=self.rank,
            device=self.device,
            world_size=self.world_size,
        )

        # ------------------------------------------------------------------
        #  Load checkpoint (if any) *before* real WandB logger init
        # ------------------------------------------------------------------
        if resume_from_checkpoint_path:
            logger.info(
                f"Rank {self.rank}: Attempting to resume training from checkpoint: {resume_from_checkpoint_path}"
            )
            # load_checkpoint loads model and activation store state internally
            # and returns the trainer_state dictionary (optimizer, scheduler, step, etc.)
            self.loaded_trainer_state = self.checkpoint_manager.load_checkpoint(resume_from_checkpoint_path)
            if self.loaded_trainer_state:
                loaded_step_for_log = self.loaded_trainer_state.get("step", -1)
                logger.info(
                    f"Rank {self.rank}: Successfully loaded checkpoint. Trainer state recovered for step {loaded_step_for_log}."
                )
                # ADDED: Detailed logging of loaded_trainer_state keys and step value
                logger.info(f"Rank {self.rank}: Keys in loaded_trainer_state: {list(self.loaded_trainer_state.keys())}")
                logger.info(
                    f"Rank {self.rank}: Value of 'step' in loaded_trainer_state: {self.loaded_trainer_state.get('step')}"
                )
                logger.info(
                    f"Rank {self.rank}: Value of 'wandb_run_id' in loaded_trainer_state: {self.loaded_trainer_state.get('wandb_run_id')}"
                )
            else:
                logger.warning(
                    f"Rank {self.rank}: Checkpoint loaded, but trainer state (optimizer, step, etc.) was empty or not found. Starting from scratch."
                )
                # loaded_trainer_state will be None or empty, so train() will start fresh

        # ------------------------------------------------------------------
        #  Initialize WandB logger (after checkpoint load so we have run_id)
        # ------------------------------------------------------------------
        if not self.distributed or self.rank == 0:
            loaded_wandb_run_id_for_init: Optional[str] = None
            if self.loaded_trainer_state:  # Check if resuming and state was loaded
                loaded_wandb_run_id_for_init = self.loaded_trainer_state.get("wandb_run_id")
                if loaded_wandb_run_id_for_init:
                    logger.info(
                        f"Rank {self.rank}: Found WandB run ID {loaded_wandb_run_id_for_init} in loaded checkpoint state. Attempting to resume."
                    )

            self.wandb_logger = WandBLogger(
                training_config=training_config,
                clt_config=clt_config,
                log_dir=self.log_dir,
                resume_wandb_id=loaded_wandb_run_id_for_init,
            )
        else:
            self.wandb_logger = DummyWandBLogger(
                training_config=training_config,  # type: ignore[arg-type]
                clt_config=clt_config,  # type: ignore[arg-type]
                log_dir=self.log_dir,  # type: ignore[arg-type]
                resume_wandb_id=None,  # type: ignore[arg-type]
            )

        # Replace placeholder in CheckpointManager with the real logger
        self.checkpoint_manager.wandb_logger = self.wandb_logger

        # ------------------------------------------------------------------
        #  MetricLogger (uses the real wandb_logger)
        # ------------------------------------------------------------------
        self.metric_logger = MetricLogger(
            distributed=self.distributed,
            rank=self.rank,
            log_dir=self.log_dir,
            wandb_logger=self.wandb_logger,
            training_config=self.training_config,
            world_size=self.world_size,
        )

    @property
    def dead_neurons_mask(self) -> torch.Tensor:
        """Boolean mask indicating dead neurons based on inactivity window."""
        # Ensure counter is initialized
        if not hasattr(self, "n_forward_passes_since_fired") or self.n_forward_passes_since_fired is None:
            # Return an all-false mask if counter doesn't exist yet
            return torch.zeros(
                (self.clt_config.num_layers, self.clt_config.num_features),
                dtype=torch.bool,
                device=self.device,
            )
        # Detach to prevent any computation graph references
        return (self.n_forward_passes_since_fired > self.training_config.dead_feature_window).detach()

    def train(self, eval_every: int = 1000) -> CrossLayerTranscoder:
        """Train the CLT model.

        Args:
            eval_every: Evaluate model every N steps

        Returns:
            Trained CLT model (local shard)
        """
        # Print startup message from rank 0 only
        if not self.distributed or self.rank == 0:
            logger.info(f"Starting CLT training on {self.device}...")
            logger.info(
                f"Model has {self.clt_config.num_features} features per layer "
                f"and {self.clt_config.num_layers} layers"
            )
            logger.info(f"Training for {self.training_config.training_steps} steps.")
            logger.info(f"Logging to {self.log_dir}")
            if self.distributed:
                logger.info(f"Distributed training with {self.world_size} processes (Tensor Parallelism)")

            # Check if using normalization and notify user
            if self.training_config.normalization_method == "mean_std":
                logger.info("\n>>> NORMALIZATION CONFIGURATION <<<")
                logger.info("Using mean/std normalization with pre-calculated statistics from norm_stats.json")
            elif self.training_config.normalization_method == "sqrt_d_model":
                logger.info("\n>>> NORMALIZATION CONFIGURATION <<<")
                logger.info("Using sqrt(d_model) normalization (EleutherAI-style)")
            elif self.training_config.normalization_method == "none":
                logger.info("\n>>> NORMALIZATION CONFIGURATION <<<")
                logger.info("No normalization will be applied to activations")

            # Make sure we flush stdout to ensure prints appear immediately,
            # especially important in Jupyter/interactive environments
            sys.stdout.flush()
            # Wait for 1 second to ensure output is displayed before training starts
            time.sleep(1)
            logger.info("\n>>> TRAINING PHASE <<<")
            sys.stdout.flush()

        # After the existing startup messages
        if self.distributed:
            logger.info("\n!!! DIAGNOSTIC INFO !!!")
            logger.info(f"Rank {self.rank}: Process group type: {type(self.process_group)}")
            logger.info(f"Rank {self.rank}: RowParallelLinear _reduce does NOT divide by world_size")
            logger.info(f"Rank {self.rank}: Using weight regularization in sparsity penalty")
            logger.info(f"Rank {self.rank}: Averaging replicated parameter gradients")
            # Check if activation store has rank/world attributes before accessing
            store_rank = getattr(self.activation_store, "rank", "N/A")
            store_world = getattr(self.activation_store, "world", "N/A")
            logger.info(f"Rank {self.rank}: Data sharding: rank={store_rank}, world={store_world}")
            logger.info(f"Rank {self.rank}: Batch size tokens: {self.training_config.train_batch_size_tokens}")
            logger.info(f"Rank {self.rank}: Sparsity lambda: {self.training_config.sparsity_lambda}")

            # Check if activation store actually loaded correctly
            batch_avail = next(iter(self.activation_store), None)
            logger.info(f"Rank {self.rank}: First batch available: {batch_avail is not None}")

            # Force torch to compile/execute our code by running a tiny forward/backward pass
            dummy = torch.ones(1, device=self.device, requires_grad=True)
            dummy_out = dummy * 2
            dummy_out.backward()
            logger.info("!!! END DIAGNOSTIC !!!\n")

        # --- Enable Anomaly Detection (if configured) ---
        if self.training_config.debug_anomaly:
            torch.autograd.set_detect_anomaly(True)
            if not self.distributed or self.rank == 0:
                logger.info("PyTorch Anomaly Detection ENABLED.")

        # Create progress bar only on rank 0
        pbar: Union[tqdm, range]

        initial_step = 0
        if self.loaded_trainer_state:  # Check if we are resuming and state was loaded
            initial_step = self.loaded_trainer_state.get("step", 0) + 1  # Start from next step
            if initial_step > 0:
                logger.info(f"Rank {self.rank}: Resuming training from step {initial_step}")
                try:
                    self.optimizer.load_state_dict(self.loaded_trainer_state["optimizer_state_dict"])
                    logger.info(f"Rank {self.rank}: Optimizer state loaded.")
                    if self.scheduler and self.loaded_trainer_state.get("scheduler_state_dict"):
                        self.scheduler.load_state_dict(self.loaded_trainer_state["scheduler_state_dict"])
                        logger.info(f"Rank {self.rank}: Scheduler state loaded.")
                    if self.scaler and self.scaler.is_enabled() and self.loaded_trainer_state.get("scaler_state_dict"):
                        self.scaler.load_state_dict(self.loaded_trainer_state["scaler_state_dict"])
                        logger.info(f"Rank {self.rank}: GradScaler state loaded.")

                    loaded_n_passes = self.loaded_trainer_state.get("n_forward_passes_since_fired")
                    if loaded_n_passes is not None:
                        self.n_forward_passes_since_fired.data = loaded_n_passes.to(self.device)
                        logger.info(f"Rank {self.rank}: n_forward_passes_since_fired state loaded.")

                    # Restore RNG states
                    if "torch_rng_state" in self.loaded_trainer_state:
                        torch.set_rng_state(
                            self.loaded_trainer_state["torch_rng_state"].cpu()
                        )  # Ensure it's on CPU before loading
                        logger.info(f"Rank {self.rank}: PyTorch RNG state loaded.")
                    if "numpy_rng_state" in self.loaded_trainer_state:
                        np.random.set_state(self.loaded_trainer_state["numpy_rng_state"])
                        logger.info(f"Rank {self.rank}: NumPy RNG state loaded.")
                    if "python_rng_state" in self.loaded_trainer_state:
                        random.setstate(self.loaded_trainer_state["python_rng_state"])
                        logger.info(f"Rank {self.rank}: Python RNG state loaded.")

                except KeyError as e:
                    logger.error(
                        f"Rank {self.rank}: KeyError when loading trainer state ({e}). Some states might not be restored correctly. Check checkpoint compatibility."
                    )
                except Exception as e:
                    logger.error(
                        f"Rank {self.rank}: Error loading trainer state: {e}. Training might not resume correctly."
                    )
            else:
                logger.info(
                    f"Rank {self.rank}: Loaded step is {initial_step - 1}, starting training from step 0 or continuing if step was 0."
                )
                initial_step = 0  # Ensure we don't start from 1 if loaded step was 0

        if not self.distributed or self.rank == 0:
            pbar = tqdm(
                range(initial_step, self.training_config.training_steps),
                desc="Training CLT",
                leave=True,
                initial=initial_step,  # Set initial for tqdm to show correct progress
                total=self.training_config.training_steps,  # Set total for tqdm
            )
        else:
            pbar = range(initial_step, self.training_config.training_steps)

        step = initial_step  # Initialize step before the loop, in case loop doesn't run
        try:
            for step in pbar:
                # Refresh progress bar on rank 0
                step_start_time = time.monotonic()  # Start timing the step
                if isinstance(pbar, tqdm):
                    pbar.refresh()

                try:
                    # Get batch directly from the iterator (handles distributed sampling internally)
                    with self.profiler.timer("data_loading") as timer:
                        inputs, targets = next(self.activation_store)
                        # Detach inputs and targets to ensure no computation graph from previous iterations
                        inputs = {k: v.detach() for k, v in inputs.items()}
                        targets = {k: v.detach() for k, v in targets.items()}
                    if hasattr(timer, "elapsed"):
                        self.profiler.record("data_loading", timer.elapsed)
                        logger.debug(f"Rank {self.rank} Step {step}: Getting batch took {timer.elapsed:.4f}s")

                    # logging to diagnose batch size mismatch
                    tok_cnt = next(iter(inputs.values())).shape[0]  # number of rows (=tokens) in this batch
                    # Only run the all_gather diagnostic when running in distributed mode
                    if self.distributed and self.world_size > 1 and dist.is_initialized():
                        with self.dist_profiler.profile_op("batch_size_all_gather"):
                            tok_cnt_t = torch.tensor([tok_cnt], device=self.device)
                            gathered = [torch.zeros_like(tok_cnt_t) for _ in range(self.world_size)]
                            dist.all_gather(gathered, tok_cnt_t)

                except StopIteration:
                    # Rank 0 prints message
                    if not self.distributed or self.rank == 0:
                        logger.info("Activation store exhausted. Training finished early.")
                    if self.distributed:
                        dist.barrier()  # Ensure all ranks see this
                    break  # Exit training loop if data runs out

                # --- Check for empty batch --- (Optional but good practice)
                # This check should ideally happen *before* moving data potentially
                if not inputs or not targets or not any(v.numel() > 0 for v in inputs.values()):
                    if not self.distributed or self.rank == 0:
                        logger.warning(f"Rank {self.rank}: Warning: Received empty batch at step {step}. Skipping.")
                    continue

                # --- BEGIN: One-time Normalization Check ---
                if step == 0 and (not self.distributed or self.rank == 0):
                    logger.info("--- Running Post-Normalization Check (First Batch) ---")
                    norm_applied = getattr(self.activation_store, "apply_normalization", None)
                    if isinstance(self.activation_store, (LocalActivationStore, RemoteActivationStore)):
                        logger.info(f"ActivationStore reports apply_normalization={norm_applied}")

                    for li in range(self.clt_config.num_layers):
                        mean_in, std_in, mean_tg, std_tg = float("nan"), float("nan"), float("nan"), float("nan")
                        try:
                            if li in inputs and inputs[li].numel() > 0:
                                input_tensor = inputs[li].float()
                                mean_in = input_tensor.mean().item()
                                std_in = input_tensor.std().item()
                            if li in targets and targets[li].numel() > 0:
                                target_tensor = targets[li].float()
                                mean_tg = target_tensor.mean().item()
                                std_tg = target_tensor.std().item()

                            if not (
                                torch.isnan(torch.tensor(mean_in)) and torch.isnan(torch.tensor(mean_tg))
                            ):  # Log if at least one value is valid
                                logger.info(
                                    f"  Layer {li:>2}: Input Mean={mean_in:+.4f}, Std={std_in:.4f} | Target Mean={mean_tg:+.4f}, Std={std_tg:.4f}"
                                )
                        except Exception as e:
                            logger.error(f"  Layer {li}: Error during normalization check: {e}")
                    logger.info("--- End Post-Normalization Check ---")
                # --- END: One-time Normalization Check ---

                # --- Forward pass and compute loss --- (All ranks)
                self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_enabled
                ):
                    # Profile forward pass
                    with self.profiler.timer("forward_pass") as timer:
                        feature_activations_batch = self.model.get_feature_activations(inputs)
                    if hasattr(timer, "elapsed"):
                        self.profiler.record("forward_pass", timer.elapsed)

                    # Profile loss computation
                    with self.profiler.timer("loss_computation") as timer:
                        loss, loss_dict = self.loss_manager.compute_total_loss(
                            self.model,
                            inputs,
                            targets,
                            step,
                            self.training_config.training_steps,
                            precomputed_activations=feature_activations_batch,
                            dead_neuron_mask=self.dead_neurons_mask,
                        )
                    if hasattr(timer, "elapsed"):
                        self.profiler.record("loss_computation", timer.elapsed)

                # Detach loss components for logging to prevent retaining graph
                loss_dict_detached = {
                    k: v.detach().item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()
                }

                # --- Update Dead Neuron Counters --- (All ranks, counter is replicated)
                # We need *full* feature activations *after* non-linearity
                if hasattr(self, "n_forward_passes_since_fired") and self.n_forward_passes_since_fired is not None:
                    with self.profiler.timer("dead_neuron_update") as timer:
                        with torch.no_grad():
                            for layer_idx, layer_acts in feature_activations_batch.items():
                                # Detach the activations to prevent retaining the computation graph
                                layer_acts = layer_acts.detach()

                                # Ensure layer index is within bounds of the counter tensor
                                if layer_idx < self.n_forward_passes_since_fired.shape[0]:
                                    if layer_acts.numel() > 0:
                                        # layer_acts shape: [batch_tokens, num_features]
                                        fired_mask_per_token = layer_acts > 1e-6
                                        fired_features_this_layer = fired_mask_per_token.any(dim=0)

                                        if (
                                            fired_features_this_layer.shape[0]
                                            == self.n_forward_passes_since_fired.shape[1]
                                        ):
                                            self.n_forward_passes_since_fired[layer_idx] += 1
                                            self.n_forward_passes_since_fired[layer_idx][fired_features_this_layer] = 0
                                        else:
                                            # Log warning only on rank 0 to avoid flooding logs
                                            if not self.distributed or self.rank == 0:
                                                logger.warning(
                                                    f"Rank {self.rank}: Shape mismatch for dead neuron update at layer {layer_idx}. "
                                                    f"Acts shape: {layer_acts.shape}, Fired mask: {fired_features_this_layer.shape}, "
                                                    f"Counter: {self.n_forward_passes_since_fired.shape}"
                                                )
                                    else:  # layer_acts.numel() == 0
                                        if not self.distributed or self.rank == 0:
                                            logger.debug(
                                                f"Rank {self.rank}: Layer {layer_idx} has empty activations, skipping dead neuron update for this layer."
                                            )
                                else:  # layer_idx out of bounds
                                    if not self.distributed or self.rank == 0:
                                        logger.warning(
                                            f"Rank {self.rank}: layer_idx {layer_idx} out of bounds for n_forward_passes_since_fired (shape {self.n_forward_passes_since_fired.shape}). Skipping dead neuron update."
                                        )
                    if hasattr(timer, "elapsed"):
                        self.profiler.record("dead_neuron_update", timer.elapsed)
                else:  # n_forward_passes_since_fired does not exist or is None
                    if not self.distributed or self.rank == 0:
                        logger.warning(
                            f"Rank {self.rank}: n_forward_passes_since_fired not available. Skipping dead neuron update."
                        )

                # --- Backward pass --- (All ranks, handles communication implicitly)
                if torch.isnan(loss):
                    if not self.distributed or self.rank == 0:
                        logger.warning(
                            f"\nRank {self.rank}: Warning: NaN loss encountered at step {step}. "
                            f"Skipping backward pass and optimizer step."
                        )
                        # Log detailed loss_dict for NaN debugging
                        logger.warning(f"Rank {self.rank}: NaN Loss - Detailed loss_dict at step {step}: {loss_dict}")
                else:
                    # ---- Back-prop with gradient scaling ----
                    with self.profiler.timer("backward_pass") as timer:
                        self.scaler.scale(loss).backward()
                    if hasattr(timer, "elapsed"):
                        self.profiler.record("backward_pass", timer.elapsed)

                    # Unscale gradients before clipping and distributed averaging
                    self.scaler.unscale_(self.optimizer)

                    # --- Synchronise gradients of replicated parameters --- #
                    if self.distributed and self.world_size > 1:
                        with self.profiler.timer("gradient_sync") as timer:
                            with self.dist_profiler.profile_op("gradient_all_reduce"):
                                average_shared_parameter_grads(self.model, self.world_size)
                        if hasattr(timer, "elapsed"):
                            self.profiler.record("gradient_sync", timer.elapsed)

                    # --- Gradient clipping (operates on unscaled gradients) --- #
                    if self.training_config.gradient_clip_val is not None:
                        with self.profiler.timer("gradient_clipping") as timer:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.training_config.gradient_clip_val,
                            )
                        if hasattr(timer, "elapsed"):
                            self.profiler.record("gradient_clipping", timer.elapsed)

                    # --- Optimizer step (scaler handles scaling/unscaling) ---
                    with self.profiler.timer("optimizer_step") as timer:
                        self.scaler.step(self.optimizer)
                    if hasattr(timer, "elapsed"):
                        self.profiler.record("optimizer_step", timer.elapsed)

                    # --- Update scaler for next iteration ---
                    self.scaler.update()

                    # --- Invalidate Caches (moved after optimizer step) --- #
                    # Ensure we invalidate any cached decoder norms to avoid retaining graphs across iterations
                    if hasattr(self.model, "decoder_module") and hasattr(
                        self.model.decoder_module, "_cached_decoder_norms"
                    ):
                        # The buffer is optional; resetting to None breaks the reference to the previous graph.
                        self.model.decoder_module._cached_decoder_norms = None  # type: ignore[assignment]

                # --- Sync Dead Neuron Counters --- #
                if (
                    self.distributed
                    and self.world_size > 1
                    and hasattr(self, "n_forward_passes_since_fired")
                    and self.n_forward_passes_since_fired is not None
                ):
                    with self.profiler.timer("dead_neuron_sync") as timer:
                        with self.dist_profiler.profile_op("dead_neuron_all_reduce"):
                            dist.all_reduce(
                                self.n_forward_passes_since_fired, op=dist.ReduceOp.MIN, group=self.process_group
                            )
                    if hasattr(timer, "elapsed"):
                        self.profiler.record("dead_neuron_sync", timer.elapsed)

                # --- Scheduler step --- (All ranks)
                if self.scheduler:
                    self.scheduler.step()

                # --- Update progress bar --- (Rank 0 only)
                if isinstance(pbar, tqdm):
                    description = (
                        f"Loss: {loss_dict_detached.get('total', float('nan')):.4f} "
                        f"(R: {loss_dict_detached.get('reconstruction', float('nan')):.4f} "
                        f"S: {loss_dict_detached.get('sparsity', float('nan')):.4f} "
                        f"P: {loss_dict_detached.get('preactivation', float('nan')):.4f})"
                    )
                    pbar.set_description(description)
                    # Force update to display progress
                    if step % 1 == 0:  # Update every step
                        pbar.refresh()
                        sys.stdout.flush()

                # --- Log metrics --- (Rank 0 logs to WandB/file)
                current_lr_for_log = self.scheduler.get_last_lr()[0] if self.scheduler else None
                current_lambda_for_log = self.loss_manager.get_current_sparsity_lambda()
                # Removed broad try-except around metric_logger.log_training_step
                self.metric_logger.log_training_step(
                    step,
                    loss_dict_detached,
                    current_lr=current_lr_for_log,
                    current_sparsity_lambda=current_lambda_for_log,
                )

                step_duration = time.monotonic() - step_start_time
                logger.debug(
                    f"Rank {self.rank} Step {step}: Main logic (incl. batch get, fwd, bwd, optim) took {step_duration:.4f}s"
                )

                # --- Evaluation & Checkpointing ---
                eval_interval = self.training_config.eval_interval
                checkpoint_interval = self.training_config.checkpoint_interval

                save_checkpoint_flag = (step > 0 and step % checkpoint_interval == 0) or (  # Avoid checkpoint at step 0
                    step == self.training_config.training_steps - 1
                )
                run_eval_flag = (step % eval_interval == 0) or (step == self.training_config.training_steps - 1)

                # --- Evaluation (all ranks participate to match collectives) ---
                # In tensor-parallel mode the model forward includes collective ops (all_reduce/all_gather).
                # If only rank 0 performed the forward pass these collectives would block on the other ranks
                # resulting in NCCL timeouts.  Therefore, *every* rank must execute the evaluation forward pass.
                # We still only log / store the resulting metrics on rank 0.
                if run_eval_flag:
                    if self.distributed:
                        with self.dist_profiler.profile_op("eval_barrier"):
                            dist.barrier()  # Sync before evaluation starts so that all ranks enter together

                    # Wrap the entire evaluation logic in no_grad to prevent graph pollution
                    with torch.no_grad():
                        # Compute evaluation metrics on all ranks to keep collective ops aligned
                        # Wrap the evaluation logic in autocast
                        with self.profiler.timer("evaluation") as timer:
                            with torch.autocast(
                                device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_enabled
                            ):
                                current_dead_mask = self.dead_neurons_mask.detach().clone()
                                # Detach inputs and targets to prevent retaining computation graph
                                inputs_detached = {k: v.detach() for k, v in inputs.items()}
                                targets_detached = {k: v.detach() for k, v in targets.items()}
                                eval_metrics = self.evaluator.compute_metrics(
                                    inputs_detached,  # These inputs are from the current training batch
                                    targets_detached,  # These targets are from the current training batch
                                    dead_neuron_mask=current_dead_mask,
                                )
                        if hasattr(timer, "elapsed"):
                            self.profiler.record("evaluation", timer.elapsed)

                            # --- Log per-layer standard deviation of pre-activations ---
                            # This requires getting the pre-activations first.
                            # _encode_all_layers returns: preactivations_dict, original_shapes_info, device, dtype
                            preactivations_eval_dict, _ = self.model._encode_all_layers(inputs_detached)
                            layerwise_preact_std_dev: Dict[str, float] = {}
                            if preactivations_eval_dict:
                                for layer_idx, preact_tensor in preactivations_eval_dict.items():
                                    if preact_tensor.numel() > 0:
                                        # Calculate std dev of all elements in the preactivation tensor for this layer
                                        std_dev_val = preact_tensor.std().item()
                                        layerwise_preact_std_dev[f"layer_{layer_idx}"] = std_dev_val
                                    else:
                                        layerwise_preact_std_dev[f"layer_{layer_idx}"] = float("nan")  # Or 0.0

                            # Add to eval_metrics for WandB logging
                            if layerwise_preact_std_dev:
                                eval_metrics["layerwise/preactivation_std_dev"] = layerwise_preact_std_dev
                            # --- End logging pre-activation std dev ---

                            # Optionally compute and log sparsity diagnostics (can be slow).
                            # IMPORTANT: every rank must execute compute_sparsity_diagnostics because it
                            # internally calls `get_decoder_norms`, which performs distributed all-reduces.
                            # We therefore compute the diagnostics on **all** ranks to keep the NCCL
                            # collectives aligned, but we still only merge the returned values into
                            # `eval_metrics` and log them on rank 0.
                            if self.training_config.compute_sparsity_diagnostics:
                                # Detach the activations from the current computation graph before using them in diagnostics
                                # to prevent the "backward through the graph a second time" error.
                                feature_activations_batch_detached = (
                                    {k: v.detach() for k, v in feature_activations_batch.items()}
                                    if feature_activations_batch is not None
                                    else None
                                )

                                if feature_activations_batch_detached is not None:
                                    sparsity_diag_metrics = compute_sparsity_diagnostics(
                                        model=self.model,
                                        training_config=self.training_config,
                                        feature_activations=feature_activations_batch_detached,
                                    )
                                    # Only rank 0 merges & logs the diagnostics, preventing duplicate WandB entries.
                                    if (not self.distributed) or (self.rank == 0):
                                        if sparsity_diag_metrics:
                                            eval_metrics.update(sparsity_diag_metrics)

                        # --- END of autocast block for evaluation ---

                    if not self.distributed or self.rank == 0:
                        # Store evaluation metrics (for saving to JSON) - Handled by MetricLogger
                        # self.metrics["eval_metrics"].append({"step": step, **eval_metrics})

                        # --- Update Progress Bar Postfix --- (Restore this block)
                        l0_str = f"AvgL0: {eval_metrics.get('sparsity/avg_l0', 0.0):.2f}"
                        ev_str = f"EV: {eval_metrics.get('reconstruction/explained_variance', 0.0):.3f}"
                        avg_density_mean = eval_metrics.get("sparsity/feature_density_mean")
                        dens_str = f"Dens: {avg_density_mean:.3f}" if avg_density_mean is not None else "Dens: N/A"
                        eval_dead_str = f"Dead(Eval): {eval_metrics.get('dead_features/total_eval', 0)}"
                        eval_msg = f"{l0_str}, {ev_str}, {dens_str}, {eval_dead_str}"

                        if isinstance(pbar, tqdm):
                            pbar.set_postfix_str(eval_msg)
                            pbar.refresh()

                        # --- Log evaluation metrics (now done by MetricLogger) ---
                        self.metric_logger.log_evaluation_metrics(step, eval_metrics)

                    # Ensure all ranks finish evaluation before proceeding
                    if self.distributed:
                        dist.barrier()

                # --- Checkpointing (All ranks participate) ---
                if save_checkpoint_flag:
                    # Removed broad try-except around checkpoint saving in the loop
                    # Specific IOErrors can be caught by self.checkpoint_manager._save_checkpoint if needed internally
                    # or training can halt if checkpointing is critical and fails.
                    current_trainer_state_for_checkpoint = {
                        "step": step,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                        "scaler_state_dict": (
                            self.scaler.state_dict() if self.scaler and self.scaler.is_enabled() else None
                        ),
                        "n_forward_passes_since_fired": (
                            self.n_forward_passes_since_fired.cpu()
                            if self.n_forward_passes_since_fired is not None
                            else None
                        ),
                        "wandb_run_id": self.wandb_logger.get_current_wandb_run_id(),
                        "torch_rng_state": torch.get_rng_state(),
                        "numpy_rng_state": np.random.get_state(),
                        "python_rng_state": random.getstate(),
                    }
                    self.checkpoint_manager._save_checkpoint(step, current_trainer_state_for_checkpoint)

                # --- Profile memory and step profiler --- #
                if not self.distributed or self.rank == 0:
                    self.memory_profiler.snapshot(f"Step {step}")
                    self.profiler.step()

                # --- Explicitly delete tensors at the very end of the loop iteration --- #
                # Do this on all ranks
                try:
                    del inputs
                    del targets
                    if "loss" in locals() and loss is not None:
                        del loss
                    if "feature_activations_batch" in locals():
                        del feature_activations_batch
                except NameError:
                    pass

        except KeyboardInterrupt:
            if not self.distributed or self.rank == 0:
                logger.info("\nTraining interrupted by user.")
        finally:
            if isinstance(pbar, tqdm):
                pbar.close()
            if not self.distributed or self.rank == 0:
                logger.info(f"Training loop finished at step {step}.")

        # Sync before final save attempt
        if self.distributed:
            dist.barrier()

        # --- Save final model and metrics --- (Rank 0 handles metrics/store, all ranks save model state)
        final_checkpoint_dir = os.path.join(self.log_dir, "final")
        final_store_path = os.path.join(final_checkpoint_dir, "activation_store_final.pt")

        # All ranks save final model state
        try:
            # final_model_state_dict = self.model.state_dict() # Not strictly needed here if CheckpointManager handles it
            final_trainer_state_for_checkpoint = {
                "step": step,  # Use the actual last completed step
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state_dict": self.scaler.state_dict() if self.scaler and self.scaler.is_enabled() else None,
                "n_forward_passes_since_fired": (
                    self.n_forward_passes_since_fired.cpu() if self.n_forward_passes_since_fired is not None else None
                ),
                "wandb_run_id": self.wandb_logger.get_current_wandb_run_id(),
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
            }
            self.checkpoint_manager._save_checkpoint(
                step=step,  # Save at the actual last completed step
                trainer_state_to_save=final_trainer_state_for_checkpoint,
            )
        except IOError as e:  # More specific: catch IOError for checkpoint saving
            logger.warning(
                f"Rank {self.rank}: Warning: Failed to save final distributed model state due to IOError: {e}"
            )
        except Exception as e:  # Catch other potential errors during final save but log them as more critical
            logger.critical(f"Rank {self.rank}: CRITICAL: Unexpected error during final model state save: {e}")

        # Rank 0 saves store, metrics, logs artifact
        if not self.distributed or self.rank == 0:
            logger.info(f"Saving final activation store state to {final_store_path}...")
            os.makedirs(final_checkpoint_dir, exist_ok=True)
            try:
                # Check if the store has a close method before calling (for compatibility)
                if hasattr(self.activation_store, "close") and callable(getattr(self.activation_store, "close")):
                    self.activation_store.close()
            except IOError as e:  # More specific: catch IOError for store closing
                logger.warning(f"Rank 0: Warning: Failed to close activation store due to IOError: {e}")
            except Exception as e:  # Catch other potential errors during store close
                logger.warning(f"Rank 0: Warning: Unexpected error closing activation store: {e}")

            logger.info("Saving final metrics...")
            # self.metric_logger._save_metrics_to_disk() # Final save - this should be robust
            try:
                self.metric_logger._save_metrics_to_disk()
            except IOError as e:
                logger.warning(f"Rank 0: Warning: Failed to save final metrics to disk due to IOError: {e}")
            except Exception as e:
                logger.warning(f"Rank 0: Warning: Unexpected error saving final metrics: {e}")

            # --- Save CLT Config to JSON ---
            # The config saved here will now reflect the configuration *during training* (e.g. BatchTopK)
            # The user will need to run estimate_theta_posthoc and then save the converted JumpReLU model themselves.
            config_save_path = os.path.join(self.log_dir, "cfg.json")
            logger.info(f"Saving CLT configuration (as trained) to {config_save_path}...")
            try:
                config_dict_as_trained = asdict(self.clt_config)

                # Attempt to add model_name if available in clt_config itself
                if hasattr(self.clt_config, "model_name") and self.clt_config.model_name:
                    config_dict_as_trained["model_name"] = self.clt_config.model_name

                if hasattr(self.training_config, "normalization_method"):
                    config_dict_as_trained["normalization_method"] = self.training_config.normalization_method
                if hasattr(self.training_config, "activation_dtype"):
                    config_dict_as_trained["expected_input_dtype"] = self.training_config.activation_dtype

                # Add hook templates if available directly in clt_config
                if hasattr(self.clt_config, "mlp_input_template") and self.clt_config.mlp_input_template:
                    config_dict_as_trained["mlp_input_template"] = self.clt_config.mlp_input_template
                if hasattr(self.clt_config, "mlp_output_template") and self.clt_config.mlp_output_template:
                    config_dict_as_trained["mlp_output_template"] = self.clt_config.mlp_output_template

                with open(config_save_path, "w") as f:
                    json.dump(config_dict_as_trained, f, indent=2)
                logger.info(f"Successfully saved training configuration to {config_save_path}")
                if self.clt_config.activation_fn == "batchtopk":
                    logger.info(
                        "NOTE: Model was trained with BatchTopK. Run estimate_theta_posthoc() on the saved model to convert to JumpReLU and finalize theta values."
                    )

            except IOError as e:  # More specific: catch IOError for config saving
                logger.warning(f"Rank 0: Warning: Failed to save CLT configuration to JSON due to IOError: {e}")
            except Exception as e:  # Catch other potential errors during config saving
                logger.warning(f"Rank 0: Warning: Unexpected error saving CLT configuration to JSON: {e}")
            # --- End Save CLT Config ---

            # Log final checkpoint directory as artifact
            self.wandb_logger.log_artifact(artifact_path=final_checkpoint_dir, artifact_type="model", name="clt_final")
            
            # Also log the config file separately for easier access
            if os.path.exists(config_save_path):
                self.wandb_logger.log_artifact(artifact_path=config_save_path, artifact_type="config", name="clt_config_final")

            # Finish WandB logging
            self.wandb_logger.finish()
            logger.info(f"Training completed! Final checkpoint saved to {final_checkpoint_dir}")

        # --- Close the activation store (stops prefetch thread if applicable) --- #
        if hasattr(self.activation_store, "close") and callable(getattr(self.activation_store, "close")):
            self.activation_store.close()

        # Log final profiling summaries
        if not self.distributed or self.rank == 0:
            logger.info("\n" + "=" * 80)
            logger.info("FINAL PROFILING SUMMARY")
            logger.info("=" * 80)
            self.profiler.log_summary()
            self.memory_profiler.log_summary()
            if self.distributed:
                self.dist_profiler.log_summary()
            logger.info("=" * 80)

        # Clean up distributed process group
        if self.distributed:
            dist.destroy_process_group()

        return self.model
