import torch
import torch.optim as optim
from typing import Dict, Literal, Optional, Union, Any
from tqdm import tqdm  # type: ignore
import os
import json
import time
import importlib.util
import sys
import logging  # Add logging import
import datetime  # Import datetime for formatting
import torch.distributed as dist  # Import torch.distributed
from torch.distributed import ProcessGroup  # Import ProcessGroup
from torch.distributed.checkpoint.state_dict_saver import save_state_dict
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner, DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemWriter, FileSystemReader  # Storage for checkpointing

from clt.config import CLTConfig, TrainingConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.data import (
    BaseActivationStore,
    StreamingActivationStore,
    # MappedActivationStore, # Removed legacy store
)

# Import the new manifest-based stores
from clt.training.local_activation_store import LocalActivationStore
from clt.training.remote_activation_store import RemoteActivationStore

from clt.training.losses import LossManager
from clt.nnsight.extractor import (
    ActivationExtractorCLT,
)
from clt.training.trainer import CLTTrainer  # Keep for StreamingStore usage
from .evaluator import CLTEvaluator  # Import the new evaluator


# Get logger for this module
logger = logging.getLogger(__name__)


# Helper function to format elapsed time
def _format_elapsed_time(seconds: float) -> str:
    """Formats elapsed seconds into HH:MM:SS or MM:SS."""
    td = datetime.timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if td.days > 0 or hours > 0:
        return f"{td.days * 24 + hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


# # Define the dummy logger class explicitly for better type checking
# class DummyWandBLogger:
#     def log_step(self, *args, **kwargs):
#         pass

#     def log_evaluation(self, *args, **kwargs):
#         pass

#     def log_artifact(self, *args, **kwargs):
#         pass

#     def finish(self, *args, **kwargs):
#         pass


# class WandBLogger:
#     """Wrapper class for Weights & Biases logging."""

#     def __init__(self, training_config: TrainingConfig, clt_config: CLTConfig, log_dir: str):
#         """Initialize the WandB logger.

#         Args:
#             training_config: Training configuration
#             clt_config: CLT model configuration
#             log_dir: Directory to save logs
#         """
#         self.enabled = training_config.enable_wandb
#         self.log_dir = log_dir

#         if not self.enabled:
#             return

#         # Check if wandb is installed
#         if not importlib.util.find_spec("wandb"):
#             print(
#                 "Warning: WandB logging requested but wandb not installed. "
#                 "Install with 'pip install wandb'. Continuing without WandB."
#             )
#             self.enabled = False
#             return

#         # Import wandb
#         import wandb

#         # Set up run name with timestamp if not provided
#         run_name = training_config.wandb_run_name
#         if run_name is None:
#             run_name = f"clt-{time.strftime('%Y%m%d-%H%M%S')}"

#         # Initialize wandb
#         wandb.init(
#             project=training_config.wandb_project,
#             entity=training_config.wandb_entity,
#             name=run_name,
#             dir=log_dir,
#             tags=training_config.wandb_tags,
#             config={
#                 **training_config.__dict__,
#                 **clt_config.__dict__,
#                 "log_dir": log_dir,
#             },
#         )

#         if wandb.run is not None:
#             print(f"WandB logging initialized: {wandb.run.name}")

#     def log_step(
#         self,
#         step: int,
#         loss_dict: Dict[str, float],
#         lr: Optional[float] = None,
#         sparsity_lambda: Optional[float] = None,
#         total_tokens_processed: Optional[int] = None,
#     ):
#         """Log metrics for a training step under the 'training/' group.

#         Args:
#             step: Current training step
#             loss_dict: Dictionary of loss values (e.g., total, reconstruction, sparsity)
#             lr: Current learning rate
#             sparsity_lambda: Current sparsity coefficient lambda
#             total_tokens_processed: Total tokens processed up to this step
#         """
#         if not self.enabled:
#             return

#         import wandb

#         # Rename loss keys for clarity and add 'training/' prefix
#         metrics = {}
#         for key, value in loss_dict.items():
#             if key == "total":
#                 metrics["training/total_loss"] = value
#             elif key == "sparsity":
#                 metrics["training/sparsity_loss"] = value
#             elif key == "reconstruction":
#                 # Reconstruction loss is part of training, log it here too if present
#                 metrics["training/reconstruction_loss"] = value
#             elif key == "preactivation":
#                 metrics["training/preactivation_loss"] = value
#             else:
#                 # Keep other potential keys, prepending 'training/'
#                 metrics[f"training/{key}"] = value

#         # Add learning rate
#         if lr is not None:
#             metrics["training/learning_rate"] = lr

#         # Add sparsity lambda
#         if sparsity_lambda is not None:
#             metrics["training/sparsity_lambda"] = sparsity_lambda

#         # Add total tokens processed
#         if total_tokens_processed is not None:
#             metrics["training/total_tokens_processed"] = total_tokens_processed

#         # Log to wandb
#         wandb.log(metrics, step=step)

#     def log_evaluation(self, step: int, eval_metrics: Dict[str, Any]):
#         """Log evaluation metrics, organized by the structure from CLTEvaluator.

#         Args:
#             step: Current training step
#             eval_metrics: Dictionary of evaluation metrics from CLTEvaluator
#                           (keys like 'reconstruction/', 'sparsity/', 'layerwise/')
#         """
#         if not self.enabled:
#             return

#         import wandb

#         # Log metrics directly, assuming keys are already structured
#         # e.g., 'reconstruction/mse', 'sparsity/avg_l0', 'layerwise/l0/layer_0'
#         wandb_log_dict: Dict[str, Any] = {}
#         for key, value in eval_metrics.items():
#             if key.startswith("layerwise/"):
#                 # Handle nested layerwise data (histograms and scalars)
#                 # layerwise_category = key.split("/")[
#                 #     1
#                 # ]  # e.g., 'l0', 'log_feature_density' # Removed unused variable
#                 if isinstance(value, dict):
#                     for layer_key, layer_value in value.items():
#                         # Construct wandb key: e.g., layerwise/l0/layer_0
#                         wandb_key = f"{key}/{layer_key}"  # Correctly forms e.g. layerwise/log_feature_density/layer_0
#                         if isinstance(layer_value, list):
#                             # Log list data as histogram
#                             try:
#                                 wandb_log_dict[wandb_key] = wandb.Histogram(layer_value)
#                             except Exception as e:
#                                 print(f"Wandb: Error creating histogram for {wandb_key}: {e}")
#                                 # Fallback: log mean or placeholder
#                                 try:
#                                     mean_val = sum(layer_value) / len(layer_value) if layer_value else 0.0
#                                     wandb_log_dict[f"{wandb_key}_mean"] = mean_val
#                                 except TypeError:
#                                     wandb_log_dict[f"{wandb_key}_mean"] = -1.0
#                         elif isinstance(layer_value, (float, int)):
#                             # Log scalar layerwise data
#                             wandb_log_dict[wandb_key] = layer_value
#                 else:
#                     # If the top level key itself is scalar (shouldn't happen with current structure)
#                     wandb_log_dict[key] = value
#             elif key.endswith("_agg_hist") and isinstance(value, list):
#                 # Handle aggregate histogram data (e.g., sparsity/log_feature_density_agg_hist)
#                 try:
#                     wandb_log_dict[key] = wandb.Histogram(value)
#                 except Exception as e:
#                     print(f"Wandb: Error creating aggregate histogram for {key}: {e}")
#                     # Optional Fallback: log mean of aggregate data
#                     try:
#                         mean_val = sum(value) / len(value) if value else 0.0
#                         wandb_log_dict[f"{key}_mean"] = mean_val
#                     except TypeError:
#                         wandb_log_dict[f"{key}_mean"] = -1.0

#             elif isinstance(value, (float, int)):  # Handle top-level scalars
#                 # Log directly, e.g., 'reconstruction/mse', 'sparsity/avg_l0', 'dead_features/total_eval'
#                 wandb_log_dict[key] = value
#             # Add other specific handling if needed (e.g., for specific non-scalar, non-layerwise data)

#         # Log the prepared dictionary to wandb
#         if wandb_log_dict:
#             wandb.log(wandb_log_dict, step=step)

#     def log_artifact(self, artifact_path: str, artifact_type: str, name: Optional[str] = None):
#         """Log an artifact to WandB.

#         Args:
#             artifact_path: Path to the artifact
#             artifact_type: Type of artifact (e.g., "model", "dataset")
#             name: Name of the artifact (defaults to filename)
#         """
#         if not self.enabled:
#             return

#         import wandb

#         # Use filename if name not provided
#         if name is None:
#             name = os.path.basename(artifact_path)

#         # Create and log artifact
#         artifact = wandb.Artifact(name=name, type=artifact_type)
#         # Check if it's a directory (for sharded checkpoints)
#         if os.path.isdir(artifact_path):
#             artifact.add_dir(artifact_path)
#         else:
#             artifact.add_file(artifact_path)
#         wandb.log_artifact(artifact)

#     def finish(self):
#         """Finish the WandB run."""
#         if not self.enabled:
#             return

#         import wandb

#         wandb.finish()


class CLTTrainerRay(CLTTrainer):
    """Trainer for Cross-Layer Transcoder models."""

    # Add type hint for the activation store attribute
    activation_store: BaseActivationStore
    # Model type hint
    model: CrossLayerTranscoder

    def __init__(
        self,
        clt_config: CLTConfig,
        training_config: TrainingConfig,
        log_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        distributed: bool = False,  # unused for Ray as of now
        use_ray: Literal['train', 'tune'] = 'tune',
    ):
        """Initialize the CLT trainer.

        Args:
            clt_config: Configuration for the CLT model
            training_config: Configuration for training
            log_dir: Directory to save logs and checkpoints
            device: Device to use for training (ignored if distributed)
            distributed: Whether to use distributed training
        """
        super().__init__(clt_config, training_config, log_dir, device, False)

        # Set up imports for Ray
        if use_ray == 'train':
            from ray.train import report
            from ray.train import Checkpoint
        elif use_ray == 'tune':
            from ray.tune import report
            from ray.tune import Checkpoint

        import tempfile

        self.ray_reporter = report
        self.ray_checkpoint = Checkpoint
        self.ray_tempfile = tempfile


    def train(self, eval_every: int = 1000) -> CrossLayerTranscoder:
        """Train the CLT model.

        Args:
            eval_every: Evaluate model every N steps

        Returns:
            Trained CLT model (local shard)
        """
        # Print startup message from rank 0 only
        if not self.distributed or self.rank == 0:
            print(f"Starting CLT training on {self.device}...")
            print(
                f"Model has {self.clt_config.num_features} features per layer "
                f"and {self.clt_config.num_layers} layers"
            )
            print(f"Training for {self.training_config.training_steps} steps.")
            print(f"Logging to {self.log_dir}")
            if self.distributed:
                print(f"Distributed training with {self.world_size} processes (Tensor Parallelism)")

            # Check if using normalization and notify user
            if self.training_config.normalization_method == "estimated_mean_std":
                print("\n>>> NORMALIZATION PHASE <<<")
                print("Normalization statistics are being estimated from dataset activations.")
                print("This may take some time, but happens only once before training begins.")
                print(f"Using {self.training_config.normalization_estimation_batches} batches for estimation.\n")

            # Make sure we flush stdout to ensure prints appear immediately,
            # especially important in Jupyter/interactive environments
            sys.stdout.flush()
            # Wait for 1 second to ensure output is displayed before training starts
            time.sleep(1)
            print("\n>>> TRAINING PHASE <<<")
            sys.stdout.flush()

        # # After the existing startup messages
        # if self.distributed:
        #     print("\n!!! DIAGNOSTIC INFO !!!")
        #     print(f"Rank {self.rank}: Process group type: {type(self.process_group)}")
        #     print(f"Rank {self.rank}: RowParallelLinear _reduce does NOT divide by world_size")
        #     print(f"Rank {self.rank}: Using weight regularization in sparsity penalty")
        #     print(f"Rank {self.rank}: Averaging replicated parameter gradients")
        #     # Check if activation store has rank/world attributes before accessing
        #     store_rank = getattr(self.activation_store, "rank", "N/A")
        #     store_world = getattr(self.activation_store, "world", "N/A")
        #     print(f"Rank {self.rank}: Data sharding: rank={store_rank}, world={store_world}")
        #     print(f"Rank {self.rank}: Batch size tokens: {self.training_config.train_batch_size_tokens}")
        #     print(f"Rank {self.rank}: Sparsity lambda: {self.training_config.sparsity_lambda}")

        #     # Check if activation store actually loaded correctly
        #     batch_avail = next(iter(self.activation_store), None)
        #     print(f"Rank {self.rank}: First batch available: {batch_avail is not None}")

        #     # Force torch to compile/execute our code by running a tiny forward/backward pass
        #     dummy = torch.ones(1, device=self.device, requires_grad=True)
        #     dummy_out = dummy * 2
        #     dummy_out.backward()
        #     print("!!! END DIAGNOSTIC !!!\n")

        # Create progress bar only on rank 0
        pbar: Union[tqdm, range]
        if not self.distributed or self.rank == 0:
            pbar = tqdm(
                range(self.training_config.training_steps),
                desc="Training CLT",
                leave=True,
            )
        else:
            pbar = range(self.training_config.training_steps)

        step = 0
        try:
            for step in pbar:
                # Refresh progress bar on rank 0
                step_start_time = time.monotonic()  # Start timing the step
                if isinstance(pbar, tqdm):
                    pbar.refresh()

                try:
                    # Get batch directly from the iterator (handles distributed sampling internally)
                    batch_get_start_time = time.monotonic()
                    inputs, targets = next(self.activation_store)
                    batch_get_duration = time.monotonic() - batch_get_start_time
                    logger.debug(f"Rank {self.rank} Step {step}: Getting batch took {batch_get_duration:.4f}s")

                except StopIteration:
                    # Rank 0 prints message
                    if not self.distributed or self.rank == 0:
                        print("Activation store exhausted. Training finished early.")
                    if self.distributed:
                        dist.barrier()  # Ensure all ranks see this
                    break  # Exit training loop if data runs out
                except Exception as e:
                    # Rank 0 prints message
                    if not self.distributed or self.rank == 0:
                        print(f"\nRank {self.rank}: Error getting batch at step {step}: {e}. Skipping step.")
                    # Maybe barrier here too? If one rank fails, others might hang?
                    # Let's continue for now, assuming store handles internal errors.
                    continue

                # --- Check for empty batch --- (Optional but good practice)
                # This check should ideally happen *before* moving data potentially
                if not inputs or not targets or not any(v.numel() > 0 for v in inputs.values()):
                    if not self.distributed or self.rank == 0:
                        print(f"\nRank {self.rank}: Warning: Received empty batch at step {step}. Skipping.")
                    continue

                # --- BEGIN: One-time Normalization Check ---
                if step == 0 and (not self.distributed or self.rank == 0):
                    logger.info("--- Running Post-Normalization Check (First Batch) ---")
                    norm_applied = getattr(self.activation_store, "apply_normalization", None)
                    if isinstance(self.activation_store, (LocalActivationStore, RemoteActivationStore)):
                        logger.info(f"ActivationStore reports apply_normalization={norm_applied}")
                    elif isinstance(self.activation_store, StreamingActivationStore):
                        logger.info(
                            f"Streaming store normalization method: {self.activation_store.normalization_method}"
                        )

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
                self.optimizer.zero_grad()

                # Compute feature activations **once** per step to avoid redundant encoder forward passes.
                feature_activations_batch = self.model.get_feature_activations(inputs)

                # Compute total loss using the pre-computed activations
                loss, loss_dict = self.loss_manager.compute_total_loss(
                    self.model,
                    inputs,
                    targets,
                    step,
                    self.training_config.training_steps,
                    precomputed_activations=feature_activations_batch,
                )

                # --- Update Dead Neuron Counters --- (All ranks, counter is replicated)
                # We need *full* feature activations *after* non-linearity
                if hasattr(self, "n_forward_passes_since_fired"):
                    with torch.no_grad():
                        for layer_idx, layer_acts in feature_activations_batch.items():
                            # Ensure layer index is within bounds of the counter tensor
                            if layer_idx < self.n_forward_passes_since_fired.shape[0]:
                                if layer_acts.numel() > 0:
                                    # layer_acts shape: [batch_tokens, num_features]
                                    fired_mask_per_token = layer_acts > 1e-6
                                    fired_features_this_layer = fired_mask_per_token.any(dim=0)

                                    if fired_features_this_layer.shape[0] == self.n_forward_passes_since_fired.shape[1]:
                                        self.n_forward_passes_since_fired[layer_idx] += 1
                                        self.n_forward_passes_since_fired[layer_idx][fired_features_this_layer] = 0
                                    else:
                                        if not self.distributed or self.rank == 0:  # Only rank 0 logs warning
                                            print(
                                                f"Rank {self.rank}: Warning: Shape mismatch for dead neuron update at layer {layer_idx}. "
                                                f"Acts shape: {layer_acts.shape}, Fired mask: {fired_features_this_layer.shape}, "
                                                f"Counter: {self.n_forward_passes_since_fired.shape}"
                                            )

                # --- Backward pass --- (All ranks, handles communication implicitly)
                if torch.isnan(loss):
                    if not self.distributed or self.rank == 0:
                        print(
                            f"\nRank {self.rank}: Warning: NaN loss encountered at step {step}. "
                            f"Skipping backward pass and optimizer step."
                        )
                else:
                    try:
                        loss.backward()

                        # --- Synchronise gradients of replicated parameters --- #
                        self._average_shared_parameter_grads()

                        # --- Gradient clipping --- #
                        if self.training_config.gradient_clip_val is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.training_config.gradient_clip_val,
                            )
                    except RuntimeError as e:
                        if not self.distributed or self.rank == 0:
                            print(
                                f"\nRank {self.rank}: Error during backward pass at step {step}: {e}. Skipping optimizer step."
                            )
                        continue

                    # --- Optimizer step --- (Applied to local parameters using local gradients)
                    self.optimizer.step()

                    # --- Invalidate Caches --- #
                    if hasattr(self.model, "_cached_decoder_norms"):
                        self.model._cached_decoder_norms = None

                # --- Scheduler step --- (All ranks)
                if self.scheduler:
                    self.scheduler.step()

                # --- Update progress bar --- (Rank 0 only)
                if isinstance(pbar, tqdm):
                    description = (
                        f"Loss: {loss_dict.get('total', float('nan')):.4f} "
                        f"(R: {loss_dict.get('reconstruction', float('nan')):.4f} "
                        f"S: {loss_dict.get('sparsity', float('nan')):.4f} "
                        f"P: {loss_dict.get('preactivation', float('nan')):.4f})"
                    )
                    pbar.set_description(description)
                    # Force update to display progress
                    if step % 1 == 0:  # Update every step
                        pbar.refresh()
                        sys.stdout.flush()

                # --- Log metrics --- (Rank 0 logs to WandB/file)
                # self._log_metrics(step, loss_dict)

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
                    # if self.distributed:
                    #     dist.barrier()  # Sync before evaluation starts so that all ranks enter together

                    # Compute evaluation metrics on all ranks to keep collective ops aligned
                    current_dead_mask = self.dead_neurons_mask.detach().clone()
                    eval_metrics = self.evaluator.compute_metrics(
                        inputs,
                        targets,
                        dead_neuron_mask=current_dead_mask,
                    )

                    if not self.distributed or self.rank == 0:
                        # # Store evaluation metrics (for saving to JSON)
                        # self.metrics["eval_metrics"].append({"step": step, **eval_metrics})

                        # --- Update Progress Bar Postfix ---
                        l0_str = f"AvgL0: {eval_metrics.get('sparsity/avg_l0', 0.0):.2f}"
                        ev_str = f"EV: {eval_metrics.get('reconstruction/explained_variance', 0.0):.3f}"
                        avg_density_mean = eval_metrics.get("sparsity/feature_density_mean")
                        dens_str = f"Dens: {avg_density_mean:.3f}" if avg_density_mean is not None else "Dens: N/A"
                        eval_dead_str = f"Dead(Eval): {eval_metrics.get('dead_features/total_eval', 0)}"
                        eval_msg = f"{l0_str}, {ev_str}, {dens_str}, {eval_dead_str}"

                        if isinstance(pbar, tqdm):
                            pbar.set_postfix_str(eval_msg)
                            pbar.refresh()

                        # # --- Log evaluation metrics to WandB ---
                        # self.wandb_logger.log_evaluation(step, eval_metrics)

                        # # --- Save metrics JSON after evaluation ---
                        # self._save_metrics()

                    # Optionally compute and log sparsity diagnostics (can be slow)
                    if self.training_config.compute_sparsity_diagnostics:
                        # Calculate diagnostics using the same batch data and cached activations/norms
                        sparsity_diag_metrics = self._compute_sparsity_diagnostics(inputs, feature_activations_batch)
                        # Merge diagnostics into the main eval metrics dict
                        if sparsity_diag_metrics:
                            eval_metrics.update(sparsity_diag_metrics)
                            # Log updated metrics to WandB (only rank 0)
                            # if not self.distributed or self.rank == 0:
                            #     self.wandb_logger.log_evaluation(step, eval_metrics)

                    # # Ensure all ranks finish evaluation before proceeding
                    # if self.distributed:
                    #     dist.barrier()

                # --- Checkpointing & Reporting ---
                # Make metrics dict to report
                ray_metrics = loss_dict

                # Add hp info
                if self.scheduler is not None:
                    # Assuming one parameter group
                    current_lr = self.scheduler.get_last_lr()[0]

                current_lambda = self.loss_manager.get_current_sparsity_lambda()

                ray_metrics |= {
                    'lr': current_lr,
                    'sparsity_lambda': current_lambda
                }

                if run_eval_flag:
                    ray_metrics |= eval_metrics

                # Now report to ray and potentially checkpoint
                with self.ray_tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    checkpoint = None

                    if save_checkpoint_flag:
                        model_checkpoint_path = os.path.join(temp_checkpoint_dir, f"clt_checkpoint_{step}.pt")
                        store_checkpoint_path = os.path.join(temp_checkpoint_dir, f"activation_store_checkpoint_{step}.pt")

                        torch.save(self.model.state_dict(), model_checkpoint_path)
                        # TODO: wandb_logger untested with ray reporting
                        # self.wandb_logger.log_artifact(
                        #     artifact_path=model_checkpoint_path, artifact_type="model", name=f"clt_checkpoint_{step}"
                        # )
                        torch.save(self.activation_store.state_dict(), store_checkpoint_path)

                        checkpoint = self.ray_checkpoint.from_directory(temp_checkpoint_dir)

                    self.ray_reporter(ray_metrics, checkpoint=checkpoint)

            # # --- Explicitly delete tensors at the very end of the loop iteration --- #
            # # Do this on all ranks
            # try:
            #     del inputs
            #     del targets
            #     if "loss" in locals() and loss is not None:
            #         del loss
            #     if "feature_activations_batch" in locals():
            #         del feature_activations_batch
            # except NameError:
            #     pass

        except KeyboardInterrupt:
            if not self.distributed or self.rank == 0:
                print("\nTraining interrupted by user.")
        finally:
            if isinstance(pbar, tqdm):
                pbar.close()
            if not self.distributed or self.rank == 0:
                print(f"Training loop finished at step {step}.")

        # # Sync before final save attempt
        # if self.distributed:
        #     dist.barrier()

        # # --- Save final model and metrics --- (Rank 0 handles metrics/store, all ranks save model state)
        # final_checkpoint_dir = os.path.join(self.log_dir, "final")
        # final_store_path = os.path.join(final_checkpoint_dir, "activation_store_final.pt")  # Store inside final dir

        # # All ranks save final model state
        # try:
        #     final_model_state_dict = self.model.state_dict()
        #     save_state_dict(
        #         state_dict=final_model_state_dict,
        #         storage_writer=FileSystemWriter(final_checkpoint_dir),
        #         planner=DefaultSavePlanner(),
        #         no_dist=(not self.distributed),  # Disable distributed save if not distributed
        #     )
        # except Exception as e:
        #     print(f"Rank {self.rank}: Warning: Failed to save final distributed model state: {e}")

        # # Rank 0 saves store, metrics, logs artifact
        # if not self.distributed or self.rank == 0:
        #     print(f"Saving final activation store state to {final_store_path}...")
        #     os.makedirs(final_checkpoint_dir, exist_ok=True)  # Ensure dir exists for store save
        #     try:
        #         # Check if the store has a close method before calling (for compatibility)
        #         if hasattr(self.activation_store, "close") and callable(getattr(self.activation_store, "close")):
        #             self.activation_store.close()
        #     except Exception as e:
        #         print(f"Rank 0: Warning: Failed to close activation store: {e}")

        #     print("Saving final metrics...")
        #     self._save_metrics()

        #     # Log final checkpoint directory as artifact
        #     self.wandb_logger.log_artifact(artifact_path=final_checkpoint_dir, artifact_type="model", name="clt_final")

        #     # Finish WandB logging
        #     self.wandb_logger.finish()
        #     print(f"Training completed! Final checkpoint saved to {final_checkpoint_dir}")

        # --- Close the activation store (stops prefetch thread if applicable) --- #
        if hasattr(self.activation_store, "close") and callable(getattr(self.activation_store, "close")):
            self.activation_store.close()

        # Clean up distributed process group
        if self.distributed:
            dist.destroy_process_group()

        return self.model
