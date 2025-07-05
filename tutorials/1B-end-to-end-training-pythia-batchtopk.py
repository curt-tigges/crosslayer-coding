# %% [markdown]
# # Tutorial 2: End-to-End CLT Training with BatchTopK Activation
#
# This tutorial demonstrates training a Cross-Layer Transcoder (CLT)
# using the **BatchTopK** activation function. We will:
# 1. Configure the CLT model for BatchTopK, activation generation, and training parameters.
# 2. Generate activations locally (with manifest) using the ActivationGenerator.
# 3. Configure the trainer to use the locally stored activations via the manifest.
# 4. Train the CLT model using BatchTopK activation.
# 5. Save and load the final trained model (which will be JumpReLU if converted).
# 6. Load a model from a distributed checkpoint.
# 7. Perform a post-hoc conversion sweep (θ scaling) on a BatchTopK checkpoint.

# %% [markdown]
# ## 1. Imports and Setup
#
# First, let's import the necessary components and set up the device.

# %%
import torch
import os
import time
import sys
import traceback
import json
from torch.distributions.normal import Normal  # For post-hoc sweep
from torch.distributed.checkpoint import load_state_dict as dist_load_state_dict
from torch.distributed.checkpoint.filesystem import FileSystemReader
from typing import Optional, Dict
import logging  # Import logging

# Configure logging to show INFO level messages for the notebook
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s")

# Import from torch.distributed.checkpoint and related modules later, only when needed for that specific section
# from torch.distributed.checkpoint import load_state_dict
# from torch.distributed.checkpoint.filesystem import FileSystemReader

# logging.basicConfig(level=logging.DEBUG)

# Import components from the clt library
# (Ensure the 'clt' directory is in your Python path or installed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from clt.config import CLTConfig, TrainingConfig, ActivationConfig
    from clt.activation_generation.generator import ActivationGenerator
    from clt.training.trainer import CLTTrainer
    from clt.models.clt import CrossLayerTranscoder
    from clt.training.data import BaseActivationStore
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure the 'clt' library is installed or the clt directory is in your PYTHONPATH.")
    raise

# Device setup
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Base model for activation extraction
BASE_MODEL_NAME = "EleutherAI/pythia-70m"

# For post-hoc sweep N(0,1) assumption
std_normal = Normal(0, 1)

# %% [markdown]
# ## 2. Configuration
#
# We configure the CLT, Activation Generation, and Training.
# Key change: `CLTConfig.activation_fn` is set to `"batchtopk"`.

# %%
# --- CLT Architecture Configuration ---
num_layers = 6
d_model = 512
expansion_factor = 32
clt_num_features = d_model * expansion_factor

batchtopk_k = 33

clt_config = CLTConfig(
    num_features=clt_num_features,
    num_layers=num_layers,
    d_model=d_model,
    activation_fn="batchtopk",  # Use BatchTopK activation
    batchtopk_k=batchtopk_k,  # Specify k directly
    batchtopk_straight_through=True,  # Use STE for gradients
    topk_mode="per_layer",  # Enable per-layer mode
)
print("CLT Configuration (BatchTopK):")
print(clt_config)

# --- Activation Generation Configuration ---
# Same as before - generate activations to train on
# Use the same directory as the first tutorial, since generation is independent of CLT activation fn
activation_dir = "./tutorial_activations_local_1M_pythia"  # Point back to original activations
dataset_name = "monology/pile-uncopyrighted"
activation_config = ActivationConfig(
    # Model Source
    model_name=BASE_MODEL_NAME,
    mlp_input_module_path_template="gpt_neox.layers.{}.mlp.input",
    mlp_output_module_path_template="gpt_neox.layers.{}.mlp.output",
    model_dtype=None,
    # Dataset Source
    dataset_path=dataset_name,
    dataset_split="train",
    dataset_text_column="text",
    # Generation Parameters
    context_size=128,
    inference_batch_size=192,
    exclude_special_tokens=True,
    prepend_bos=True,
    # Dataset Handling
    streaming=True,
    dataset_trust_remote_code=False,
    cache_path=None,
    # Generation Output Control
    target_total_tokens=1_000_000,  # Keep it small for tutorial
    # Storage Parameters
    activation_dir=activation_dir,
    output_format="hdf5",
    compression="gzip",
    chunk_token_threshold=8_000,
    activation_dtype="float32",
    # Normalization
    compute_norm_stats=True,
    # NNsight args
    nnsight_tracer_kwargs={},
    nnsight_invoker_args={},
)
print("Activation Generation Configuration:")
print(activation_config)

# --- Training Configuration ---
expected_activation_path = os.path.join(
    activation_config.activation_dir,
    activation_config.model_name,
    f"{os.path.basename(activation_config.dataset_path)}_{activation_config.dataset_split}",
)

# --- Determine WandB Run Name (using config values) ---
_lr = 1e-4
_batch_size = 1024
_k_int = clt_config.batchtopk_k  # ADDED: Use k for name

wdb_run_name = (
    f"{clt_config.num_features}-width-"
    f"batchtopk-k{_k_int}-"  # ADDED: Indicate BatchTopK and k
    f"{_batch_size}-batch-"
    f"{_lr:.1e}-lr"
    # Sparsity lambda/c less relevant when apply_sparsity_penalty_to_batchtopk=False
)
print("\nGenerated WandB run name: " + wdb_run_name)

training_config = TrainingConfig(
    # Training loop parameters
    learning_rate=_lr,
    training_steps=1000,  # Reduced steps for tutorial
    seed=42,
    # Activation source
    activation_source="local_manifest",
    activation_path=expected_activation_path,
    activation_dtype="float32",
    # Training batch size
    train_batch_size_tokens=_batch_size,
    sampling_strategy="sequential",
    # Normalization
    normalization_method="mean_std",  # Use pre-calculated stats
    # Loss function coefficients
    sparsity_lambda=0.0,  # Disable standard sparsity penalty
    sparsity_lambda_schedule="linear",
    sparsity_c=0.0,  # Disable standard sparsity penalty
    preactivation_coef=0,  # Disable preactivation loss (AuxK handles dead latents)
    aux_loss_factor=1 / 32,  # Enable AuxK loss with typical factor from paper
    apply_sparsity_penalty_to_batchtopk=False,  # Ensure standard sparsity penalty is off for BatchTopK
    # Optimizer & Scheduler
    optimizer="adamw",
    lr_scheduler="linear_final20",
    optimizer_beta2=0.98,
    # Logging & Checkpointing
    log_interval=10,
    eval_interval=50,
    diag_every_n_eval_steps=1,  # run diagnostics every eval
    max_features_for_diag_hist=1000,  # optional cap per layer
    checkpoint_interval=500,
    dead_feature_window=200,
    # WandB (Optional)
    enable_wandb=True,
    wandb_project="clt-hp-sweeps-pythia-70m",
    wandb_run_name=wdb_run_name,
)
print("\nTraining Configuration (BatchTopK):")
print(training_config)


# %% [markdown]
# ## 3. Generate Activations (One-Time Step)
#
# Generate the activation dataset, including the manifest file (`index.bin`). This step is the same
# as in the previous tutorial, just saving to a different directory (`activation_dir`).

# %%
print("Step 1: Generating/Verifying Activations (including manifest)...")

metadata_path = os.path.join(expected_activation_path, "metadata.json")
manifest_path = os.path.join(expected_activation_path, "index.bin")

if os.path.exists(metadata_path) and os.path.exists(manifest_path):
    print(f"Activations and manifest already found at: {expected_activation_path}")
    print("Skipping generation. Delete the directory to regenerate.")
else:
    print(f"Activations or manifest not found. Generating them now at: {expected_activation_path}")
    try:
        generator = ActivationGenerator(
            cfg=activation_config,
            device=device,
        )
        generation_start_time = time.time()
        generator.generate_and_save()
        generation_end_time = time.time()
        print(f"Activation generation complete in {generation_end_time - generation_start_time:.2f}s.")
    except Exception as gen_err:
        print(f"[ERROR] Activation generation failed: {gen_err}")
        traceback.print_exc()
        raise

# %% [markdown]
# ## 4. Training the CLT with BatchTopK Activation
#
# Instantiate the `CLTTrainer` using the configurations defined above.
# The trainer will use the `LocalActivationStore` (driven by the manifest) and the CLT model
# will use the BatchTopK activation function internally based on `clt_config`.

# %%
print("Initializing CLTTrainer for training with BatchTopK...")

log_dir = f"clt_training_logs/clt_pythia_batchtopk_train_{int(time.time())}"
os.makedirs(log_dir, exist_ok=True)
print(f"Logs and checkpoints will be saved to: {log_dir}")

try:
    print("Creating CLTTrainer instance...")
    print(f"- Using device: {device}")
    print(f"- CLT config (BatchTopK): {vars(clt_config)}")
    print(f"- Activation Source: {training_config.activation_source}")
    print(f"- Reading activations from: {training_config.activation_path}")

    trainer = CLTTrainer(
        clt_config=clt_config,
        training_config=training_config,
        log_dir=log_dir,
        device=device,
        distributed=False,
    )
    print("CLTTrainer instance created successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize CLTTrainer: {e}")
    traceback.print_exc()
    raise

# Start training
print("Beginning training using BatchTopK activation...")
print(f"Training for {training_config.training_steps} steps.")
print(f"Normalization method set to: {training_config.normalization_method}")
print(
    f"Standard sparsity penalty applied to BatchTopK activations: {training_config.apply_sparsity_penalty_to_batchtopk}"
)

try:
    start_train_time = time.time()
    trained_clt_model = trainer.train(eval_every=training_config.eval_interval)
    end_train_time = time.time()
    print(f"Training finished in {end_train_time - start_train_time:.2f} seconds.")
except Exception as train_err:
    print(f"[ERROR] Training failed: {train_err}")
    traceback.print_exc()
    raise

# %% [markdown]
# ## 5. Saving and Loading the Final Trained Model
#
# The `CLTTrainer` automatically saves the final model and its configuration (cfg.json)
# in the `log_dir/final/` directory. If the training started with BatchTopK,
# the trainer converts the model to JumpReLU before this final save.
# Here, we'll also demonstrate a manual save of the model state and its config as Python dict,
# and then load it back. This manually saved model will be the one returned by trainer.train(),
# so it will also be JumpReLU if conversion occurred.

# %%
# The trained_clt_model is what trainer.train() returned.
# If clt_config.activation_fn was 'batchtopk', trainer.train() converts it to JumpReLU in-place.
final_model_state_path = os.path.join(log_dir, "clt_final_manual_state.pt")
final_model_config_path = os.path.join(log_dir, "clt_final_manual_config.json")

print(f"\nManually saving final model state to: {final_model_state_path}")
print(f"Manually saving final model config to: {final_model_config_path}")

torch.save(trained_clt_model.state_dict(), final_model_state_path)
with open(final_model_config_path, "w") as f:
    # The config on trained_clt_model will reflect 'jumprelu' if conversion happened
    json.dump(trained_clt_model.config.__dict__, f, indent=4)

print(f"\nContents of log directory ({log_dir}):")
for item in os.listdir(log_dir):
    print(f"- {item}")

# --- Loading the manually saved model ---
print("\nLoading the manually saved model...")

# 1. Load the saved configuration
with open(final_model_config_path, "r") as f:
    loaded_config_dict_manual = json.load(f)
loaded_clt_config_manual = CLTConfig(**loaded_config_dict_manual)

print(f"Loaded manual config, activation_fn: {loaded_clt_config_manual.activation_fn}")

# 2. Instantiate model with this loaded config and load state dict
loaded_clt_model_manual = CrossLayerTranscoder(
    config=loaded_clt_config_manual,
    process_group=None,  # Assuming non-distributed for this load
    device=torch.device(device),
)
loaded_clt_model_manual.load_state_dict(torch.load(final_model_state_path, map_location=device))
loaded_clt_model_manual.eval()  # Set to evaluation mode

print("Manually saved model loaded successfully.")
print(f"Loaded model is on device: {next(loaded_clt_model_manual.parameters()).device}")


# %% [markdown]
# ## 6. Loading from Distributed Checkpoint (DC)
#
# The trainer saves checkpoints in a distributed-compatible format (using `torch.distributed.checkpoint`)
# in `log_dir/step_<N>/` and `log_dir/final/`. We can load the `final` one.
# This model will also be in JumpReLU format if the original training was BatchTopK.

# %%
# Imports moved to top:
# from torch.distributed.checkpoint import load_state_dict as dist_load_state_dict
# from torch.distributed.checkpoint.filesystem import FileSystemReader

# Path to the 'final' directory created by the trainer
# This contains the sharded checkpoint and the cfg.json (which reflects JumpReLU if converted)
dc_final_checkpoint_dir = os.path.join(log_dir, "final")

print(f"\nLoading model from distributed checkpoint: {dc_final_checkpoint_dir}")

# 1. Load the config from cfg.json in that directory
dc_config_path = os.path.join(dc_final_checkpoint_dir, "cfg.json")
if not os.path.exists(dc_config_path):
    print(f"ERROR: cfg.json not found in {dc_final_checkpoint_dir}. Cannot load distributed checkpoint correctly.")
else:
    with open(dc_config_path, "r") as f:
        loaded_config_dict_dc = json.load(f)
    loaded_clt_config_dc = CLTConfig(**loaded_config_dict_dc)
    print(f"Loaded DC config, activation_fn: {loaded_clt_config_dc.activation_fn}")

    # 2. Instantiate the model with this config
    # Determine device (mps not directly supported by some distributed ops, fallback to cpu if necessary for loading)
    device_to_load_on = device if device != "mps" else "cpu"
    print(f"Instantiating model on device: {device_to_load_on} for DC load")

    model_for_dc_load = CrossLayerTranscoder(
        config=loaded_clt_config_dc,
        process_group=None,  # For non-distributed loading of a dist checkpoint
        device=torch.device(device_to_load_on),
    )
    model_for_dc_load.eval()

    # 3. Create an empty state dict and load into it
    state_dict_to_populate_dc = model_for_dc_load.state_dict()

    try:
        dist_load_state_dict(
            state_dict=state_dict_to_populate_dc,
            storage_reader=FileSystemReader(dc_final_checkpoint_dir),
            no_dist=True,  # Important for loading a sharded checkpoint into a non-distributed model
        )
        # Load the populated state dict into the model
        model_for_dc_load.load_state_dict(state_dict_to_populate_dc)
        print("Model loaded successfully from distributed checkpoint.")
        print(f"Model is on device: {next(model_for_dc_load.parameters()).device}")
    except Exception as e_dc:
        print(f"ERROR loading distributed checkpoint: {e_dc}")
        traceback.print_exc()

# %% [markdown]
# ## 7. Post-hoc Conversion Sweep (θ scaling) from a BatchTopK Checkpoint
#
# To experiment with different θ scaling factors for BatchTopK-to-JumpReLU conversion,
# we need a model checkpoint that was saved *before* any automatic conversion by the trainer.
# The trainer saves checkpoints periodically (e.g., `clt_checkpoint_500.pt`).
# We'll load one of these, assuming it's still in BatchTopK format.

# %%

# Path to a BatchTopK checkpoint (e.g., one saved mid-training)
# Ensure this checkpoint was saved when the model was still BatchTopK.
# The tutorial saves clt_checkpoint_500.pt and clt_checkpoint_latest.pt
# The trainer converts to JumpReLU only at the very end of training if the original was BatchTopK.
# So, a checkpoint from step 500 should be BatchTopK.
log_dir = "clt_training_logs/clt_pythia_batchtopk_train_1746852317"
batchtopk_checkpoint_path = os.path.join(log_dir, "clt_checkpoint_900.pt")

if not os.path.exists(batchtopk_checkpoint_path):
    print(f"WARNING: BatchTopK checkpoint {batchtopk_checkpoint_path} not found. Skipping sweep.")
    print("Ensure your training ran for at least 500 steps and saved a checkpoint.")
else:
    print(f"\nLoading BatchTopK model from checkpoint: {batchtopk_checkpoint_path} for sweep...")

    # clt_config_for_batchtopk_load is now defined INSIDE the loop below

    # 2. Load the BatchTopK model state
    batchtopk_model_state = torch.load(batchtopk_checkpoint_path, map_location=device)

    # This is the StateDict from the BatchTopK model
    # It will be used as the starting point for each conversion in the sweep.

    # std_normal is already defined at the top of the script if using the sweep code from previous turn
    from torch.distributions.normal import Normal  # Moved to top

    std_normal = Normal(0, 1)

    # Define quick_l0_checks here
    def quick_l0_checks(
        model: CrossLayerTranscoder, sample_batch_inputs: Dict[int, torch.Tensor], num_tokens_for_l0_check: int = 100
    ) -> tuple[float, float]:
        """Return (avg_empirical_l0_layer0, expected_l0)
        using an average over random tokens from sample_batch_inputs for empirical L0."""
        model.eval()
        avg_empirical_l0_layer0 = float("nan")
        std_normal_dist = torch.distributions.normal.Normal(0, 1)

        # Assume sample_batch_inputs[0] is valid if this function is called after store initialization
        layer0_inputs_all_tokens = sample_batch_inputs.get(0)  # Use .get() for safety, though we assume it exists

        if layer0_inputs_all_tokens is None or layer0_inputs_all_tokens.numel() == 0:
            print("Warning: quick_l0_checks received no valid input for layer 0. Empirical L0 will be NaN.")
        else:
            layer0_inputs_all_tokens = layer0_inputs_all_tokens.to(device=model.device, dtype=model.dtype)
            if layer0_inputs_all_tokens.dim() == 3:  # B, S, D
                num_tokens_in_batch = layer0_inputs_all_tokens.shape[0] * layer0_inputs_all_tokens.shape[1]
                layer0_inputs_flat = layer0_inputs_all_tokens.reshape(num_tokens_in_batch, model.config.d_model)
            elif layer0_inputs_all_tokens.dim() == 2:  # Already [num_tokens, d_model]
                num_tokens_in_batch = layer0_inputs_all_tokens.shape[0]
                layer0_inputs_flat = layer0_inputs_all_tokens
            else:
                print(
                    f"Warning: quick_l0_checks received unexpected input shape {layer0_inputs_all_tokens.shape} for layer 0. Empirical L0 will be NaN."
                )
                layer0_inputs_flat = None

            if layer0_inputs_flat is not None and num_tokens_in_batch > 0:
                num_to_sample = min(num_tokens_for_l0_check, num_tokens_in_batch)
                indices = torch.randperm(num_tokens_in_batch, device=model.device)[:num_to_sample]
                selected_tokens_for_l0 = layer0_inputs_flat[indices]
                if selected_tokens_for_l0.numel() > 0:
                    acts_layer0_selected = model.encode(selected_tokens_for_l0, layer_idx=0)
                    l0_per_token_selected = (acts_layer0_selected > 1e-6).sum(dim=1).float()
                    avg_empirical_l0_layer0 = l0_per_token_selected.mean().item()
                else:
                    print(
                        "Warning: No tokens selected for empirical L0 check after sampling. Empirical L0 will be NaN."
                    )
            # Removed redundant checks for layer0_inputs_flat being None or num_tokens_in_batch == 0, covered by outer if/else

        expected_l0 = float("nan")
        if hasattr(model, "log_threshold") and model.log_threshold is not None:
            theta = model.log_threshold.exp().cpu()
            p_fire = 1.0 - std_normal_dist.cdf(theta.float())
            expected_l0 = p_fire.sum().item()
        else:
            print("Warning: Model does not have log_threshold. Cannot compute expected_l0.")
        return avg_empirical_l0_layer0, expected_l0

    # Initialize LocalActivationStore for the sweep, assuming training_config is available from earlier cells
    print("Initializing LocalActivationStore for theta estimation sweep...")
    posthoc_activation_store: Optional[BaseActivationStore] = None
    try:
        from clt.training.local_activation_store import LocalActivationStore  # Ensure import

        if training_config.activation_path is None:  # This check is still good practice
            raise ValueError("training_config.activation_path is None. Cannot initialize activation store for sweep.")

        posthoc_activation_store = LocalActivationStore(
            dataset_path=training_config.activation_path,
            train_batch_size_tokens=1024,  # Can use a reasonable batch size for estimation
            device=torch.device(device),
            dtype=training_config.activation_dtype,
            rank=0,
            world=1,
            seed=42,
            sampling_strategy="sequential",
            normalization_method="auto",
        )
        print(f"Successfully initialized LocalActivationStore from: {training_config.activation_path}")
    except NameError:  # Handles case where training_config might not be defined if cells are run out of order
        print("Error: 'training_config' not defined. Please ensure previous cells initializing it have been run.")
        print("Skipping post-hoc theta scaling sweep.")
    except Exception as e_store_init:
        print(f"Error initializing LocalActivationStore for post-hoc sweep: {e_store_init}")
        print("Skipping post-hoc theta scaling sweep.")

    if posthoc_activation_store:
        scale_factors = [1.0]
        n_batches_for_theta_estimation = 1  # Number of batches to use for theta estimation

        print("\n=== θ-scaling sweep (from BatchTopK checkpoint) using estimate_theta_posthoc ===")
        print(f"Using {n_batches_for_theta_estimation} batches for theta estimation in each iteration.")

        # Import tqdm for the progress bar
        from tqdm.auto import tqdm

        for sf in tqdm(scale_factors, desc="Scaling Factor Sweep"):
            # Define clt_config_for_batchtopk_load INSIDE the loop
            # to ensure a fresh BatchTopK config for each iteration.
            clt_config_for_sweep = CLTConfig(
                num_features=16384,
                num_layers=6,
                d_model=512,
                activation_fn="batchtopk",  # Start with BatchTopK config
                batchtopk_k=batchtopk_k,  # Specify k directly
                batchtopk_straight_through=True,
                clt_dtype="float32",  # Match model dtype for consistency during load
            )

            tmp_model_for_sweep = CrossLayerTranscoder(
                config=clt_config_for_sweep,
                process_group=None,
                device=torch.device(device),
            )
            # Load the original BatchTopK state dict
            tmp_model_for_sweep.load_state_dict(batchtopk_model_state)
            tmp_model_for_sweep.eval()

            print(f"Estimating theta and converting with scale_factor = {sf:.2f}...")
            try:
                # Ensure the data iterator is reset or re-created if it's a one-shot iterator
                # For this tutorial, assuming posthoc_activation_store can be iterated multiple times
                # or we re-initialize it if it's a generator type that gets exhausted.
                data_iterator_for_estimation = iter(posthoc_activation_store)

                estimated_thetas = tmp_model_for_sweep.estimate_theta_posthoc(
                    data_iter=data_iterator_for_estimation,
                    num_batches=n_batches_for_theta_estimation,
                    scale_factor=sf,
                    default_theta_value=1e6,  # Default from convert_to_jumprelu_inplace
                )
                # estimate_theta_posthoc now calls convert_to_jumprelu_inplace internally
                print(f"Estimated theta shape: {estimated_thetas.shape}")
                # Now tmp_model_for_sweep is a JumpReLU model

                # Get a sample batch for quick_l0_checks
                # We need to be careful if data_iterator_for_estimation was exhausted
                # For simplicity, let's try to get one more batch or re-initialize iterator for this check
                sample_batch_for_l0_check_inputs: Dict[int, torch.Tensor] = {}
                try:
                    sample_inputs_l0, _ = next(data_iterator_for_estimation)  # Try to get next from current iterator
                    sample_batch_for_l0_check_inputs = sample_inputs_l0
                except StopIteration:
                    print("Warning: data_iterator_for_estimation exhausted. Re-initializing for L0 check.")
                    try:
                        reinitialized_iterator = iter(posthoc_activation_store)
                        sample_inputs_l0, _ = next(reinitialized_iterator)
                        sample_batch_for_l0_check_inputs = sample_inputs_l0
                    except Exception as e_reinit_fetch:
                        print(f"Error re-fetching batch for L0 check: {e_reinit_fetch}. L0 check might use zeros.")
                except Exception as e_fetch_l0_batch:
                    print(f"Error fetching batch for L0 check: {e_fetch_l0_batch}. L0 check might use zeros.")

                d_l0, exp_l0 = quick_l0_checks(tmp_model_for_sweep, sample_batch_for_l0_check_inputs)
                print(
                    f"scale {sf:4.2f}  | dummy-L0 {d_l0:6.0f} | expected-L0 {exp_l0:7.1f} (num_features={tmp_model_for_sweep.config.num_features}, num_layers={tmp_model_for_sweep.config.num_layers})"
                )
            except Exception as e_sweep_iter:
                print(f"ERROR during sweep iteration for scale_factor={sf:.2f}: {e_sweep_iter}")
                traceback.print_exc()
                continue  # Continue to next scale factor
    else:
        print("Skipping post-hoc theta scaling sweep as activation store could not be initialized.")

# %% [markdown]
# ## 8. Next Steps
#
# This tutorial showed how to train a CLT using the global BatchTopK activation,
# save/load models in various formats, and perform a post-hoc analysis of
# converting a BatchTopK model to JumpReLU with different scaling factors for the threshold.

# %%
print("\nBatchTopK Tutorial Complete!")
print(f"Logs and checkpoints will be saved to: {log_dir}")

# %%
weights = torch.load(batchtopk_checkpoint_path)

# %%
weights.keys()
# %%
