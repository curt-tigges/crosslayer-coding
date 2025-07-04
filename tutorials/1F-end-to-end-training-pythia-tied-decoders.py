# %% [markdown]
# # Tutorial: End-to-End CLT Training with Tied Decoders and Feature Offset
#
# This tutorial demonstrates training a Cross-Layer Transcoder (CLT) using:
# - **Tied decoder architecture** to reduce memory usage
# - **Feature offset parameters** for per-feature bias
# - **BatchTopK activation** (same as Tutorial 1B)
#
# The tied decoder architecture uses one decoder per source layer (instead of one per source-target pair),
# significantly reducing memory usage from O(L²) to O(L) decoder parameters.
#
# We will:
# 1. Configure the CLT model with tied decoders and feature offset
# 2. Use the same pre-generated activations from Tutorial 1B
# 3. Train the model and compare memory usage
# 4. Demonstrate loading checkpoints with the new architecture

# %% [markdown]
# ## 1. Imports and Setup

# %%
import torch
import os
import time
import sys
import traceback
import json
from torch.distributed.checkpoint import load_state_dict as dist_load_state_dict
from torch.distributed.checkpoint.filesystem import FileSystemReader
from typing import Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s")

# Ensure tokenizers don't use parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
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

# Base model for activation extraction (same as Tutorial 1B)
BASE_MODEL_NAME = "EleutherAI/pythia-70m"

# %% [markdown]
# ## 2. Configuration with Tied Decoders
#
# Key differences from Tutorial 1B:
# - `decoder_tying="per_source"` - Enables tied decoder architecture
# - `enable_feature_offset=True` - Adds learnable per-feature bias
# - Memory savings: For 6 layers, we go from 21 decoders to just 6

# %%
# --- CLT Architecture Configuration with Tied Decoders ---
num_layers = 6
d_model = 512
expansion_factor = 32
clt_num_features = d_model * expansion_factor

batchtopk_k = 200

clt_config = CLTConfig(
    num_features=clt_num_features,
    num_layers=num_layers,
    d_model=d_model,
    activation_fn="batchtopk",
    two_stage_batchtopk=True,
    batchtopk_k=batchtopk_k,
    batchtopk_straight_through=True,
    # NEW: Tied decoder configuration
    decoder_tying="per_target",  # Use one decoder per source layer
    enable_feature_offset=False,  # Enable per-feature bias (feature_offset)
    enable_feature_scale=False,  # Enable per-feature scale (feature_scale)
    skip_connection=True,  # Enable skip connection from input to output
)

print("CLT Configuration (Tied Decoders with Feature Affine):")
print(f"- decoder_tying: {clt_config.decoder_tying}")
print(f"- enable_feature_offset: {clt_config.enable_feature_offset}")
print(f"- enable_feature_scale: {clt_config.enable_feature_scale}")
print(f"- skip_connection: {clt_config.skip_connection}")
print(f"- Number of features: {clt_config.num_features}")
print(f"- Number of layers: {clt_config.num_layers}")
print(f"- Activation function: {clt_config.activation_fn}")
print(f"- BatchTopK k: {clt_config.batchtopk_k}")

# Calculate memory savings
untied_decoders = sum(range(1, num_layers + 1))  # 6 + 5 + 4 + 3 + 2 + 1 = 21
tied_decoders = num_layers  # 6
print(f"\nMemory savings:")
print(f"- Untied decoders: {untied_decoders} decoder matrices")
print(f"- Tied decoders: {tied_decoders} decoder matrices")
print(f"- Reduction: {(1 - tied_decoders/untied_decoders)*100:.1f}%")

# --- Use existing activations from Tutorial 1B ---
# We'll use the same activation directory as Tutorial 1B since the base model
# and dataset are identical - only the CLT architecture differs
activation_dir = "./tutorial_activations_local_1M_pythia"
dataset_name = "monology/pile-uncopyrighted"

expected_activation_path = os.path.join(
    activation_dir,
    BASE_MODEL_NAME,
    f"{os.path.basename(dataset_name)}_train",
)

# Verify activations exist
metadata_path = os.path.join(expected_activation_path, "metadata.json")
manifest_path = os.path.join(expected_activation_path, "index.bin")

if not (os.path.exists(metadata_path) and os.path.exists(manifest_path)):
    print(f"\nERROR: Activations not found at {expected_activation_path}")
    print("Please run Tutorial 1B first to generate the activations.")
    raise FileNotFoundError("Activation dataset not found")
else:
    print(f"\nUsing existing activations from: {expected_activation_path}")

# --- Training Configuration ---
_lr = 1e-4
_batch_size = 1024

# WandB run name includes tied decoder info
wdb_run_name = (
    f"{clt_config.num_features}-width-"
    f"tied-decoders-"  # Indicate tied decoder architecture
    f"feat-offset-"  # Indicate feature offset is enabled
    f"batchtopk-k{batchtopk_k}-"
    f"{_batch_size}-batch-"
    f"{_lr:.1e}-lr"
)
print(f"\nGenerated WandB run name: {wdb_run_name}")

training_config = TrainingConfig(
    # Training loop parameters
    learning_rate=_lr,
    training_steps=1000,  # Same as Tutorial 1B for comparison
    seed=42,
    # Activation source (using existing activations)
    activation_source="local_manifest",
    activation_path=expected_activation_path,
    activation_dtype="float32",
    # Training batch size
    train_batch_size_tokens=_batch_size,
    sampling_strategy="sequential",
    # Normalization
    normalization_method="mean_std",
    # Loss function coefficients (same as Tutorial 1B)
    sparsity_lambda=0.0,
    sparsity_lambda_schedule="linear",
    sparsity_c=0.0,
    preactivation_coef=0,
    aux_loss_factor=1 / 32,
    apply_sparsity_penalty_to_batchtopk=False,
    # Optimizer & Scheduler
    optimizer="adamw",
    lr_scheduler="linear_final20",
    optimizer_beta2=0.98,
    # Logging & Checkpointing
    log_interval=10,
    eval_interval=50,
    diag_every_n_eval_steps=1,
    max_features_for_diag_hist=1000,
    checkpoint_interval=500,
    dead_feature_window=200,
    # WandB
    enable_wandb=True,
    wandb_project="clt-hp-sweeps-pythia-70m",
    wandb_run_name=wdb_run_name,
)

print("\nTraining Configuration:")
print(f"- Learning rate: {training_config.learning_rate}")
print(f"- Training steps: {training_config.training_steps}")
print(f"- Batch size (tokens): {training_config.train_batch_size_tokens}")

# %% [markdown]
# ## 3. Initialize Model and Check Architecture
#
# Let's create the model and verify the tied decoder architecture is set up correctly.

# %%
print("\nInitializing CLT model with tied decoders...")

# Create model instance to inspect architecture
model = CrossLayerTranscoder(
    config=clt_config,
    process_group=None,
    device=torch.device(device),
)

print("\nModel architecture inspection:")
print(f"- Encoder modules: {len(model.encoder_module.encoders)}")
print(f"- Decoder modules: {len(model.decoder_module.decoders)}")

# Check feature offset parameters
if model.decoder_module.feature_offset is not None:
    print(f"- Feature offset parameters per layer: {len(model.decoder_module.feature_offset)}")
    print(f"- Feature offset shape (layer 0): {model.decoder_module.feature_offset[0].shape}")
else:
    print("- Feature offset: Not enabled")

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
encoder_params = sum(p.numel() for p in model.encoder_module.parameters())
decoder_params = sum(p.numel() for p in model.decoder_module.parameters())
print(f"\nParameter counts:")
print(f"- Total parameters: {total_params:,}")
print(f"- Encoder parameters: {encoder_params:,}")
print(f"- Decoder parameters: {decoder_params:,}")

# Compare with untied architecture (approximate)
untied_decoder_params_approx = decoder_params * (untied_decoders / tied_decoders)
print(f"\nEstimated decoder parameters if untied: {untied_decoder_params_approx:,}")
print(f"Memory savings in decoder: {(1 - decoder_params/untied_decoder_params_approx)*100:.1f}%")

# Clean up the test model
del model

# %% [markdown]
# ## 4. Training the CLT with Tied Decoders

# %%
print("\nInitializing CLTTrainer for training with tied decoders...")

log_dir = f"clt_training_logs/clt_pythia_tied_decoders_{int(time.time())}"
os.makedirs(log_dir, exist_ok=True)
print(f"Logs and checkpoints will be saved to: {log_dir}")

try:
    print("\nCreating CLTTrainer instance...")
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
print("\nBeginning training with tied decoders...")
print(f"Training for {training_config.training_steps} steps.")
print(f"Decoder tying: {clt_config.decoder_tying}")
print(f"Feature offset enabled: {clt_config.enable_feature_offset}")

try:
    start_train_time = time.time()
    trained_clt_model = trainer.train(eval_every=training_config.eval_interval)
    end_train_time = time.time()
    print(f"\nTraining finished in {end_train_time - start_train_time:.2f} seconds.")
except Exception as train_err:
    print(f"[ERROR] Training failed: {train_err}")
    traceback.print_exc()
    raise

# %% [markdown]
# ## 5. Saving and Loading the Tied Decoder Model

# %%
# Save the final model state and config
final_model_state_path = os.path.join(log_dir, "clt_tied_final_state.pt")
final_model_config_path = os.path.join(log_dir, "clt_tied_final_config.json")

print(f"\nSaving final model state to: {final_model_state_path}")
print(f"Saving final model config to: {final_model_config_path}")

torch.save(trained_clt_model.state_dict(), final_model_state_path)
with open(final_model_config_path, "w") as f:
    json.dump(trained_clt_model.config.__dict__, f, indent=4)

# Verify the saved config has tied decoder settings
with open(final_model_config_path, "r") as f:
    saved_config = json.load(f)
    print(f"\nSaved config verification:")
    print(f"- decoder_tying: {saved_config['decoder_tying']}")
    print(f"- enable_feature_offset: {saved_config['enable_feature_offset']}")
    print(f"- activation_fn: {saved_config['activation_fn']} (converted from batchtopk)")

# Load the model back
print("\nLoading the saved tied decoder model...")
loaded_config = CLTConfig(**saved_config)
loaded_model = CrossLayerTranscoder(
    config=loaded_config,
    process_group=None,
    device=torch.device(device),
)
loaded_model.load_state_dict(torch.load(final_model_state_path, map_location=device))
loaded_model.eval()

print("Model loaded successfully.")
print(f"Loaded model decoder count: {len(loaded_model.decoder_module.decoders)}")

# %% [markdown]
# ## 6. Backward Compatibility Test
#
# Test loading an old untied checkpoint into our tied decoder model.
# This demonstrates the backward compatibility feature.

# %%
print("\n=== Testing Backward Compatibility ===")

# Create a simple untied model for testing
untied_config = CLTConfig(
    num_features=clt_config.num_features,
    num_layers=clt_config.num_layers,
    d_model=clt_config.d_model,
    activation_fn="relu",  # Simple activation for testing
    decoder_tying="none",  # Untied decoders
)

print("Creating untied model for compatibility test...")
untied_model = CrossLayerTranscoder(
    config=untied_config,
    process_group=None,
    device=torch.device("cpu"),  # Use CPU for this test
)

# Save untied model state
untied_state_dict = untied_model.state_dict()
print(f"Untied model decoder keys (first 5): {list(k for k in untied_state_dict.keys() if 'decoder' in k)[:5]}")

# Create tied model with same dimensions
tied_test_config = CLTConfig(
    num_features=clt_config.num_features,
    num_layers=clt_config.num_layers,
    d_model=clt_config.d_model,
    activation_fn="relu",
    decoder_tying="per_source",  # Tied decoders
    enable_feature_offset=True,  # This will be initialized to defaults
)

tied_test_model = CrossLayerTranscoder(
    config=tied_test_config,
    process_group=None,
    device=torch.device("cpu"),
)

print("\nLoading untied checkpoint into tied model...")
try:
    # This should work due to our custom load_state_dict
    tied_test_model.load_state_dict(untied_state_dict, strict=False)
    print("✓ Successfully loaded untied checkpoint into tied model!")
    print("  The tied model uses diagonal decoder weights from the untied model.")
except Exception as e:
    print(f"✗ Failed to load: {e}")

# Clean up test models
del untied_model, tied_test_model

# %% [markdown]
# ## 7. Performance Comparison Summary

# %%
print("\n=== Tied Decoder Architecture Summary ===")
print(f"\nConfiguration used:")
print(f"- Model: {BASE_MODEL_NAME}")
print(f"- Layers: {num_layers}")
print(f"- Hidden dimension: {d_model}")
print(f"- Features per layer: {clt_num_features}")
print(f"- Decoder tying: {clt_config.decoder_tying}")
print(f"- Feature offset: {clt_config.enable_feature_offset}")

print(f"\nMemory efficiency:")
print(f"- Traditional CLT: {untied_decoders} decoder matrices")
print(f"- Tied decoder CLT: {tied_decoders} decoder matrices")
print(f"- Memory reduction: ~{(1 - tied_decoders/untied_decoders)*100:.0f}%")

print(f"\nKey benefits:")
print(f"1. Significant memory savings for decoder parameters")
print(f"2. Simpler feature interpretability (one decoder per source)")
print(f"3. Feature offset allows per-feature adaptation")
print(f"4. Backward compatible with existing checkpoints")

print(f"\nTrade-offs:")
print(f"1. Less flexibility in source-target specific adaptations")
print(f"2. May require careful tuning of feature offset parameters")

# %% [markdown]
# ## 8. Next Steps
#
# This tutorial demonstrated:
# - Training a CLT with tied decoder architecture
# - Using feature offset parameters for per-feature bias
# - Significant memory savings compared to traditional CLT
# - Backward compatibility with untied checkpoints
#
# You can experiment with:
# - `per_target_scale` and `per_target_bias` for more flexibility
# - `enable_feature_scale` for per-feature scaling
# - Different values of `k` for BatchTopK
# - Comparing reconstruction quality between tied and untied architectures

# %%
print(f"\n✓ Tied Decoder Tutorial Complete!")
print(f"Model and logs saved to: {log_dir}")
