import torch
import os
import json
import argparse
import sys
import logging
from typing import Dict, List, Optional
import math

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from clt.config import CLTConfig
    from clt.models.clt import CrossLayerTranscoder
    from clt.training.data.local_activation_store import LocalActivationStore  # Default store for this script
    from clt.training.evaluator import CLTEvaluator  # Added for NMSE check
    from safetensors.torch import load_file, save_file  # Added for safetensors support

    # Add other store types if needed, e.g., RemoteActivationStore
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure the 'clt' library is installed or the clt directory is in your PYTHONPATH.")
    print("If the error is related to 'safetensors', please install it: pip install safetensors")
    raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _remap_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remaps old state_dict keys to the new format with module prefixes.
    
    Handles both old-style keys (encoders.*, decoders.*) and new tied decoder parameters:
    - encoder_module.feature_offset.{layer_idx}: ParameterList for per-feature bias
    - encoder_module.feature_scale.{layer_idx}: ParameterList for per-feature scale  
    - decoder_module.skip_weights.{layer_idx}: ParameterList for tied decoders
    - decoder_module.skip_weights.{src}->{tgt}: ParameterDict for untied decoders
    - decoder_module.per_target_scale: Tensor for per src->tgt scale (tied decoders)
    - decoder_module.per_target_bias: Tensor for per src->tgt bias (tied decoders)
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoders."):
            new_key = "encoder_module." + key
        elif key.startswith("decoders."):
            new_key = "decoder_module." + key
        else:
            new_key = key
        new_state_dict[new_key] = value
        
    # Handle new parameter names that might not have the correct module prefix
    # Create a list of keys to avoid modifying dict during iteration
    keys_to_check = list(new_state_dict.keys())
    for key in keys_to_check:
        # Handle feature_offset/feature_scale that might be saved without module prefix
        # Note: These are now in the decoder module for tied decoders
        if key.startswith("feature_offset.") and not key.startswith("decoder_module.") and not key.startswith("encoder_module."):
            new_state_dict[f"decoder_module.{key}"] = new_state_dict.pop(key)
        elif key.startswith("feature_scale.") and not key.startswith("decoder_module.") and not key.startswith("encoder_module."):
            new_state_dict[f"decoder_module.{key}"] = new_state_dict.pop(key)
        # Handle skip_weights that might be saved without module prefix
        elif key.startswith("skip_weights.") and not key.startswith("decoder_module."):
            new_state_dict[f"decoder_module.{key}"] = new_state_dict.pop(key)
        # Handle per_target parameters
        elif key in ["per_target_scale", "per_target_bias"] and not key.startswith("decoder_module."):
            new_state_dict[f"decoder_module.{key}"] = new_state_dict.pop(key)
    
    if not any(k.startswith("encoder_module.") or k.startswith("decoder_module.") for k in new_state_dict.keys()):
        if any(k.startswith("encoders.") or k.startswith("decoders.") for k in state_dict.keys()):
            logger.warning(
                "Key remapping applied, but no keys were actually changed. "
                "This might indicate the checkpoint is already in the new format or the remapping logic is flawed."
            )
        else:
            # Neither old nor new prefixes found, probably a different kind of checkpoint or already remapped.
            pass  # Don't log a warning if it seems like it's already fine.

    # Check if we missed any expected old keys
    remapped_old_keys = any(k.startswith("encoders.") or k.startswith("decoders.") for k in new_state_dict.keys())
    if remapped_old_keys:
        logger.error("CRITICAL: Key remapping failed to remove all old-style keys. Check remapping logic.")
        # This state is problematic, so we might choose to raise an error or return the original dict.
        # For now, returning the partially remapped dict and letting load_state_dict fail might be more informative.

    return new_state_dict


def print_performance_metrics(stage_name: str, nmse: float, ev: float, l0_total: float = None, l0_per_layer: Dict[int, float] = None):
    """Print performance metrics in a standardized, prominent format."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"PERFORMANCE METRICS - {stage_name.upper()}")
    logger.info("=" * 80)
    logger.info(f"NMSE (Normalized Mean Squared Error): {nmse:.6f}")
    logger.info(f"EV (Explained Variance):              {ev:.6f}")
    if l0_total is not None:
        logger.info(f"L0 (Total across all layers):         {l0_total:.2f}")
    if l0_per_layer is not None:
        logger.info("L0 per layer:")
        for layer_idx in sorted(l0_per_layer.keys()):
            l0_val = l0_per_layer[layer_idx]
            if not math.isnan(l0_val):
                logger.info(f"  Layer {layer_idx}: {l0_val:.2f}")
    logger.info("=" * 80)
    logger.info("")


def main(args):
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load the BatchTopK CLTConfig
    if not os.path.exists(args.config_path):
        logger.error(f"BatchTopK config file not found at: {args.config_path}")
        return
    with open(args.config_path, "r") as f:
        config_dict_batchtopk = json.load(f)

    # Ensure the loaded config is indeed for BatchTopK for loading the checkpoint
    config_activation_fn = config_dict_batchtopk.get("activation_fn")
    if config_activation_fn not in ["batchtopk", "topk"]:
        logger.warning(
            f"Warning: Config at {args.config_path} has activation_fn='{config_activation_fn}'. "
            f"This script is designed for 'batchtopk' or 'topk' models. "
            f"Proceeding as if it's a compatible config for loading the checkpoint, "
            f"but ensure this is intended and the checkpoint matches."
        )
        # Override to ensure BatchTopK for initial loading if needed, though ideally cfg.json matches.
        # config_dict_batchtopk['activation_fn'] = 'batchtopk' # Or 'topk'
        # It is better if the user provides the correct config file that was used to save the checkpoint.

    try:
        clt_config_batchtopk = CLTConfig(**config_dict_batchtopk)
        logger.info(f"Loaded BatchTopK CLTConfig: {clt_config_batchtopk}")
        
        # Print detailed config for debugging
        logger.info("=" * 80)
        logger.info("DETAILED CLT CONFIG:")
        logger.info("=" * 80)
        for key, value in clt_config_batchtopk.__dict__.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error creating CLTConfig from {args.config_path}: {e}")
        return

    # 2. Instantiate BatchTopK model and load state_dict
    logger.info("Instantiating CLT model with BatchTopK configuration...")
    model = CrossLayerTranscoder(
        config=clt_config_batchtopk, process_group=None, device=device  # Assuming non-distributed for conversion script
    )

    if not os.path.exists(args.batchtopk_checkpoint_path):
        logger.error(f"BatchTopK checkpoint file not found at: {args.batchtopk_checkpoint_path}")
        return

    logger.info(f"Loading BatchTopK model state from: {args.batchtopk_checkpoint_path}")
    try:
        # Load checkpoint, handling potential sharded format if it's a directory
        if os.path.isdir(args.batchtopk_checkpoint_path):
            from torch.distributed.checkpoint.state_dict_loader import load_state_dict as dist_load_state_dict
            from torch.distributed.checkpoint.filesystem import FileSystemReader

            logger.info("Checkpoint path is a directory, attempting distributed load into a single model.")
            state_dict_to_populate = model.state_dict()  # Get state_dict from the instantiated model
            dist_load_state_dict(
                state_dict=state_dict_to_populate,
                storage_reader=FileSystemReader(args.batchtopk_checkpoint_path),
                no_dist=True,  # Load sharded into a non-distributed model structure
            )
            model.load_state_dict(state_dict_to_populate)

        else:  # Standard single-file checkpoint
            if args.batchtopk_checkpoint_path.endswith(".safetensors"):
                logger.info(f"Loading BatchTopK model state from safetensors file: {args.batchtopk_checkpoint_path}")
                state_dict = load_file(args.batchtopk_checkpoint_path, device=device.type)
            else:
                logger.info(f"Loading BatchTopK model state from .pt file: {args.batchtopk_checkpoint_path}")
                state_dict = torch.load(args.batchtopk_checkpoint_path, map_location=device)

            state_dict = _remap_checkpoint_keys(state_dict)  # Remap keys

            # --- Add config validation against checkpoint ---
            first_encoder_weight_key = "encoder_module.encoders.0.weight"
            if first_encoder_weight_key in state_dict:
                checkpoint_num_features = state_dict[first_encoder_weight_key].shape[0]
                config_num_features = clt_config_batchtopk.num_features
                if checkpoint_num_features != config_num_features:
                    logger.error("--- CONFIGURATION MISMATCH ---")
                    logger.error(
                        f"The 'num_features' in your config file ({config_num_features}) "
                        f"does not match the number of features in the checkpoint ({checkpoint_num_features})."
                    )
                    logger.error(f"  Config path: {args.config_path}")
                    logger.error(f"  Checkpoint path: {args.batchtopk_checkpoint_path}")
                    logger.error(
                        "Please ensure you are using the correct config file that was used to train this model."
                    )
                    return  # Exit the script

            model.load_state_dict(state_dict)

        model.eval()
        logger.info("BatchTopK model loaded and set to eval mode.")
    except Exception as e:
        logger.error(f"Error loading BatchTopK model state: {e}")
        return

    # 3. Calculate NMSE/EV BEFORE conversion
    logger.info("Calculating NMSE/EV for original BatchTopK model before conversion...")
    
    # Log batch size for BatchTopK models
    if clt_config_batchtopk.activation_fn == "batchtopk":
        logger.info(
            f"BatchTopK model with k={clt_config_batchtopk.batchtopk_k} (per token), "
            f"mode={clt_config_batchtopk.topk_mode}, batch_size={args.estimation_batch_size_tokens}"
        )
    
    try:
        # Initialize activation store for pre-conversion evaluation
        pre_conv_store = LocalActivationStore(
            dataset_path=args.activation_data_path,
            train_batch_size_tokens=args.estimation_batch_size_tokens,
            device=device,
            dtype=args.activation_dtype or clt_config_batchtopk.expected_input_dtype or "float32",
            rank=0,
            world=1,
            seed=args.seed + 100,  # Different seed for pre-conversion
            sampling_strategy="sequential",
            normalization_method=clt_config_batchtopk.normalization_method,
        )
        
        # Collect data for evaluation
        pre_conv_inputs = {}
        pre_conv_targets = {}
        data_iter = iter(pre_conv_store)
        
        # Use limited batches for pre-conversion check to save memory
        num_pre_conv_batches = min(args.num_batches_for_l0_check, 5)
        
        for batch_idx in range(num_pre_conv_batches):  # Use limited batches for pre-conversion check
            try:
                inputs_batch, targets_batch = next(data_iter)
                for layer_idx, inp in inputs_batch.items():
                    if layer_idx not in pre_conv_inputs:
                        pre_conv_inputs[layer_idx] = []
                        pre_conv_targets[layer_idx] = []
                    if inp.numel() > 0:
                        target = targets_batch[layer_idx]
                        # Flatten if needed
                        if inp.dim() == 3:
                            inp = inp.reshape(-1, inp.shape[-1])
                            target = target.reshape(-1, target.shape[-1])
                        pre_conv_inputs[layer_idx].append(inp.cpu())
                        pre_conv_targets[layer_idx].append(target.cpu())
            except StopIteration:
                break
        
        # Concatenate and move to device
        for layer_idx in pre_conv_inputs:
            if pre_conv_inputs[layer_idx]:
                pre_conv_inputs[layer_idx] = torch.cat(pre_conv_inputs[layer_idx], dim=0).to(device)
                pre_conv_targets[layer_idx] = torch.cat(pre_conv_targets[layer_idx], dim=0).to(device)
        
        # Get reconstructions from original model
        with torch.no_grad():
            pre_conv_reconstructions = model(pre_conv_inputs)
        
        # Calculate metrics
        mean_tg = pre_conv_store.mean_tg if hasattr(pre_conv_store, 'mean_tg') else None
        std_tg = pre_conv_store.std_tg if hasattr(pre_conv_store, 'std_tg') else None
        
        evaluator = CLTEvaluator(
            model=model, 
            device=device, 
            mean_tg=mean_tg, 
            std_tg=std_tg,
            normalization_method=clt_config_batchtopk.normalization_method,
            d_model=clt_config_batchtopk.d_model
        )
        reconstruction_metrics = evaluator._compute_reconstruction_metrics(
            targets=pre_conv_targets, 
            reconstructions=pre_conv_reconstructions
        )
        
        pre_conv_nmse = reconstruction_metrics.get("reconstruction/normalized_mean_reconstruction_error", float("nan"))
        pre_conv_ev = reconstruction_metrics.get("reconstruction/explained_variance", float("nan"))
        
        # Calculate L0
        pre_conv_l0s = run_quick_l0_checks_script(model, pre_conv_inputs, args.num_tokens_for_l0_check_script)
        pre_conv_l0_total = sum(l0 for l0 in pre_conv_l0s.values() if not math.isnan(l0))
        
        # Report pre-conversion metrics
        print_performance_metrics("BEFORE CONVERSION (BatchTopK)", pre_conv_nmse, pre_conv_ev, pre_conv_l0_total, pre_conv_l0s)
        
        if hasattr(pre_conv_store, "close"):
            pre_conv_store.close()
            
    except Exception as e:
        logger.warning(f"Failed to calculate pre-conversion metrics: {e}")
    
    # 4. Initialize ActivationStore for theta estimation
    logger.info(f"Initializing LocalActivationStore from: {args.activation_data_path} for theta estimation")
    if not os.path.exists(args.activation_data_path):
        logger.error(f"Activation data path not found: {args.activation_data_path}")
        return

    try:
        # For simplicity, this script defaults to LocalActivationStore.
        # Users can modify this to use other store types if needed.
        activation_store_theta = LocalActivationStore(
            dataset_path=args.activation_data_path,
            train_batch_size_tokens=args.estimation_batch_size_tokens,  # Use a batch size for estimation
            device=device,
            dtype=args.activation_dtype or clt_config_batchtopk.expected_input_dtype or "float32",
            rank=0,
            world=1,
            seed=args.seed,
            sampling_strategy="sequential",  # Sequential is fine for estimation
            normalization_method=clt_config_batchtopk.normalization_method,  # Use the same normalization as training
        )
        logger.info("Activation store for theta estimation initialized.")
    except Exception as e:
        logger.error(f"Error initializing activation store for theta estimation: {e}")
        return

    # 4. Estimate Theta and Convert Model
    logger.info(
        f"Starting theta estimation using {args.num_batches_for_theta_estimation} batches "
        f"with default_theta_value={args.default_theta_value}."
    )
    # Store original K before it gets wiped by model conversion
    # original_batchtopk_k = clt_config_batchtopk.batchtopk_k # Will be determined later in scale search section

    try:
        estimated_thetas = model.estimate_theta_posthoc(
            data_iter=iter(activation_store_theta),
            num_batches=args.num_batches_for_theta_estimation,
            default_theta_value=args.default_theta_value,
            device=device,  # Pass device to ensure buffers are on correct device
        )
        logger.info(
            f"Theta estimation and conversion to JumpReLU complete. Estimated theta tensor shape: {estimated_thetas.shape}"
        )
        logger.info(f"Model config after conversion: {model.config}")
    except Exception as e:
        logger.error(f"Error during estimate_theta_posthoc: {e}")
        return
    finally:
        if hasattr(activation_store_theta, "close") and callable(getattr(activation_store_theta, "close")):
            activation_store_theta.close()

    # 5. Save the Converted JumpReLU Model and its Config
    logger.info(f"Saving converted JumpReLU model state to: {args.output_model_path}")
    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    if args.output_model_path.endswith(".safetensors"):
        save_file(model.state_dict(), args.output_model_path)
        logger.info(f"Saved converted JumpReLU model state as safetensors to: {args.output_model_path}")
    else:
        torch.save(model.state_dict(), args.output_model_path)
        logger.info(f"Saved converted JumpReLU model state as .pt to: {args.output_model_path}")

    logger.info(f"Saving converted JumpReLU model config to: {args.output_config_path}")
    os.makedirs(os.path.dirname(args.output_config_path), exist_ok=True)
    with open(args.output_config_path, "w") as f:
        # model.config is updated inplace by convert_to_jumprelu_inplace (called by estimate_theta_posthoc)
        json.dump(model.config.__dict__, f, indent=4)

    # 6. Perform a quick L0 and NMSE check on the converted model
    logger.info("Performing L0 and NMSE check on the converted JumpReLU model...")

    l0_check_fetch_batch_size = (
        args.l0_check_batch_size_tokens
        if hasattr(args, "l0_check_batch_size_tokens") and args.l0_check_batch_size_tokens is not None
        else args.estimation_batch_size_tokens
    )

    logger.info(
        f"Collecting data for L0/NMSE check using {args.num_batches_for_l0_check} batches with fetch batch size {l0_check_fetch_batch_size} tokens."
    )

    # --- Initialise accumulators for batch-wise processing to conserve memory ---
    total_l0_per_layer: Dict[int, float] = {i: 0.0 for i in range(model.config.num_layers)}
    total_tokens_for_l0_per_layer: Dict[int, int] = {i: 0 for i in range(model.config.num_layers)}
    all_reconstructions_for_nmse: Dict[int, List[torch.Tensor]] = {i: [] for i in range(model.config.num_layers)}
    all_targets_for_nmse_check: Dict[int, List[torch.Tensor]] = {i: [] for i in range(model.config.num_layers)}
    mean_tg_for_eval = None
    std_tg_for_eval = None

    try:
        activation_store_for_l0_check = LocalActivationStore(
            dataset_path=args.activation_data_path,
            train_batch_size_tokens=l0_check_fetch_batch_size,
            device=device,
            dtype=args.activation_dtype or clt_config_batchtopk.expected_input_dtype or "float32",
            rank=0,
            world=1,
            seed=args.seed + 1,
            sampling_strategy="sequential",
            normalization_method=clt_config_batchtopk.normalization_method,  # Use the same normalization as training
        )
        data_iterator_for_l0_check = iter(activation_store_for_l0_check)

        if hasattr(activation_store_for_l0_check, "mean_tg") and hasattr(activation_store_for_l0_check, "std_tg"):
            mean_tg_for_eval = activation_store_for_l0_check.mean_tg
            std_tg_for_eval = activation_store_for_l0_check.std_tg
            if mean_tg_for_eval and std_tg_for_eval:
                logger.info("Retrieved mean_tg and std_tg from L0 check activation store for NMSE de-normalization.")
            else:
                logger.info(
                    "mean_tg or std_tg not available from L0 check store. NMSE will be on potentially normalized values."
                )

        # --- Process data batch by batch ---
        for batch_idx in range(args.num_batches_for_l0_check):
            try:
                sample_inputs_batch, sample_targets_batch = next(data_iterator_for_l0_check)
                # Flatten inputs and targets from (B, S, D) to (B*S, D)
                flat_inputs_batch = {
                    layer_idx: (t.reshape(-1, t.shape[-1]) if t.dim() == 3 else t)
                    for layer_idx, t in sample_inputs_batch.items()
                }
                flat_targets_batch = {
                    layer_idx: (t.reshape(-1, t.shape[-1]) if t.dim() == 3 else t)
                    for layer_idx, t in sample_targets_batch.items()
                }
                if not flat_inputs_batch:
                    continue

                # 1. Compute and accumulate L0 statistics
                with torch.no_grad():
                    l0s_batch = run_quick_l0_checks_script(
                        model, flat_inputs_batch, args.num_tokens_for_l0_check_script
                    )
                for layer_idx, l0_val in l0s_batch.items():
                    if layer_idx in flat_inputs_batch and not math.isnan(l0_val):
                        tokens_in_batch = flat_inputs_batch[layer_idx].shape[0]
                        total_l0_per_layer[layer_idx] += l0_val * tokens_in_batch
                        total_tokens_for_l0_per_layer[layer_idx] += tokens_in_batch

                # 2. Get reconstructions for NMSE, moving to CPU to free up VRAM
                with torch.no_grad():
                    reconstructions_batch = model(flat_inputs_batch)
                for layer_idx, recon_tensor in reconstructions_batch.items():
                    if layer_idx in flat_targets_batch and recon_tensor.numel() > 0:
                        all_reconstructions_for_nmse[layer_idx].append(recon_tensor.cpu())
                        all_targets_for_nmse_check[layer_idx].append(flat_targets_batch[layer_idx].cpu())

            except StopIteration:
                logger.warning(
                    f"Activation store exhausted after {batch_idx + 1} batches. Proceeding with collected data."
                )
                break

        if hasattr(activation_store_for_l0_check, "close"):
            activation_store_for_l0_check.close()

    except Exception as e_l0_fetch:
        logger.error(f"Error during L0/NMSE check data processing: {e_l0_fetch}. Check may be incomplete.")
        # Allow to proceed with any data collected so far

    # --- Finalize and Log Metrics ---
    total_empirical_l0 = 0.0
    empirical_l0s_per_layer = {}
    for layer_idx in range(model.config.num_layers):
        if total_tokens_for_l0_per_layer.get(layer_idx, 0) > 0:
            avg_l0 = total_l0_per_layer[layer_idx] / total_tokens_for_l0_per_layer[layer_idx]
            empirical_l0s_per_layer[layer_idx] = avg_l0
            if not math.isnan(avg_l0):
                total_empirical_l0 += avg_l0
        else:
            empirical_l0s_per_layer[layer_idx] = float("nan")

    # Compute and log NMSE
    logger.info("Computing NMSE on collected data...")
    final_reconstructions = {
        layer_idx: torch.cat(tensors).to(device)
        for layer_idx, tensors in all_reconstructions_for_nmse.items()
        if tensors
    }
    final_targets = {
        layer_idx: torch.cat(tensors).to(device) for layer_idx, tensors in all_targets_for_nmse_check.items() if tensors
    }

    if not final_reconstructions or not final_targets:
        logger.warning(
            "Insufficient data for NMSE calculation after processing (empty reconstructions or targets). Skipping NMSE."
        )
    else:
        evaluator = CLTEvaluator(
            model=model, 
            device=device, 
            mean_tg=mean_tg_for_eval, 
            std_tg=std_tg_for_eval,
            normalization_method=clt_config_batchtopk.normalization_method,
            d_model=clt_config_batchtopk.d_model
        )
        # Filter to common layers with data
        valid_layers_for_nmse = set(final_reconstructions.keys()) & set(final_targets.keys())
        reconstructions_for_metric = {k: v for k, v in final_reconstructions.items() if k in valid_layers_for_nmse}
        targets_for_metric = {k: v for k, v in final_targets.items() if k in valid_layers_for_nmse}

        if not reconstructions_for_metric or not targets_for_metric:
            logger.warning("No common layers with data between reconstructions and targets. Skipping NMSE.")
        else:
            reconstruction_metrics = evaluator._compute_reconstruction_metrics(
                targets=targets_for_metric, reconstructions=reconstructions_for_metric
            )
            nmse_value = reconstruction_metrics.get("reconstruction/normalized_mean_reconstruction_error", float("nan"))
            explained_variance = reconstruction_metrics.get("reconstruction/explained_variance", float("nan"))
            
            # Report post-conversion metrics using standardized format
            print_performance_metrics("AFTER CONVERSION (JumpReLU)", nmse_value, explained_variance, total_empirical_l0, empirical_l0s_per_layer)

    # --- Optional Layer-wise L0 Calibration --- #
    if args.l0_layerwise_calibrate:
        logger.info("--- Starting Layer-wise L0 Calibration Step ---")

        # 1. Determine paths for the original model
        original_config_path = args.l0_target_model_config_path or args.config_path
        original_checkpoint_path = args.l0_target_model_checkpoint_path or args.batchtopk_checkpoint_path

        logger.info(
            f"Loading original model for L0 targets from config: {original_config_path} and checkpoint: {original_checkpoint_path}"
        )
        if not os.path.exists(original_config_path):
            logger.error(f"Original model config file not found at: {original_config_path}. Skipping L0 calibration.")
            return  # Or skip calibration and continue
        with open(original_config_path, "r") as f:
            original_config_dict = json.load(f)

        try:
            original_clt_config = CLTConfig(**original_config_dict)
            original_model = CrossLayerTranscoder(config=original_clt_config, process_group=None, device=device)

            if not os.path.exists(original_checkpoint_path):
                logger.error(
                    f"Original model checkpoint file not found at: {original_checkpoint_path}. Skipping L0 calibration."
                )
                return  # Or skip calibration

            if os.path.isdir(original_checkpoint_path):
                from torch.distributed.checkpoint.state_dict_loader import load_state_dict as dist_load_state_dict
                from torch.distributed.checkpoint.filesystem import FileSystemReader

                state_dict_to_populate_orig = original_model.state_dict()
                dist_load_state_dict(
                    state_dict=state_dict_to_populate_orig,
                    storage_reader=FileSystemReader(original_checkpoint_path),
                    no_dist=True,
                )
                state_dict_to_populate_orig = _remap_checkpoint_keys(state_dict_to_populate_orig)
                original_model.load_state_dict(state_dict_to_populate_orig)
            elif original_checkpoint_path.endswith(".safetensors"):
                state_dict_orig = load_file(original_checkpoint_path, device=device.type)
                state_dict_orig = _remap_checkpoint_keys(state_dict_orig)
                original_model.load_state_dict(state_dict_orig)
            else:
                state_dict_orig = torch.load(original_checkpoint_path, map_location=device)
                state_dict_orig = _remap_checkpoint_keys(state_dict_orig)
                original_model.load_state_dict(state_dict_orig)

            original_model.eval()
            logger.info("Original model loaded successfully for L0 target calculation.")
        except Exception as e:
            logger.error(f"Error loading original model for L0 calibration: {e}. Skipping calibration.")
            return  # Or skip calibration

        # 2. Prepare calibration data (using a small subset of the activation data)
        calib_batch_size = (
            args.l0_calibration_batch_size_tokens
            or args.l0_check_batch_size_tokens
            or args.estimation_batch_size_tokens
        )
        logger.info(
            f"Preparing {args.l0_calibration_batches} batch(es) for L0 calibration with batch size {calib_batch_size} tokens."
        )

        calibration_inputs_collected: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(original_model.config.num_layers)
        }
        try:
            activation_store_for_calib = LocalActivationStore(
                dataset_path=args.activation_data_path,  # Use the same main data path
                train_batch_size_tokens=calib_batch_size,
                device=device,
                dtype=args.activation_dtype or original_clt_config.expected_input_dtype or "float32",
                rank=0,
                world=1,
                seed=args.seed + 2,  # Use a different seed
                sampling_strategy="sequential",
                normalization_method=original_clt_config.normalization_method,  # Use the same normalization as the original model
            )
            data_iterator_for_calib = iter(activation_store_for_calib)
            for _ in range(args.l0_calibration_batches):
                cal_inputs_b, _ = next(data_iterator_for_calib)
                for layer_idx_cal, tensor_data_cal in cal_inputs_b.items():
                    if layer_idx_cal in calibration_inputs_collected:
                        if tensor_data_cal.dim() == 3:
                            num_tokens_cal = tensor_data_cal.shape[0] * tensor_data_cal.shape[1]
                            calibration_inputs_collected[layer_idx_cal].append(
                                tensor_data_cal.reshape(num_tokens_cal, original_model.config.d_model)
                            )
                        elif tensor_data_cal.dim() == 2:
                            calibration_inputs_collected[layer_idx_cal].append(tensor_data_cal)

            if hasattr(activation_store_for_calib, "close") and callable(getattr(activation_store_for_calib, "close")):
                activation_store_for_calib.close()

            final_calibration_inputs: Dict[int, torch.Tensor] = {}
            for layer_idx_cal, tensor_list_cal in calibration_inputs_collected.items():
                if tensor_list_cal:
                    concatenated = torch.cat(tensor_list_cal, dim=0)
                    # --- NEW: Down-sample to avoid huge tensors that crash MPS ---
                    num_tokens_available = concatenated.shape[0]
                    max_tokens = (
                        args.num_tokens_for_l0_check_script
                        if args.num_tokens_for_l0_check_script > 0
                        else num_tokens_available
                    )
                    if num_tokens_available > max_tokens:
                        sel_indices = torch.randperm(num_tokens_available, device=concatenated.device)[:max_tokens]
                        concatenated = concatenated[sel_indices]
                    final_calibration_inputs[layer_idx_cal] = concatenated
                else:
                    logger.warning(f"Layer {layer_idx_cal}: No calibration input tokens collected.")
                    final_calibration_inputs[layer_idx_cal] = torch.empty(
                        (0, original_model.config.d_model), device=device, dtype=original_model.dtype
                    )

            if not any(v.numel() > 0 for v in final_calibration_inputs.values()):
                logger.error("No data collected for L0 calibration. Skipping calibration step.")
                return  # Or skip calibration

        except Exception as e_cal_fetch:
            logger.error(f"Error fetching data for L0 calibration: {e_cal_fetch}. Skipping calibration.")
            return  # Or skip calibration

        # 3. Get Target L0s from the original model
        logger.info("Calculating target L0s from the original model...")
        target_l0s = run_quick_l0_checks_script(
            original_model,
            final_calibration_inputs,
            args.num_tokens_for_l0_check_script,  # Use same num_tokens for consistency in how L0 is measured
        )
        logger.info(f"Target L0s per layer from original model: {target_l0s}")

        # 4. Calibrate the converted JumpReLU model (model_for_l0_check is the one already converted)
        calibrate_layerwise_theta_for_l0_matching(
            model_to_calibrate=model,
            calibration_inputs=final_calibration_inputs,
            target_l0s_per_layer=target_l0s,
            num_tokens_for_l0_check=args.num_tokens_for_l0_check_script,
            min_scale=args.l0_calibration_search_min_scale,
            max_scale=args.l0_calibration_search_max_scale,
            tolerance=args.l0_calibration_tolerance,
            max_iters=args.l0_calibration_max_iters,
            device=device,
        )

        # 5. Log final L0s of the calibrated model
        calibrated_l0s_per_layer = run_quick_l0_checks_script(
            model, final_calibration_inputs, args.num_tokens_for_l0_check_script
        )
        total_calibrated_l0 = sum(l0 for l0 in calibrated_l0s_per_layer.values() if not math.isnan(l0))

        # 6. Re-save the calibrated model
        logger.info(f"Re-saving calibrated JumpReLU model state to: {args.output_model_path}")
        if args.output_model_path.endswith(".safetensors"):
            save_file(model.state_dict(), args.output_model_path)
            logger.info(f"Re-saved calibrated JumpReLU model state as safetensors to: {args.output_model_path}")
        else:
            torch.save(model.state_dict(), args.output_model_path)
            logger.info(f"Re-saved calibrated JumpReLU model state as .pt to: {args.output_model_path}")
        # Config remains the same (JumpReLU), only log_thresholds changed.
        logger.info("--- Layer-wise L0 Calibration Step Finished ---")

        # --- Re-evaluate NMSE/EV after L0 calibration ---
        logger.info("--- NMSE/EV after Layer-wise Calibration ---")
        logger.info("Re-fetching data for post-calibration NMSE/EV evaluation...")

        post_calib_recons: Dict[int, List[torch.Tensor]] = {i: [] for i in range(model.config.num_layers)}
        post_calib_targets: Dict[int, List[torch.Tensor]] = {i: [] for i in range(model.config.num_layers)}

        try:
            store_for_post_calib_eval = LocalActivationStore(
                dataset_path=args.activation_data_path,
                train_batch_size_tokens=l0_check_fetch_batch_size,
                device=device,
                dtype=args.activation_dtype or clt_config_batchtopk.expected_input_dtype or "float32",
                rank=0,
                world=1,
                seed=args.seed + 1,  # Use same seed as pre-calibration check
                sampling_strategy="sequential",
                normalization_method=clt_config_batchtopk.normalization_method,  # Use the same normalization as training
            )
            iterator_post_calib = iter(store_for_post_calib_eval)

            for _ in range(args.num_batches_for_l0_check):
                try:
                    inputs, targets = next(iterator_post_calib)
                    flat_inputs = {
                        layer_idx: (t.reshape(-1, t.shape[-1]) if t.dim() == 3 else t)
                        for layer_idx, t in inputs.items()
                    }
                    flat_targets = {
                        layer_idx: (t.reshape(-1, t.shape[-1]) if t.dim() == 3 else t)
                        for layer_idx, t in targets.items()
                    }

                    if not flat_inputs:
                        continue

                    with torch.no_grad():
                        recons = model(flat_inputs)  # Use calibrated model

                    for layer_idx, recon_tensor in recons.items():
                        if layer_idx in flat_targets and recon_tensor.numel() > 0:
                            post_calib_recons[layer_idx].append(recon_tensor.cpu())
                            post_calib_targets[layer_idx].append(flat_targets[layer_idx].cpu())
                except StopIteration:
                    logger.info("Store exhausted during post-calibration evaluation.")
                    break

            if hasattr(store_for_post_calib_eval, "close"):
                store_for_post_calib_eval.close()
        except Exception as e_post_calib:
            logger.error(f"Error during post-calibration NMSE data processing: {e_post_calib}")

        final_post_calib_recons = {
            layer_idx: torch.cat(tensors).to(device) for layer_idx, tensors in post_calib_recons.items() if tensors
        }
        final_post_calib_targets = {
            layer_idx: torch.cat(tensors).to(device) for layer_idx, tensors in post_calib_targets.items() if tensors
        }

        if not final_post_calib_recons or not final_post_calib_targets:
            logger.warning("Insufficient data for post-calibration NMSE calculation. Skipping.")
        else:
            evaluator_after_calib = CLTEvaluator(
                model=model, 
                device=device, 
                mean_tg=mean_tg_for_eval, 
                std_tg=std_tg_for_eval,
                normalization_method=clt_config_batchtopk.normalization_method,
                d_model=clt_config_batchtopk.d_model
            )
            valid_layers = set(final_post_calib_recons.keys()) & set(final_post_calib_targets.keys())
            recons_metric = {k: v for k, v in final_post_calib_recons.items() if k in valid_layers}
            targets_metric = {k: v for k, v in final_post_calib_targets.items() if k in valid_layers}

            if recons_metric and targets_metric:
                metrics_after_calib = evaluator_after_calib._compute_reconstruction_metrics(
                    targets=targets_metric, reconstructions=recons_metric
                )
                nmse_after_calib = metrics_after_calib.get(
                    "reconstruction/normalized_mean_reconstruction_error", float("nan")
                )
                ev_after_calib = metrics_after_calib.get("reconstruction/explained_variance", float("nan"))
                
                # Report post-L0-calibration metrics using standardized format
                print_performance_metrics("AFTER L0 CALIBRATION (JumpReLU)", nmse_after_calib, ev_after_calib, total_calibrated_l0, calibrated_l0s_per_layer)
            else:
                logger.warning("No common layers with data for post-calibration NMSE calculation. Skipping.")

    logger.info("Conversion script finished successfully.")


def calibrate_layerwise_theta_for_l0_matching(
    model_to_calibrate: CrossLayerTranscoder,
    calibration_inputs: Dict[int, torch.Tensor],
    target_l0s_per_layer: Dict[int, float],
    num_tokens_for_l0_check: int,
    min_scale: float,
    max_scale: float,
    tolerance: float,
    max_iters: int,
    device: torch.device,
) -> None:
    """Calibrates model.log_threshold layer by layer to match target L0s."""
    logger.info("Starting layer-wise L0 calibration...")
    model_to_calibrate.eval()  # Ensure model is in eval mode

    if model_to_calibrate.config.activation_fn != "jumprelu":
        logger.error(
            f"Model activation function is '{model_to_calibrate.config.activation_fn}', not 'jumprelu'. "
            "Cannot perform theta calibration. Exiting calibration."
        )
        return
    if model_to_calibrate.log_threshold is None:
        logger.error("model_to_calibrate.log_threshold is None. Cannot perform calibration. Exiting.")
        return

    # Detach original log_thresholds to use as base for scaling, to avoid them changing with each layer's calibration
    original_log_thetas_exp = model_to_calibrate.log_threshold.exp().detach().clone()

    for layer_idx in range(model_to_calibrate.config.num_layers):
        if layer_idx not in target_l0s_per_layer or math.isnan(target_l0s_per_layer[layer_idx]):
            logger.warning(f"Layer {layer_idx}: No valid target L0. Skipping calibration for this layer.")
            continue

        target_l0 = target_l0s_per_layer[layer_idx]
        if target_l0 < 0:  # Should not happen if target_l0s come from run_quick_l0_checks_script
            logger.warning(f"Layer {layer_idx}: Target L0 ({target_l0:.2f}) is negative. Skipping calibration.")
            continue

        logger.info(f"Layer {layer_idx}: Calibrating to target L0 = {target_l0:.2f}")

        # Get the base theta for this layer (vector of per-feature thetas)
        base_theta_layer = original_log_thetas_exp[layer_idx].to(device)

        low_s = min_scale
        high_s = max_scale
        current_best_scale = 1.0  # Start with no scale change
        current_best_diff = float("inf")

        for iter_num in range(max_iters):
            mid_s = (low_s + high_s) / 2.0
            if mid_s == low_s or mid_s == high_s:  # Avoid getting stuck
                logger.debug(
                    f"Layer {layer_idx} Iter {iter_num + 1}: Scale search converged or stuck at mid_s={mid_s:.3f}. Breaking."
                )
                break

            # Apply current scale to the base theta for this layer
            # The log_threshold parameter is a tensor of shape [num_layers, num_features]
            # We are calibrating one layer at a time. When checking layer_idx,
            # we modify only model.log_threshold.data[layer_idx].
            current_scaled_theta_layer = base_theta_layer * mid_s
            if model_to_calibrate.log_threshold is not None:
                model_to_calibrate.log_threshold.data[layer_idx] = torch.log(current_scaled_theta_layer.clamp_min(1e-9))

            # Check L0 with this new theta for the current layer
            # run_quick_l0_checks_script returns a dict {layer_idx: l0_val}
            # We only care about the L0 of the current layer_idx
            empirical_l0s_current_iter = run_quick_l0_checks_script(
                model_to_calibrate, calibration_inputs, num_tokens_for_l0_check
            )
            empirical_l0_this_layer = empirical_l0s_current_iter.get(layer_idx, float("nan"))

            if math.isnan(empirical_l0_this_layer):
                logger.warning(
                    f"Layer {layer_idx} Iter {iter_num + 1}: Empirical L0 is NaN with scale {mid_s:.3f}. Trying to increase scale (reduce L0)."
                )
                # If L0 is NaN (e.g. no inputs for layer), it often means too many things fired leading to instability or empty selections somewhere.
                # Pushing scale higher (reducing L0) might help recover.
                low_s = mid_s
                continue

            diff = empirical_l0_this_layer - target_l0  # Signed difference

            if abs(diff) < abs(current_best_diff):
                current_best_diff = diff
                current_best_scale = mid_s
            elif abs(diff) == abs(current_best_diff) and mid_s < current_best_scale:  # Prefer smaller scale on ties
                current_best_scale = mid_s

            logger.debug(
                f"Layer {layer_idx} Iter {iter_num + 1}: Scale={mid_s:.3f}, EmpL0={empirical_l0_this_layer:.2f}, TargetL0={target_l0:.2f}, Diff={diff:.2f}"
            )

            if abs(diff) <= tolerance:
                logger.info(
                    f"Layer {layer_idx}: Converged at scale {mid_s:.3f} (EmpL0={empirical_l0_this_layer:.2f}, Diff={diff:.2f}) within tolerance {tolerance}."
                )
                break

            if diff > 0:  # Empirical L0 is too high, need to increase theta (increase scale)
                low_s = mid_s
            else:  # Empirical L0 is too low, need to decrease theta (decrease scale)
                high_s = mid_s
        else:  # Loop finished without break (max_iters reached)
            logger.info(
                f"Layer {layer_idx}: Max iterations ({max_iters}) reached. Best scale {current_best_scale:.3f} (EmpL0 diff {current_best_diff:.2f})."
            )

        # Set the layer's log_threshold to the one corresponding to the best scale found
        final_scaled_theta_layer = base_theta_layer * current_best_scale
        if model_to_calibrate.log_threshold is not None:
            model_to_calibrate.log_threshold.data[layer_idx] = torch.log(final_scaled_theta_layer.clamp_min(1e-9))
        logger.info(f"Layer {layer_idx}: Set final scale {current_best_scale:.3f}. Log_threshold updated.")

    logger.info("Layer-wise L0 calibration finished.")


def run_quick_l0_checks_script(
    model: CrossLayerTranscoder, sample_batch_inputs: Dict[int, torch.Tensor], num_tokens_to_check: Optional[int] = None
) -> Dict[int, float]:
    """Helper function for L0 checks within the script.
    Returns a dictionary of empirical L0 per layer.
    
    Args:
        model: The CrossLayerTranscoder model
        sample_batch_inputs: Dict of layer inputs
        num_tokens_to_check: If provided, sample this many tokens for L0 calculation.
                           If None, use all tokens in the batch.
    """
    model.eval()  # Ensure model is in eval mode
    empirical_l0s_per_layer: Dict[int, float] = {}

    # If the model is batchtopk or topk, we need to get activations via get_feature_activations
    # to correctly account for global K selection before calculating L0.
    # For JumpReLU (during calibration), model.encode() per layer is fine.
    if model.config.activation_fn in ["batchtopk", "topk"]:
        # Ensure inputs are on the correct device for the model
        inputs_on_device = {
            k: v.to(device=model.device, dtype=model.dtype) for k, v in sample_batch_inputs.items() if v.numel() > 0
        }
        if not inputs_on_device:
            logger.warning(
                f"run_quick_l0_checks_script (for {model.config.activation_fn}): No valid input tensors after device transfer. Returning empty L0s."
            )
            for layer_idx in range(model.config.num_layers):
                empirical_l0s_per_layer[layer_idx] = float("nan")
            return empirical_l0s_per_layer

        # We no longer attempt to infer the original (B, S) dimensions, as it provided little
        # benefit and introduced several edge-cases.  L0 will be computed directly on the flat
        # `[total_tokens, num_features]` representation returned by `get_feature_activations`.

        feature_activations = model.get_feature_activations(inputs_on_device)

        for layer_idx in range(model.config.num_layers):
            if layer_idx in feature_activations and feature_activations[layer_idx].numel() > 0:
                acts_layer = feature_activations[layer_idx]  # Shape: [total_tokens_processed, num_features]
                total_tokens_processed = acts_layer.shape[0]

                # --- Simplified L0 computation --- #
                # We directly compute the number of active features (>1e-6) **per token** and then
                # take the average over all tokens.  This avoids the brittle logic of trying to
                # reconstruct the original (B, S) dimensions, which often led to an under-estimate
                # of the true sparsity.

                if total_tokens_processed == 0:
                    avg_empirical_l0_this_layer = float("nan")
                else:
                    # Apply consistent sampling logic with JumpReLU path
                    if num_tokens_to_check is not None and num_tokens_to_check < total_tokens_processed:
                        # Sample a subset of tokens for consistency
                        indices = torch.randperm(total_tokens_processed, device=acts_layer.device)[:num_tokens_to_check]
                        acts_layer_sampled = acts_layer[indices]
                        logger.debug(f"Layer {layer_idx}: Sampling {num_tokens_to_check} tokens out of {total_tokens_processed} for L0 calculation")
                    else:
                        # Use all tokens
                        acts_layer_sampled = acts_layer
                    
                    # Count **non-zero** activations regardless of sign – BatchTopK can select negative
                    # pre-activations.  We therefore look at the absolute value to decide if a feature
                    # is active.
                    l0_per_token = (acts_layer_sampled.abs() > 1e-8).sum(dim=1).float()
                    avg_empirical_l0_this_layer = l0_per_token.mean().item()

                empirical_l0s_per_layer[layer_idx] = avg_empirical_l0_this_layer
            else:
                logger.warning(
                    f"run_quick_l0_checks_script (for {model.config.activation_fn}): No activations found for layer {layer_idx}. L0 will be NaN."
                )
                empirical_l0s_per_layer[layer_idx] = float("nan")
    else:  # For JumpReLU or other per-layer activation models (like during calibration)
        for layer_idx in range(model.config.num_layers):
            avg_empirical_l0_this_layer = float("nan")
            if (
                not sample_batch_inputs
                or layer_idx not in sample_batch_inputs
                or sample_batch_inputs[layer_idx].numel() == 0
            ):
                logger.warning(
                    f"run_quick_l0_checks_script (for {model.config.activation_fn}) received empty/invalid sample_batch_inputs for layer {layer_idx}. Empirical L0 for this layer will be NaN."
                )
            else:
                layer_inputs_all_tokens = sample_batch_inputs[layer_idx].to(device=model.device, dtype=model.dtype)

                if layer_inputs_all_tokens.dim() == 3:  # B, S, D
                    num_tokens_in_batch = layer_inputs_all_tokens.shape[0] * layer_inputs_all_tokens.shape[1]
                    layer_inputs_flat = layer_inputs_all_tokens.reshape(num_tokens_in_batch, model.config.d_model)
                elif layer_inputs_all_tokens.dim() == 2:  # Already [num_tokens, d_model]
                    num_tokens_in_batch = layer_inputs_all_tokens.shape[0]
                    layer_inputs_flat = layer_inputs_all_tokens
                else:
                    logger.warning(
                        f"run_quick_l0_checks_script (for {model.config.activation_fn}) received unexpected input shape {layer_inputs_all_tokens.shape} for layer {layer_idx}. Empirical L0 for this layer will be NaN."
                    )
                    layer_inputs_flat = None

                if layer_inputs_flat is not None and num_tokens_in_batch > 0:
                    num_to_sample = min(num_tokens_to_check, num_tokens_in_batch)
                    indices = torch.randperm(num_tokens_in_batch, device=model.device)[:num_to_sample]
                    selected_tokens_for_l0 = layer_inputs_flat[indices]

                    if selected_tokens_for_l0.numel() > 0:
                        # model.encode() is appropriate here for JumpReLU as it's per-layer
                        acts_layer_selected = model.encode(selected_tokens_for_l0, layer_idx=layer_idx)
                        l0_per_token_selected = (acts_layer_selected.abs() > 1e-8).sum(dim=1).float()
                        avg_empirical_l0_this_layer = l0_per_token_selected.mean().item()
                    else:
                        logger.warning(
                            f"No tokens selected for empirical L0 check (for {model.config.activation_fn}) for layer {layer_idx} after sampling. Empirical L0 for this layer will be NaN."
                        )
                elif layer_inputs_flat is None:
                    pass  # Warning already issued
                else:  # num_tokens_in_batch == 0
                    logger.warning(
                        f"Batch for L0 check (for {model.config.activation_fn}) contains no tokens for layer {layer_idx}. Empirical L0 for this layer will be NaN."
                    )
            empirical_l0s_per_layer[layer_idx] = avg_empirical_l0_this_layer

    return empirical_l0s_per_layer


def _override_store_norm_stats(store, stats_path: Optional[str]):
    """If stats_path is provided, load that JSON and inject into the store for normalisation & evaluator."""
    if not stats_path:
        return None, None
    try:
        with open(stats_path) as f:
            stats_json = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load norm_stats from {stats_path}: {e}")
        return None, None
    # Populate the tensors the same way ManifestActivationStore._prep_norm does
    mean_tg: Dict[int, torch.Tensor] = {}
    std_tg: Dict[int, torch.Tensor] = {}
    mean_in: Dict[int, torch.Tensor] = {}
    std_in: Dict[int, torch.Tensor] = {}
    device = store.device
    for layer_idx_str, stats in stats_json.items():
        li = int(layer_idx_str)
        if "inputs" in stats and "mean" in stats["inputs"] and "std" in stats["inputs"]:
            mean_in[li] = torch.tensor(stats["inputs"]["mean"], device=device, dtype=torch.float32).unsqueeze(0)
            std_in[li] = (torch.tensor(stats["inputs"]["std"], device=device, dtype=torch.float32) + 1e-6).unsqueeze(0)
        if "targets" in stats and "mean" in stats["targets"] and "std" in stats["targets"]:
            mean_tg[li] = torch.tensor(stats["targets"]["mean"], device=device, dtype=torch.float32).unsqueeze(0)
            std_tg[li] = (torch.tensor(stats["targets"]["std"], device=device, dtype=torch.float32) + 1e-6).unsqueeze(0)
    if mean_in and std_in:
        store.mean_in, store.std_in = mean_in, std_in
        store.mean_tg, store.std_tg = mean_tg, std_tg
        store.apply_normalization = True
        logger.info(f"Overrode normalisation stats with {stats_path}")
    return mean_tg, std_tg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a BatchTopK CLT model to JumpReLU with post-hoc theta estimation."
    )

    parser.add_argument(
        "--batchtopk-checkpoint-path",
        type=str,
        required=True,
        help="Path to the saved BatchTopK model checkpoint (.pt file or sharded directory).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the JSON config file corresponding to the BatchTopK model.",
    )
    parser.add_argument(
        "--activation-data-path",
        type=str,
        required=True,
        help="Path to the activation data manifest directory (for LocalActivationStore).",
    )
    parser.add_argument(
        "--output-model-path",
        type=str,
        required=True,
        help="Path to save the converted JumpReLU model's state_dict (.pt).",
    )
    parser.add_argument(
        "--output-config-path",
        type=str,
        required=True,
        help="Path to save the converted JumpReLU model's config (.json).",
    )

    parser.add_argument(
        "--num-batches-for-theta-estimation",
        type=int,
        default=100,
        help="Number of batches to use for theta estimation.",
    )
    parser.add_argument(
        "--estimation-batch-size-tokens",
        type=int,
        default=1024,
        help="Number of tokens per batch for theta estimation.",
    )
    parser.add_argument(
        "--default-theta-value", type=float, default=1e6, help="Default theta value for features that never activated."
    )

    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (e.g., 'cpu', 'cuda', 'cuda:0'). Auto-detects if None."
    )
    parser.add_argument(
        "--activation-dtype",
        type=str,
        default="float32",
        help="Data type for loading activations (e.g., 'float16', 'bfloat16', 'float32').",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for activation store sampling (if applicable)."
    )
    parser.add_argument(
        "--num-tokens-for-l0-check-script",
        type=int,
        default=1024,
        help="Number of random tokens to sample from the collected data for the empirical L0 check in this script.",
    )
    parser.add_argument(
        "--num-batches-for-l0-check",
        type=int,
        default=1,
        help="Number of batches to fetch from activation store for the empirical L0 check.",
    )
    parser.add_argument(
        "--l0-check-batch-size-tokens",
        type=int,
        default=None,  # Default to None, will use estimation_batch_size_tokens if not set
        help="Number of tokens per batch for fetching data for the L0 check. Defaults to estimation_batch_size_tokens if not specified.",
    )

    # Args for Layer-wise L0 Calibration
    parser.add_argument(
        "--l0-layerwise-calibrate",
        action="store_true",
        help="If set, perform an additional calibration step to match layer-wise L0s from the original model.",
    )
    parser.add_argument(
        "--l0-calibration-batches",
        type=int,
        default=1,
        help="Number of batches from activation_data_path to use for L0 calibration step.",
    )
    parser.add_argument(
        "--l0-calibration-batch-size-tokens",
        type=int,
        default=None,
        help="Batch size in tokens for L0 calibration. Defaults to l0_check_batch_size_tokens or estimation_batch_size_tokens.",
    )
    parser.add_argument(
        "--l0-target-model-config-path",
        type=str,
        default=None,
        help="Path to the original model's config JSON for L0 target calculation. Defaults to --config_path.",
    )
    parser.add_argument(
        "--l0-target-model-checkpoint-path",
        type=str,
        default=None,
        help="Path to the original model's checkpoint for L0 target calculation. Defaults to --batchtopk_checkpoint_path.",
    )
    parser.add_argument(
        "--l0-calibration-tolerance",
        type=float,
        default=0.5,
        help="Tolerance (in number of active features) for matching target L0 during layer-wise calibration.",
    )
    parser.add_argument(
        "--l0-calibration-search-min-scale",
        type=float,
        default=0.1,
        help="Minimum scale factor for layer-wise L0 calibration search.",
    )
    parser.add_argument(
        "--l0-calibration-search-max-scale",
        type=float,
        default=10.0,
        help="Maximum scale factor for layer-wise L0 calibration search.",
    )
    parser.add_argument(
        "--l0-calibration-max-iters",
        type=int,
        default=15,
        help="Maximum iterations for binary search per layer during L0 calibration.",
    )

    parser.add_argument(
        "--norm-stats-path",
        type=str,
        default=None,
        help="Optional path to a norm_stats.json file to override the store's statistics (use the training stats for consistent NMSE/EV).",
    )

    # Note: clt_dtype for the model will be taken from the loaded config_dict_batchtopk initially.

    args = parser.parse_args()
    main(args)

# Example Usage:
# python scripts/convert_batchtopk_to_jumprelu.py \\
#   --batchtopk-checkpoint-path clt_training_logs/clt_pythia_batchtopk_train_XYZ/final \\
#   --config-path clt_training_logs/clt_pythia_batchtopk_train_XYZ/cfg.json \\
#   --activation-data-path ./tutorial_activations_local_1M_pythia/EleutherAI/pythia-70m/monology/pile-uncopyrighted_train \\
#   --output-model-path clt_training_logs/clt_pythia_batchtopk_train_XYZ/final_jumprelu_sf1.5/clt_model_jumprelu.pt \\
#   --output-config-path clt_training_logs/clt_pythia_batchtopk_train_XYZ/final_jumprelu_sf1.5/cfg_jumprelu.json \\
#   --num-batches-for-theta-estimation 50 \\
#   --device cuda
#   --l0-layerwise-calibrate \
#   --l0-calibration-tolerance 0.2
