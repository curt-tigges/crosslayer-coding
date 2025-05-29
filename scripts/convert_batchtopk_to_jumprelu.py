import torch
import os
import json
import argparse
import sys
import logging
from typing import Dict, List
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
    from safetensors.torch import load_file as load_safetensors_file  # Added for safetensors support

    # Add other store types if needed, e.g., RemoteActivationStore
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure the 'clt' library is installed or the clt directory is in your PYTHONPATH.")
    print("If the error is related to 'safetensors', please install it: pip install safetensors")
    raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _remap_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remaps old state_dict keys to the new format with module prefixes."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoders."):
            new_key = "encoder_module." + key
        elif key.startswith("decoders."):
            new_key = "decoder_module." + key
        else:
            new_key = key
        new_state_dict[new_key] = value
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
                state_dict = load_safetensors_file(args.batchtopk_checkpoint_path, device=device.type)
                state_dict = _remap_checkpoint_keys(state_dict)  # Remap keys
                model.load_state_dict(state_dict)
            else:
                logger.info(f"Loading BatchTopK model state from .pt file: {args.batchtopk_checkpoint_path}")
                state_dict = torch.load(args.batchtopk_checkpoint_path, map_location=device)
                state_dict = _remap_checkpoint_keys(state_dict)  # Remap keys
                model.load_state_dict(state_dict)

        model.eval()
        logger.info("BatchTopK model loaded and set to eval mode.")
    except Exception as e:
        logger.error(f"Error loading BatchTopK model state: {e}")
        return

    # 3. Initialize ActivationStore for theta estimation
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
            normalization_method="auto",  # Or "none" if data is not normalized
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
            data_iter=activation_store_theta,
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
    torch.save(model.state_dict(), args.output_model_path)

    logger.info(f"Saving converted JumpReLU model config to: {args.output_config_path}")
    os.makedirs(os.path.dirname(args.output_config_path), exist_ok=True)
    with open(args.output_config_path, "w") as f:
        # model.config is updated inplace by convert_to_jumprelu_inplace (called by estimate_theta_posthoc)
        json.dump(model.config.__dict__, f, indent=4)

    # 6. Perform a quick L0 and NMSE check on the converted model
    logger.info("Performing L0 and NMSE check on the converted JumpReLU model...")

    all_sample_inputs_for_l0: Dict[int, List[torch.Tensor]] = {i: [] for i in range(model.config.num_layers)}
    all_sample_targets_for_nmse: Dict[int, List[torch.Tensor]] = {i: [] for i in range(model.config.num_layers)}
    # total_tokens_collected_per_layer: Dict[int, int] = {i: 0 for i in range(model.config.num_layers)} # No longer needed with direct cat

    l0_check_fetch_batch_size = (
        args.l0_check_batch_size_tokens
        if hasattr(args, "l0_check_batch_size_tokens") and args.l0_check_batch_size_tokens is not None
        else args.estimation_batch_size_tokens
    )

    logger.info(
        f"Collecting data for L0/NMSE check using {args.num_batches_for_l0_check} batches with fetch batch size {l0_check_fetch_batch_size} tokens."
    )

    mean_tg_for_eval = None  # To store normalization stats for Evaluator
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
            normalization_method="auto",
        )
        data_iterator_for_l0_check = iter(activation_store_for_l0_check)

        # Retrieve mean_tg and std_tg if store has them (after iter is created, stats should be available if auto)
        if hasattr(activation_store_for_l0_check, "mean_tg") and hasattr(activation_store_for_l0_check, "std_tg"):
            mean_tg_for_eval = activation_store_for_l0_check.mean_tg
            std_tg_for_eval = activation_store_for_l0_check.std_tg
            if mean_tg_for_eval and std_tg_for_eval:
                logger.info("Retrieved mean_tg and std_tg from L0 check activation store for NMSE de-normalization.")
            else:
                logger.info(
                    "mean_tg or std_tg not available from L0 check store. NMSE will be on potentially normalized values."
                )

        for batch_idx in range(args.num_batches_for_l0_check):
            try:
                sample_inputs_batch, sample_targets_batch = next(data_iterator_for_l0_check)
                for layer_idx in sample_inputs_batch.keys():  # Iterate over layers present in the input batch
                    if layer_idx in all_sample_inputs_for_l0:
                        input_tensor_data = sample_inputs_batch[layer_idx]
                        if input_tensor_data.dim() == 3:
                            num_tokens = input_tensor_data.shape[0] * input_tensor_data.shape[1]
                            all_sample_inputs_for_l0[layer_idx].append(
                                input_tensor_data.reshape(num_tokens, model.config.d_model)
                            )
                        elif input_tensor_data.dim() == 2:
                            all_sample_inputs_for_l0[layer_idx].append(input_tensor_data)

                    if (
                        layer_idx in sample_targets_batch and layer_idx in all_sample_targets_for_nmse
                    ):  # Check if target exists for the layer
                        target_tensor_data = sample_targets_batch[layer_idx]
                        if target_tensor_data.dim() == 3:
                            num_tokens = target_tensor_data.shape[0] * target_tensor_data.shape[1]
                            all_sample_targets_for_nmse[layer_idx].append(
                                target_tensor_data.reshape(num_tokens, model.config.d_model)
                            )
                        elif target_tensor_data.dim() == 2:
                            all_sample_targets_for_nmse[layer_idx].append(target_tensor_data)

            except StopIteration:
                logger.warning(
                    f"Activation store exhausted after {batch_idx + 1} batches during L0/NMSE check data collection. Proceeding with collected data."
                )
                break

        if hasattr(activation_store_for_l0_check, "close") and callable(
            getattr(activation_store_for_l0_check, "close")
        ):
            activation_store_for_l0_check.close()

    except Exception as e_l0_fetch:
        logger.warning(
            f"Error initializing or fetching batches for L0/NMSE check: {e_l0_fetch}. Check might use zero or incomplete input."
        )

    final_sample_batch_for_l0_inputs: Dict[int, torch.Tensor] = {}
    for layer_idx, tensor_list in all_sample_inputs_for_l0.items():
        if tensor_list:
            final_sample_batch_for_l0_inputs[layer_idx] = torch.cat(tensor_list, dim=0)
            logger.info(
                f"Layer {layer_idx}: Collected {final_sample_batch_for_l0_inputs[layer_idx].shape[0]} total input tokens for L0/NMSE check."
            )
        else:
            logger.warning(f"Layer {layer_idx}: No input tokens collected for L0/NMSE check.")
            final_sample_batch_for_l0_inputs[layer_idx] = torch.empty(
                (0, model.config.d_model), device=device, dtype=model.dtype
            )

    final_sample_targets_for_nmse_check: Dict[int, torch.Tensor] = {}
    for layer_idx, tensor_list in all_sample_targets_for_nmse.items():
        if tensor_list:
            final_sample_targets_for_nmse_check[layer_idx] = torch.cat(tensor_list, dim=0)
            logger.info(
                f"Layer {layer_idx}: Collected {final_sample_targets_for_nmse_check[layer_idx].shape[0]} total target tokens for NMSE check."
            )
        else:
            logger.warning(f"Layer {layer_idx}: No target tokens collected for NMSE check.")
            final_sample_targets_for_nmse_check[layer_idx] = torch.empty(
                (0, model.config.d_model), device=device, dtype=model.dtype
            )

    model_for_l0_check = model

    empirical_l0s_per_layer = run_quick_l0_checks_script(
        model_for_l0_check, final_sample_batch_for_l0_inputs, args.num_tokens_for_l0_check_script
    )
    logger.info(f"Empirical L0 per layer (out of {args.num_tokens_for_l0_check_script} sampled tokens):")
    total_empirical_l0 = 0.0
    for l_idx, l0_val in empirical_l0s_per_layer.items():
        logger.info(f"  Layer {l_idx}: {l0_val:.2f}")
        if not (isinstance(l0_val, float) and math.isnan(l0_val)):
            total_empirical_l0 += l0_val
    logger.info(f"Total Empirical L0 across all layers: {total_empirical_l0:.2f}")

    # Compute and log NMSE
    logger.info("Computing NMSE...")
    evaluator = CLTEvaluator(model=model_for_l0_check, device=device, mean_tg=mean_tg_for_eval, std_tg=std_tg_for_eval)

    # Ensure inputs and targets for NMSE have corresponding layers and non-empty tensors before calling
    valid_layers_for_nmse = set(final_sample_batch_for_l0_inputs.keys()) & set(
        final_sample_targets_for_nmse_check.keys()
    )
    inputs_for_nmse_metric = {
        k: v for k, v in final_sample_batch_for_l0_inputs.items() if k in valid_layers_for_nmse and v.numel() > 0
    }
    targets_for_nmse_metric = {
        k: v for k, v in final_sample_targets_for_nmse_check.items() if k in valid_layers_for_nmse and v.numel() > 0
    }

    if (
        not inputs_for_nmse_metric
        or not targets_for_nmse_metric
        or not any(v.numel() > 0 for v in inputs_for_nmse_metric.values())
        or not any(v.numel() > 0 for v in targets_for_nmse_metric.values())
    ):
        logger.warning(
            "Insufficient data for NMSE calculation after filtering (empty inputs or targets for common layers). Skipping NMSE."
        )
    else:
        # Get reconstructions from the model using the collected inputs
        with torch.no_grad():  # Ensure no gradients are computed during this forward pass
            reconstructions_for_nmse = model_for_l0_check(inputs_for_nmse_metric)

        reconstruction_metrics = evaluator._compute_reconstruction_metrics(
            targets=targets_for_nmse_metric, reconstructions=reconstructions_for_nmse
        )
        nmse_value = reconstruction_metrics.get("reconstruction/normalized_mean_reconstruction_error", float("nan"))
        explained_variance = reconstruction_metrics.get("reconstruction/explained_variance", float("nan"))
        logger.info(f"Normalized Mean Squared Error (NMSE) on collected data: {nmse_value:.4f}")
        logger.info(f"Explained Variance (EV) on collected data: {explained_variance:.4f}")

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
                state_dict_orig = load_safetensors_file(original_checkpoint_path, device=device.type)
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
                normalization_method="auto",  # Let it normalize if needed, though L0 is on post-activation
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
                    final_calibration_inputs[layer_idx_cal] = torch.cat(tensor_list_cal, dim=0)
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
            model_to_calibrate=model_for_l0_check,
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
        logger.info("--- L0s after Layer-wise Calibration ---")
        calibrated_l0s_per_layer = run_quick_l0_checks_script(
            model_for_l0_check, final_calibration_inputs, args.num_tokens_for_l0_check_script
        )
        total_calibrated_l0 = 0.0
        for l_idx, l0_val in calibrated_l0s_per_layer.items():
            logger.info(f"  Layer {l_idx}: {l0_val:.2f} (Target: {target_l0s.get(l_idx, float('nan')):.2f})")
            if not (isinstance(l0_val, float) and math.isnan(l0_val)):
                total_calibrated_l0 += l0_val
        logger.info(f"Total Empirical L0 across all layers (Calibrated): {total_calibrated_l0:.2f}")

        # 6. Re-save the calibrated model
        logger.info(f"Re-saving calibrated JumpReLU model state to: {args.output_model_path}")
        torch.save(model_for_l0_check.state_dict(), args.output_model_path)
        # Config remains the same (JumpReLU), only log_thresholds changed.
        logger.info("--- Layer-wise L0 Calibration Step Finished ---")

        # --- Re-evaluate NMSE/EV after L0 calibration ---
        logger.info("--- NMSE/EV after Layer-wise Calibration ---")
        if (
            not inputs_for_nmse_metric  # This was defined before the calibration block
            or not targets_for_nmse_metric
            or not any(v.numel() > 0 for v in inputs_for_nmse_metric.values())
            or not any(v.numel() > 0 for v in targets_for_nmse_metric.values())
        ):
            logger.warning(
                "Insufficient data for NMSE re-evaluation after calibration (empty inputs or targets for common layers). Skipping."
            )
        else:
            # Ensure the evaluator uses the potentially updated model_for_l0_check (calibrated model)
            # If evaluator was initialized with model, and model_for_l0_check is the same instance that was modified, this is fine.
            # If not, evaluator might need to be updated or re-initialized with the calibrated model.
            # Assuming model_for_l0_check is the same instance that evaluator holds or that CLTEvaluator uses the model passed at evaluation time.
            # CLTEvaluator constructor takes a model, but its _compute_reconstruction_metrics does not, it uses self.model.
            # So, we need to ensure the evaluator has the *calibrated* model.
            evaluator_after_calib = CLTEvaluator(
                model=model_for_l0_check, device=device, mean_tg=mean_tg_for_eval, std_tg=std_tg_for_eval
            )
            with torch.no_grad():
                reconstructions_after_calib = model_for_l0_check(inputs_for_nmse_metric)

            metrics_after_calib = evaluator_after_calib._compute_reconstruction_metrics(
                targets=targets_for_nmse_metric, reconstructions=reconstructions_after_calib
            )
            nmse_after_calib = metrics_after_calib.get(
                "reconstruction/normalized_mean_reconstruction_error", float("nan")
            )
            ev_after_calib = metrics_after_calib.get("reconstruction/explained_variance", float("nan"))
            logger.info(f"NMSE (post-L0-calibration): {nmse_after_calib:.4f}")
            logger.info(f"EV (post-L0-calibration): {ev_after_calib:.4f}")

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
        model_to_calibrate.log_threshold.data[layer_idx] = torch.log(final_scaled_theta_layer.clamp_min(1e-9))
        logger.info(f"Layer {layer_idx}: Set final scale {current_best_scale:.3f}. Log_threshold updated.")

    logger.info("Layer-wise L0 calibration finished.")


def run_quick_l0_checks_script(
    model: CrossLayerTranscoder, sample_batch_inputs: Dict[int, torch.Tensor], num_tokens_to_check: int
) -> Dict[int, float]:
    """Helper function for L0 checks within the script.
    Returns a dictionary of empirical L0 per layer."""
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
                    # Count **non-zero** activations regardless of sign – BatchTopK can select negative
                    # pre-activations.  We therefore look at the absolute value to decide if a feature
                    # is active.
                    l0_per_token = (acts_layer.abs() > 1e-8).sum(dim=1).float()
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
