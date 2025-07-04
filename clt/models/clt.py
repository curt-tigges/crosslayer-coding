import torch
from typing import Dict, Optional, Union, Tuple, List, Any
import logging

from clt.config import CLTConfig
from clt.models.base import BaseTranscoder

from clt.models.activations import _apply_batch_topk_helper, _apply_token_topk_helper
from clt.models.activations_local_global import _apply_batch_topk_local_global
from clt.models.activations_optimized import _apply_batch_topk_two_stage
from clt.models.encoder import Encoder
from clt.models.decoder import Decoder
from clt.models.theta import ThetaManager

from clt.activations.registry import get_activation_fn
from clt.parallel import ops as dist_ops

from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class CrossLayerTranscoder(BaseTranscoder):
    """Implementation of a Cross-Layer Transcoder (CLT) with tensor parallelism."""

    _cached_decoder_norms: Optional[torch.Tensor] = None

    device: torch.device
    dtype: torch.dtype

    def __init__(
        self,
        config: CLTConfig,
        process_group: Optional["ProcessGroup"],
        device: Optional[torch.device] = None,
        profiler: Optional[Any] = None,
    ):
        super().__init__(config)
        self.process_group = process_group
        self.world_size = dist_ops.get_world_size(process_group)
        self.rank = dist_ops.get_rank(process_group)
        self.profiler = profiler

        self.dtype = self._resolve_dtype(config.clt_dtype)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"CLT TP model initialized on rank {self.rank} with device {self.device} and dtype {self.dtype}")

        self.encoder_module = Encoder(
            config=config, process_group=self.process_group, device=self.device, dtype=self.dtype
        )
        self.decoder_module = Decoder(
            config=config, process_group=self.process_group, device=self.device, dtype=self.dtype
        )
        self.theta_manager = ThetaManager(
            config=config, process_group=self.process_group, device=self.device, dtype=self.dtype
        )

    def _resolve_dtype(self, dtype_input: Optional[Union[str, torch.dtype]]) -> torch.dtype:
        if isinstance(dtype_input, torch.dtype):
            return dtype_input
        if isinstance(dtype_input, str):
            try:
                dtype = getattr(torch, dtype_input)
                if isinstance(dtype, torch.dtype):
                    return dtype
                else:
                    logger.warning(f"Resolved '{dtype_input}' but it is not a torch.dtype. " f"Defaulting to float32.")
                    return torch.float32
            except AttributeError:
                logger.warning(f"Unsupported CLT dtype string: '{dtype_input}'. " f"Defaulting to float32.")
                return torch.float32
        return torch.float32

    def jumprelu(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply JumpReLU activation function for a specific layer."""
        return self.theta_manager.jumprelu(x, layer_idx)

    def get_preactivations(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.encoder_module.get_preactivations(x, layer_idx)

    def _encode_all_layers(
        self, inputs: Dict[int, torch.Tensor]
    ) -> Tuple[Dict[int, torch.Tensor], List[Tuple[int, int, int]]]:
        return self.encoder_module.encode_all_layers(inputs)

    def _apply_batch_topk(
        self,
        preactivations_dict: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        # Use two-stage optimization if enabled
        if self.config.two_stage_batchtopk:
            return _apply_batch_topk_two_stage(
                preactivations_dict, self.config, self.device, self.dtype, self.rank, self.process_group, self.profiler
            )
        # Use optimized local-global approach for multi-GPU training
        elif self.world_size > 1:
            return _apply_batch_topk_local_global(
                preactivations_dict, self.config, self.device, self.dtype, self.rank, self.process_group, self.profiler
            )
        else:
            # Single GPU uses original implementation
            return _apply_batch_topk_helper(
                preactivations_dict, self.config, self.device, self.dtype, self.rank, self.process_group, self.profiler
            )

    def _apply_token_topk(
        self,
        preactivations_dict: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        return _apply_token_topk_helper(
            preactivations_dict, self.config, self.device, self.dtype, self.rank, self.process_group, self.profiler
        )

    def encode(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Encode the input activations at the specified layer.
        This method is primarily for 'relu' and 'jumprelu' activations.
        BatchTopK/TokenTopK are handled in get_feature_activations.
        """
        x = x.to(device=self.device, dtype=self.dtype)
        fallback_tensor: Optional[torch.Tensor] = None
        activated: Optional[torch.Tensor] = None
        preact = self.get_preactivations(x, layer_idx)

        if preact.numel() == 0:
            logger.warning(f"Rank {self.rank}: Received empty preactivations for encode layer {layer_idx}.")
            batch_dim = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1] if x.dim() == 3 else 0
            fallback_shape = (batch_dim, self.config.num_features)
            fallback_tensor = torch.zeros(fallback_shape, device=self.device, dtype=self.dtype)
        elif preact.shape[-1] != self.config.num_features:
            logger.warning(
                f"Rank {self.rank}: Received invalid preactivations shape {preact.shape} for encode layer {layer_idx}."
            )
            fallback_shape = (preact.shape[0], self.config.num_features)
            fallback_tensor = torch.zeros(fallback_shape, device=self.device, dtype=self.dtype)
        else:
            try:
                if self.config.activation_fn == "jumprelu":
                    activated = self.theta_manager.jumprelu(preact, layer_idx)
                elif self.config.activation_fn == "relu":
                    activation_fn_callable = get_activation_fn("relu")  # Standard ReLU from registry
                    activated = activation_fn_callable(self, preact, layer_idx)  # Corrected signature
                else:
                    # This path should ideally not be taken if BatchTopK/TokenTopK are handled elsewhere.
                    # If other activation functions are added that fit this per-layer, per-token model,
                    # ensure get_activation_fn returns a callable with the correct signature.
                    logger.error(
                        f"Rank {self.rank}: Unsupported activation function '{self.config.activation_fn}' encountered in encode method path. Expected jumprelu or relu."
                    )
                    # Fallback to zero tensor to avoid crashing, but this indicates a logic issue.
                    fallback_shape = (preact.shape[0], self.config.num_features)
                    fallback_tensor = torch.zeros(fallback_shape, device=self.device, dtype=self.dtype)

            except ValueError as e:  # Catch if activation function name is not in registry
                logger.error(
                    f"Rank {self.rank}: Error getting activation function '{self.config.activation_fn}' for layer {layer_idx}: {e}"
                )
                fallback_shape = (preact.shape[0], self.config.num_features)
                fallback_tensor = torch.zeros(fallback_shape, device=self.device, dtype=self.dtype)
            except Exception as e:
                logger.error(
                    f"Rank {self.rank}: Unexpected error during activation function '{self.config.activation_fn}' for layer {layer_idx}: {e}"
                )
                fallback_shape = (preact.shape[0], self.config.num_features)
                fallback_tensor = torch.zeros(fallback_shape, device=self.device, dtype=self.dtype)

        if activated is not None:
            return activated
        elif fallback_tensor is not None:
            return fallback_tensor
        else:
            # This state implies an issue if neither `activated` nor `fallback_tensor` was set.
            # For instance, if preact was valid but the activation_fn logic path didn't set either.
            logger.critical(
                f"Rank {self.rank}: Critical logic error in encode for layer {layer_idx}. Activation function '{self.config.activation_fn}' not properly handled leading to no output."
            )
            # Return a zero tensor of the expected output shape as a last resort before crashing.
            expected_batch_dim = (
                preact.shape[0]
                if preact.numel() > 0
                else (x.shape[0] if x.dim() == 2 else (x.shape[0] * x.shape[1] if x.dim() == 3 else 0))
            )
            return torch.zeros((expected_batch_dim, self.config.num_features), device=self.device, dtype=self.dtype)

    def decode(
        self, a: Dict[int, torch.Tensor], layer_idx: int, source_inputs: Optional[Dict[int, torch.Tensor]] = None
    ) -> torch.Tensor:
        return self.decoder_module.decode(a, layer_idx, source_inputs)

    def forward(self, inputs: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        activations = self.get_feature_activations(inputs)

        # Note: feature affine transformations are now applied in the decoder

        reconstructions = {}
        for layer_idx in range(self.config.num_layers):
            relevant_activations = {k: v for k, v in activations.items() if k <= layer_idx and v.numel() > 0}
            if layer_idx in inputs and relevant_activations:
                # Pass source inputs for EleutherAI-style skip connections
                source_inputs = (
                    {k: inputs[k] for k in range(layer_idx + 1) if k in inputs} if self.config.skip_connection else None
                )
                reconstruction = self.decode(relevant_activations, layer_idx, source_inputs)

                reconstructions[layer_idx] = reconstruction
            elif layer_idx in inputs:
                batch_size = 0
                input_tensor = inputs[layer_idx]
                if input_tensor.dim() >= 1:
                    batch_size = (
                        input_tensor.shape[0] * input_tensor.shape[1]
                        if input_tensor.dim() == 3
                        else input_tensor.shape[0]
                    )
                else:
                    logger.warning(
                        f"Rank {self.rank}: Could not determine batch size for fallback tensor in forward layer {layer_idx} from input shape {input_tensor.shape}. Using 0."
                    )
                reconstructions[layer_idx] = torch.zeros(
                    (batch_size, self.config.d_model),
                    device=self.device,
                    dtype=self.dtype,
                )
        return reconstructions

    def get_feature_activations(self, inputs: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        processed_inputs: Dict[int, torch.Tensor] = {}
        for layer_idx, x_orig in inputs.items():
            processed_inputs[layer_idx] = x_orig.to(device=self.device, dtype=self.dtype)

        if self.config.activation_fn == "batchtopk" or self.config.activation_fn == "topk":
            # Check if we should use per-layer mode
            if self.config.topk_mode == "per_layer":
                # Use per-layer top-k by calling encode on each layer
                activations = {}
                for layer_idx in sorted(processed_inputs.keys()):
                    x_input = processed_inputs[layer_idx]
                    act = self.encode(x_input, layer_idx)
                    activations[layer_idx] = act
                return activations

            # Otherwise use global top-k
            preactivations_dict, _ = self._encode_all_layers(processed_inputs)
            if not preactivations_dict:
                activations = {}
                for layer_idx_orig_input in inputs.keys():
                    x_orig_input = inputs[layer_idx_orig_input]
                    batch_dim_fallback = 0
                    if x_orig_input.dim() == 3:
                        batch_dim_fallback = x_orig_input.shape[0] * x_orig_input.shape[1]
                    elif x_orig_input.dim() == 2:
                        batch_dim_fallback = x_orig_input.shape[0]
                    activations[layer_idx_orig_input] = torch.zeros(
                        (batch_dim_fallback, self.config.num_features), device=self.device, dtype=self.dtype
                    )
                return activations

            if self.config.activation_fn == "batchtopk":
                if self.profiler:
                    with self.profiler.timer("batchtopk_activation") as timer:
                        activations = self._apply_batch_topk(preactivations_dict)
                    if hasattr(timer, "elapsed"):
                        self.profiler.record("batchtopk_activation", timer.elapsed)
                else:
                    activations = self._apply_batch_topk(preactivations_dict)
            elif self.config.activation_fn == "topk":
                if self.profiler:
                    with self.profiler.timer("topk_activation") as timer:
                        activations = self._apply_token_topk(preactivations_dict)
                    if hasattr(timer, "elapsed"):
                        self.profiler.record("topk_activation", timer.elapsed)
                else:
                    activations = self._apply_token_topk(preactivations_dict)
            else:
                raise ValueError(f"Unexpected activation_fn '{self.config.activation_fn}' in BatchTopK/TokenTopK path.")
            return activations
        else:  # ReLU or JumpReLU (per-layer activation)
            activations = {}
            for layer_idx in sorted(processed_inputs.keys()):
                x_input = processed_inputs[layer_idx]
                act = self.encode(x_input, layer_idx)
                activations[layer_idx] = act
            return activations

    def get_decoder_norms(self) -> torch.Tensor:
        return self.decoder_module.get_decoder_norms()

    @torch.no_grad()
    def estimate_theta_posthoc(
        self,
        data_iter: torch.utils.data.IterableDataset,
        num_batches: Optional[int] = None,
        default_theta_value: float = 1e6,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Estimate theta post-hoc using a specified number of batches."""
        original_tm_device = self.theta_manager.device
        target_device_tm = device if device is not None else self.device

        if target_device_tm != original_tm_device:
            logger.info(f"Rank {self.rank}: Moving ThetaManager to {target_device_tm} for theta estimation.")
            self.theta_manager.to(target_device_tm)

        estimated_thetas_result = self.theta_manager.estimate_theta_posthoc(
            encode_all_layers_fn=self.encoder_module.encode_all_layers,
            data_iter=data_iter,
            num_batches=num_batches,
            default_theta_value=default_theta_value,
        )
        if target_device_tm != original_tm_device:
            logger.info(f"Rank {self.rank}: Moving ThetaManager back to {original_tm_device}.")
            self.theta_manager.to(original_tm_device)
        return estimated_thetas_result

    @torch.no_grad()
    def convert_to_jumprelu_inplace(self, default_theta_value: float = 1e6) -> None:
        """
        Converts the model to use JumpReLU activation based on learned BatchTopK thresholds.
        This method delegates to ThetaManager, which updates the shared config object.
        """
        self.theta_manager.convert_to_jumprelu_inplace(default_theta_value=default_theta_value)
        logger.info(
            f"Rank {self.rank}: CLT model config updated by ThetaManager. New activation_fn='{self.config.activation_fn}'."
        )

    # --- Back-compat: expose ThetaManager.log_threshold at model level ---
    @property
    def log_threshold(self) -> Optional[torch.nn.Parameter]:
        """Proxy to ``theta_manager.log_threshold`` for backward compatibility.

        Older training scripts, conversion utilities and tests referenced
        ``model.log_threshold`` directly.  After the Step-5 refactor the
        parameter migrated into the dedicated ``ThetaManager`` module.  We
        now expose a read-only view that always returns the *current* parameter
        held by ``self.theta_manager``.  Modifying the returned tensor (e.g.
        in-place updates to ``.data``) therefore continues to work as before.
        Assigning a brand-new ``nn.Parameter`` to ``model.log_threshold`` will
        forward the assignment to ``theta_manager`` to preserve the linkage.
        """
        if hasattr(self, "theta_manager") and self.theta_manager is not None:
            return self.theta_manager.log_threshold
        return None

    @log_threshold.setter
    def log_threshold(self, new_param: Optional[torch.nn.Parameter]) -> None:
        # Keep property writable so callers that used to assign a fresh
        # parameter (rare) do not break.  We delegate the storage to
        # ``ThetaManager`` so there is a single source of truth.
        if not hasattr(self, "theta_manager") or self.theta_manager is None:
            raise AttributeError("ThetaManager is not initialised; cannot set log_threshold.")
        self.theta_manager.log_threshold = new_param

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with backward compatibility for old checkpoints.

        Handles:
        1. Old untied decoder format -> new tied/untied format
        2. Missing theta_bias/theta_scale parameters
        3. Missing per_target_scale/per_target_bias parameters
        """
        # Check if this is an old checkpoint by looking for decoder keys
        old_format_decoder_keys = [k for k in state_dict.keys() if "decoders." in k and "->" in k]
        is_old_checkpoint = len(old_format_decoder_keys) > 0

        if is_old_checkpoint and self.config.decoder_tying == "per_source":
            logger.warning(
                "Loading old untied decoder checkpoint into tied decoder model. "
                "This will use weights from the first target layer for each source layer."
            )

            # Convert old decoder weights to tied format
            # For each source layer, use the weights from src->src decoder
            new_state_dict = {}
            for key, value in state_dict.items():
                if "decoders." in key and "->" in key:
                    # Extract source and target layer indices
                    # Key format: "decoder_module.decoders.{src}->{tgt}.weight" or ".bias"
                    parts = key.split(".")
                    decoder_key_idx = parts.index("decoders") + 1
                    src_tgt = parts[decoder_key_idx].split("->")
                    src_layer = int(src_tgt[0])
                    tgt_layer = int(src_tgt[1])
                    param_type = parts[-1]  # 'weight' or 'bias'

                    # Only use diagonal decoders (src->src) for tied architecture
                    if src_layer == tgt_layer:
                        new_key = ".".join(parts[:decoder_key_idx] + [str(src_layer), param_type])
                        new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        # Handle feature affine parameters migration from encoder to decoder module
        # (for backward compatibility with old checkpoints)
        for i in range(self.config.num_layers):
            old_offset_key = f"encoder_module.feature_offset.{i}"
            new_offset_key = f"decoder_module.feature_offset.{i}"
            if old_offset_key in state_dict and new_offset_key not in state_dict:
                logger.info(f"Migrating {old_offset_key} to {new_offset_key}")
                state_dict[new_offset_key] = state_dict.pop(old_offset_key)

            old_scale_key = f"encoder_module.feature_scale.{i}"
            new_scale_key = f"decoder_module.feature_scale.{i}"
            if old_scale_key in state_dict and new_scale_key not in state_dict:
                logger.info(f"Migrating {old_scale_key} to {new_scale_key}")
                state_dict[new_scale_key] = state_dict.pop(old_scale_key)

        # Handle missing feature affine parameters (now in decoder module)
        if (
            self.config.enable_feature_offset
            and hasattr(self.decoder_module, "feature_offset")
            and self.decoder_module.feature_offset is not None
        ):
            for i in range(self.config.num_layers):
                key = f"decoder_module.feature_offset.{i}"
                if key not in state_dict:
                    logger.info(f"Initializing missing {key} to zeros")
                    # Don't add to state_dict to let it be initialized by the module

        if (
            self.config.enable_feature_scale
            and hasattr(self.decoder_module, "feature_scale")
            and self.decoder_module.feature_scale is not None
        ):
            for i in range(self.config.num_layers):
                key = f"decoder_module.feature_scale.{i}"
                if key not in state_dict:
                    logger.info(f"Initializing missing {key} (first target layer to ones, rest to zeros)")
                    # Don't add to state_dict to let it be initialized by the module

        # Handle missing skip weights
        if self.config.skip_connection and hasattr(self.decoder_module, "skip_weights"):
            for i in range(self.config.num_layers):
                key = f"decoder_module.skip_weights.{i}"
                if key not in state_dict:
                    logger.info(f"Initializing missing {key} to identity matrix")

        # Call parent's load_state_dict
        return super().load_state_dict(state_dict, strict=strict)
