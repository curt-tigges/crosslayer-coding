import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, cast, Callable  # Removed Union
import logging
import torch.distributed as dist
from torch.distributed import ProcessGroup

from clt.config import CLTConfig
from clt.models.activations import JumpReLU, BatchTopK, TokenTopK  # Added BatchTopK, TokenTopK
from . import mark_replicated  # For marking log_threshold

logger = logging.getLogger(__name__)


class ThetaManager(nn.Module):
    """
    Manages the log_threshold parameter for JumpReLU, its estimation, and conversion.
    This module also handles the JumpReLU activation function itself.
    """

    log_threshold: Optional[nn.Parameter]
    _sum_min_selected_preact: Optional[torch.Tensor]
    _count_min_selected_preact: Optional[torch.Tensor]
    _avg_layer_means: Optional[torch.Tensor]
    _avg_layer_stds: Optional[torch.Tensor]
    _processed_batches_for_stats: Optional[torch.Tensor]

    def __init__(
        self,
        config: CLTConfig,  # Needs full CLTConfig for num_layers, features, jumprelu_threshold etc.
        process_group: Optional[ProcessGroup],
        device: torch.device,
        dtype: torch.dtype,
        # Add any other necessary parameters from CrossLayerTranscoder like encoder_module if needed for estimate_theta_posthoc
    ):
        super().__init__()
        self.config = config
        self.process_group = process_group
        self.device = device
        self.dtype = dtype
        self.bandwidth = 1.0  # As it was in CrossLayerTranscoder for jumprelu

        if process_group is None or not dist.is_initialized():
            self.world_size = 1
            self.rank = 0
        else:
            self.world_size = dist.get_world_size(process_group)
            self.rank = dist.get_rank(process_group)

        if self.config.activation_fn == "jumprelu":
            if self.config.jumprelu_threshold == 0:
                logger.warning(
                    f"Rank {self.rank}: jumprelu_threshold is 0, expecting to load log_threshold from checkpoint."
                )
            initial_threshold_val = torch.ones(
                config.num_layers, config.num_features, device=self.device, dtype=self.dtype
            ) * torch.log(torch.tensor(config.jumprelu_threshold, device=self.device, dtype=self.dtype))
            self.log_threshold = nn.Parameter(initial_threshold_val)
            mark_replicated(self.log_threshold)
        else:
            self.log_threshold = None

        # Register buffers for theta estimation
        self.register_buffer("_sum_min_selected_preact", None, persistent=False)
        self.register_buffer("_count_min_selected_preact", None, persistent=False)
        self.register_buffer("_avg_layer_means", None, persistent=False)
        self.register_buffer("_avg_layer_stds", None, persistent=False)
        self.register_buffer("_processed_batches_for_stats", None, persistent=False)

    def jumprelu(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply JumpReLU activation function for a specific layer."""
        if self.log_threshold is None:
            logger.error(f"Rank {self.rank}: log_threshold not initialized for JumpReLU. Returning input.")
            return x.to(device=self.device, dtype=self.dtype)  # Ensure output device/dtype
        if layer_idx >= self.log_threshold.shape[0]:
            logger.error(f"Rank {self.rank}: Invalid layer_idx {layer_idx} for log_threshold. Returning input.")
            return x.to(device=self.device, dtype=self.dtype)  # Ensure output device/dtype
        threshold = torch.exp(self.log_threshold[layer_idx]).to(device=self.device, dtype=self.dtype)
        return cast(torch.Tensor, JumpReLU.apply(x, threshold, self.bandwidth))

    @torch.no_grad()
    def _update_min_selected_preactivations(
        self,
        concatenated_preactivations_original: torch.Tensor,
        activated_concatenated: torch.Tensor,
        layer_feature_sizes: List[Tuple[int, int]],
    ):
        """
        Updates the _sum_min_selected_preact and _count_min_selected_preact buffers
        with minimum pre-activation values for features selected by BatchTopK/TokenTopK.
        This function operates with no_grad.
        Buffers are attributes of self (ThetaManager instance).
        """
        if (
            not hasattr(self, "_sum_min_selected_preact")
            or self._sum_min_selected_preact is None
            or not hasattr(self, "_count_min_selected_preact")
            or self._count_min_selected_preact is None
        ):
            # This check is primarily for safety; buffers are initialized in __init__.
            # However, if called when activation_fn isn't batchtopk/topk (where estimation might not run),
            # it's good to be cautious. The main guard is in estimate_theta_posthoc.
            if self.config.activation_fn in ["batchtopk", "topk"]:
                logger.warning(
                    f"Rank {self.rank}: ThetaManager running stats buffers not found or None. Skipping theta update contribution."
                )
            return

        assert self._sum_min_selected_preact is not None and isinstance(
            self._sum_min_selected_preact, torch.Tensor
        ), f"Rank {self.rank}: ThetaManager's _sum_min_selected_preact is not a Tensor or is None."
        assert self._count_min_selected_preact is not None and isinstance(
            self._count_min_selected_preact, torch.Tensor
        ), f"Rank {self.rank}: ThetaManager's _count_min_selected_preact is not a Tensor or is None."

        current_total_feature_offset = 0
        for i, (original_layer_idx, num_features_this_layer) in enumerate(layer_feature_sizes):
            if original_layer_idx >= self._sum_min_selected_preact.shape[0]:
                logger.warning(
                    f"Rank {self.rank}: Invalid original_layer_idx {original_layer_idx} for _min_selected_preact update. Skipping layer."
                )
                current_total_feature_offset += num_features_this_layer
                continue

            preact_orig_this_layer = concatenated_preactivations_original[
                :, current_total_feature_offset : current_total_feature_offset + num_features_this_layer
            ]
            gated_acts_segment = activated_concatenated[
                :, current_total_feature_offset : current_total_feature_offset + num_features_this_layer
            ]

            if gated_acts_segment.shape == preact_orig_this_layer.shape:
                mask_active = gated_acts_segment > 0
                if mask_active.any():
                    masked_preact = torch.where(
                        mask_active,
                        preact_orig_this_layer,
                        torch.full_like(preact_orig_this_layer, float("inf")),
                    )
                    per_feature_min_this_batch = masked_preact.amin(dim=0)
                    if logger.isEnabledFor(logging.DEBUG):
                        finite_mins_for_log = per_feature_min_this_batch[torch.isfinite(per_feature_min_this_batch)]
                        if finite_mins_for_log.numel() > 0:
                            logger.debug(
                                f"Rank {self.rank} Layer {original_layer_idx}: per_feature_min_this_batch (finite values for log) "
                                f"min={finite_mins_for_log.min().item():.4f}, "
                                f"max={finite_mins_for_log.max().item():.4f}, "
                                f"mean={finite_mins_for_log.mean().item():.4f}, "
                                f"median={torch.median(finite_mins_for_log).item():.4f}"
                            )
                        else:
                            logger.debug(
                                f"Rank {self.rank} Layer {original_layer_idx}: No finite per_feature_min_this_batch values to log stats for."
                            )
                        original_preacts_leading_to_positive_gated = preact_orig_this_layer[mask_active]
                        if original_preacts_leading_to_positive_gated.numel() > 0:
                            num_negative_contrib = (original_preacts_leading_to_positive_gated < 0).sum().item()
                            if num_negative_contrib > 0:
                                logger.debug(
                                    f"Rank {self.rank} Layer {original_layer_idx}: {num_negative_contrib} negative original pre-activations "
                                    f"(out of {mask_active.sum().item()} active selections) contributed to theta estimation via positive gated_acts_segment."
                                )
                    valid_mask = torch.isfinite(per_feature_min_this_batch)
                    self._sum_min_selected_preact[original_layer_idx, valid_mask] += per_feature_min_this_batch[
                        valid_mask
                    ]
                    self._count_min_selected_preact[original_layer_idx, valid_mask] += 1
            else:
                logger.warning(
                    f"Rank {self.rank}: Shape mismatch for theta update, layer {original_layer_idx}. "
                    f"Original: {preact_orig_this_layer.shape}, Gated: {gated_acts_segment.shape}"
                )
            current_total_feature_offset += num_features_this_layer

    @torch.no_grad()
    def estimate_theta_posthoc(
        self,
        encode_all_layers_fn: Callable[
            [Dict[int, torch.Tensor]], Tuple[Dict[int, torch.Tensor], List[Tuple[int, int, int]]]
        ],
        data_iter: torch.utils.data.IterableDataset,
        num_batches: Optional[int] = None,
        default_theta_value: float = 1e6,
        # Removed target_device, as self.device is used. The caller should ensure ThetaManager is on the correct device.
    ) -> torch.Tensor:
        """Estimate theta post-hoc using a specified number of batches."""
        logger.info(f"Rank {self.rank}: Starting post-hoc theta estimation on device {self.device}.")
        # No self.to(target_device) needed here, assumes ThetaManager is already on the correct device.
        # self.eval() is not applicable as ThetaManager itself doesn't have training/eval mode in the same way as a full model.

        if not hasattr(self, "_sum_min_selected_preact") or self._sum_min_selected_preact is None:
            self._sum_min_selected_preact = torch.zeros(
                (self.config.num_layers, self.config.num_features),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self._sum_min_selected_preact.data.zero_()

        if not hasattr(self, "_count_min_selected_preact") or self._count_min_selected_preact is None:
            self._count_min_selected_preact = torch.zeros(
                (self.config.num_layers, self.config.num_features),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self._count_min_selected_preact.data.zero_()

        buffer_shape = (self.config.num_layers, self.config.num_features)
        if not hasattr(self, "_avg_layer_means") or self._avg_layer_means is None:
            self._avg_layer_means = torch.zeros(buffer_shape, dtype=self.dtype, device=self.device)
        else:
            self._avg_layer_means.data.zero_()

        if not hasattr(self, "_avg_layer_stds") or self._avg_layer_stds is None:
            self._avg_layer_stds = torch.zeros(buffer_shape, dtype=self.dtype, device=self.device)
        else:
            self._avg_layer_stds.data.zero_()

        if not hasattr(self, "_processed_batches_for_stats") or self._processed_batches_for_stats is None:
            self._processed_batches_for_stats = torch.zeros(
                self.config.num_layers, dtype=torch.long, device=self.device
            )
        else:
            self._processed_batches_for_stats.data.zero_()

        processed_batches_total = 0
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterable_data_iter = (
                tqdm(data_iter, total=num_batches, desc=f"Estimating Theta & Stats (Rank {self.rank})")
                if num_batches
                else tqdm(data_iter, desc=f"Estimating Theta & Stats (Rank {self.rank})")
            )
        except ImportError:
            logger.info("tqdm not found, proceeding without progress bar for theta estimation.")
            iterable_data_iter = data_iter

        for inputs_batch, _ in iterable_data_iter:
            if num_batches is not None and processed_batches_total >= num_batches:
                break

            inputs_on_device = {k: v.to(device=self.device, dtype=self.dtype) for k, v in inputs_batch.items()}
            preactivations_dict, _ = encode_all_layers_fn(inputs_on_device)

            if not preactivations_dict:
                logger.warning(f"Rank {self.rank}: No preactivations. Skipping batch {processed_batches_total + 1}.")
                processed_batches_total += 1
                continue

            first_valid_preact = next((p for p in preactivations_dict.values() if p.numel() > 0), None)
            if first_valid_preact is None:
                logger.warning(
                    f"Rank {self.rank}: All preactivations empty. Skipping batch {processed_batches_total + 1}."
                )
                processed_batches_total += 1
                continue

            ordered_preactivations_original_posthoc: List[torch.Tensor] = []
            ordered_preactivations_normalized_posthoc: List[torch.Tensor] = []
            layer_feature_sizes_posthoc: List[Tuple[int, int]] = []
            batch_tokens_dim_posthoc = first_valid_preact.shape[0]

            for layer_idx_loop in range(self.config.num_layers):
                num_feat_for_layer: int
                mean_loop: Optional[torch.Tensor] = None
                std_loop: Optional[torch.Tensor] = None
                preact_norm_loop: Optional[torch.Tensor] = None

                if layer_idx_loop in preactivations_dict:
                    preact_orig_loop = preactivations_dict[layer_idx_loop]
                    num_feat_for_layer = (
                        preact_orig_loop.shape[1] if preact_orig_loop.numel() > 0 else self.config.num_features
                    )

                    if preact_orig_loop.shape[0] != batch_tokens_dim_posthoc and preact_orig_loop.numel() > 0:
                        logger.warning(
                            f"Rank {self.rank} Layer {layer_idx_loop}: Mismatched token dim (expected {batch_tokens_dim_posthoc}, got {preact_orig_loop.shape[0]}). Using zeros."
                        )
                        mean_loop = torch.zeros((1, num_feat_for_layer), device=self.device, dtype=self.dtype)
                        std_loop = torch.ones((1, num_feat_for_layer), device=self.device, dtype=self.dtype)
                        preact_norm_loop = torch.zeros(
                            (batch_tokens_dim_posthoc, num_feat_for_layer), device=self.device, dtype=self.dtype
                        )
                        ordered_preactivations_original_posthoc.append(
                            torch.zeros(
                                (batch_tokens_dim_posthoc, num_feat_for_layer), device=self.device, dtype=self.dtype
                            )
                        )
                        ordered_preactivations_normalized_posthoc.append(preact_norm_loop)
                    elif preact_orig_loop.numel() == 0 and batch_tokens_dim_posthoc > 0:
                        mean_loop = torch.zeros((1, num_feat_for_layer), device=self.device, dtype=self.dtype)
                        std_loop = torch.ones((1, num_feat_for_layer), device=self.device, dtype=self.dtype)
                        preact_norm_loop = torch.zeros(
                            (batch_tokens_dim_posthoc, num_feat_for_layer), device=self.device, dtype=self.dtype
                        )
                        ordered_preactivations_original_posthoc.append(
                            torch.zeros(
                                (batch_tokens_dim_posthoc, num_feat_for_layer), device=self.device, dtype=self.dtype
                            )
                        )
                        ordered_preactivations_normalized_posthoc.append(preact_norm_loop)
                    elif preact_orig_loop.numel() > 0:
                        mean_loop = preact_orig_loop.mean(dim=0, keepdim=True)
                        std_loop = preact_orig_loop.std(dim=0, keepdim=True)
                        preact_norm_loop = (preact_orig_loop - mean_loop) / (std_loop + 1e-6)
                        ordered_preactivations_original_posthoc.append(preact_orig_loop)
                        ordered_preactivations_normalized_posthoc.append(preact_norm_loop)
                        assert (
                            self._avg_layer_means is not None
                            and self._avg_layer_stds is not None
                            and self._processed_batches_for_stats is not None
                        )
                        self._avg_layer_means.data[layer_idx_loop] += mean_loop.squeeze().clone()
                        self._avg_layer_stds.data[layer_idx_loop] += std_loop.squeeze().clone()
                        self._processed_batches_for_stats.data[layer_idx_loop] += 1
                    else:
                        num_feat_for_layer = self.config.num_features
                else:
                    num_feat_for_layer = self.config.num_features
                    if batch_tokens_dim_posthoc > 0:
                        ordered_preactivations_original_posthoc.append(
                            torch.zeros(
                                (batch_tokens_dim_posthoc, num_feat_for_layer), device=self.device, dtype=self.dtype
                            )
                        )
                        ordered_preactivations_normalized_posthoc.append(
                            torch.zeros(
                                (batch_tokens_dim_posthoc, num_feat_for_layer), device=self.device, dtype=self.dtype
                            )
                        )
                layer_feature_sizes_posthoc.append((layer_idx_loop, num_feat_for_layer))

            if not ordered_preactivations_normalized_posthoc or not any(
                t.numel() > 0 for t in ordered_preactivations_normalized_posthoc
            ):
                logger.warning(
                    f"Rank {self.rank}: No normalized preactivations. Skipping batch {processed_batches_total + 1}."
                )
                processed_batches_total += 1
                continue

            if not ordered_preactivations_original_posthoc or not any(
                t.numel() > 0 for t in ordered_preactivations_original_posthoc
            ):
                concatenated_preactivations_for_gating = torch.cat(ordered_preactivations_normalized_posthoc, dim=1)
                logger.debug(
                    f"Rank {self.rank} Batch {processed_batches_total + 1}: Using normalized preactivations for gating due to empty/all-empty original list."
                )
            else:
                concatenated_preactivations_for_gating = torch.cat(ordered_preactivations_original_posthoc, dim=1)

            concatenated_preactivations_for_ranking = torch.cat(ordered_preactivations_normalized_posthoc, dim=1)

            activated_concatenated_posthoc: Optional[torch.Tensor] = None
            if self.config.activation_fn == "batchtopk":
                k_val_int = (
                    int(self.config.batchtopk_k)
                    if self.config.batchtopk_k is not None
                    else concatenated_preactivations_for_gating.size(1)
                )
                straight_through_btk = self.config.batchtopk_straight_through
                activated_concatenated_posthoc = BatchTopK.apply(
                    concatenated_preactivations_for_gating,
                    float(k_val_int),
                    straight_through_btk,
                    concatenated_preactivations_for_ranking,
                )
            elif self.config.activation_fn == "topk":
                if not hasattr(self.config, "topk_k") or self.config.topk_k is None:
                    logger.error(
                        f"Rank {self.rank}: 'topk_k' not found in config for 'topk' activation during theta estimation. Defaulting to all features for this batch."
                    )
                    k_val_float = float(concatenated_preactivations_for_gating.size(1))
                else:
                    k_val_float = float(self.config.topk_k)

                straight_through_tk = getattr(self.config, "topk_straight_through", True)
                activated_concatenated_posthoc = TokenTopK.apply(
                    concatenated_preactivations_for_gating,
                    k_val_float,
                    straight_through_tk,
                    concatenated_preactivations_for_ranking,
                )
            else:
                logger.error(
                    f"Rank {self.rank}: Unsupported activation_fn '{self.config.activation_fn}' for theta estimation. Cannot determine gating mechanism. Using zeros for activated_concatenated_posthoc."
                )
                activated_concatenated_posthoc = torch.zeros_like(concatenated_preactivations_for_gating)

            if activated_concatenated_posthoc is not None:
                self._update_min_selected_preactivations(
                    concatenated_preactivations_for_ranking,
                    activated_concatenated_posthoc,
                    layer_feature_sizes_posthoc,
                )
            processed_batches_total += 1

        logger.info(
            f"Rank {self.rank}: Processed {processed_batches_total} batches for theta estimation and stats accumulation."
        )
        assert (
            self._processed_batches_for_stats is not None
            and self._avg_layer_means is not None
            and self._avg_layer_stds is not None
        )
        if (
            self._processed_batches_for_stats is not None
            and self._avg_layer_means is not None
            and self._avg_layer_stds is not None
        ):
            active_stat_batches = self._processed_batches_for_stats.data.unsqueeze(-1).clamp_min(1.0)
            self._avg_layer_means.data /= active_stat_batches
            self._avg_layer_stds.data /= active_stat_batches
            logger.info(f"Rank {self.rank}: Averaged layer-wise normalization stats computed.")
        else:
            logger.warning(f"Rank {self.rank}: Could not finalize normalization stats, buffers missing.")

        self.convert_to_jumprelu_inplace(default_theta_value=default_theta_value)

        # Buffers are part of the module, no need to delete them here unless they are truly temporary and not nn.Buffer
        # Since they are registered buffers, they persist with the module unless explicitly deleted.

        logger.info(f"Rank {self.rank}: Post-hoc theta estimation and conversion to JumpReLU complete.")
        if self.log_threshold is not None and hasattr(self.log_threshold, "data"):
            return torch.exp(self.log_threshold.data)
        else:
            logger.warning(
                f"Rank {self.rank}: log_threshold not available for returning estimated theta. Returning empty tensor."
            )
            return torch.empty(0, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def convert_to_jumprelu_inplace(self, default_theta_value: float = 1e6) -> None:
        """
        Converts the model to use JumpReLU activation based on learned BatchTopK/TokenTopK thresholds.
        This method updates the ThetaManager's self.config and self.log_threshold parameter.
        """
        if self.config.activation_fn not in ["batchtopk", "topk"]:
            logger.warning(
                f"Rank {self.rank}: Model original activation_fn was {self.config.activation_fn}, not batchtopk or topk. "
                "Skipping conversion to JumpReLU based on learned thetas."
            )
            if self.config.activation_fn == "relu":
                logger.error(f"Rank {self.rank}: Model is ReLU, cannot convert to JumpReLU via learned thetas.")
            return

        required_buffers = [
            "_sum_min_selected_preact",
            "_count_min_selected_preact",
            "_avg_layer_means",
            "_avg_layer_stds",
        ]
        for buf_name in required_buffers:
            if not hasattr(self, buf_name) or getattr(self, buf_name) is None:
                raise RuntimeError(
                    f"Rank {self.rank}: Required buffer {buf_name} for JumpReLU conversion not found or not populated. "
                    "Run estimate_theta_posthoc() with appropriate settings before converting."
                )
        assert self._sum_min_selected_preact is not None and self._count_min_selected_preact is not None
        assert self._avg_layer_means is not None and self._avg_layer_stds is not None

        logger.info(
            f"Rank {self.rank}: Starting conversion of {self.config.activation_fn} model to JumpReLU (per-layer avg norm. theta, then unnormalize)."
        )

        theta_sum_norm = self._sum_min_selected_preact.clone()
        theta_cnt_norm = self._count_min_selected_preact.clone()
        avg_mus = self._avg_layer_means.clone()
        avg_sigmas = self._avg_layer_stds.clone()

        if self.process_group is not None and dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(theta_sum_norm, op=dist.ReduceOp.SUM, group=self.process_group)
            dist.all_reduce(theta_cnt_norm, op=dist.ReduceOp.SUM, group=self.process_group)
            dist.all_reduce(avg_mus, op=dist.ReduceOp.SUM, group=self.process_group)
            avg_mus /= self.world_size
            dist.all_reduce(avg_sigmas, op=dist.ReduceOp.SUM, group=self.process_group)
            avg_sigmas /= self.world_size
            logger.info(f"Rank {self.rank}: AllReduced and averaged mu/sigma for unnormalization across ranks.")

        theta_raw = torch.zeros_like(theta_sum_norm)
        fallback_norm_theta_value = 1e-5

        for l_idx in range(self.config.num_layers):
            layer_theta_sum_norm = theta_sum_norm[l_idx]
            layer_theta_cnt_norm = theta_cnt_norm[l_idx]
            active_mask_layer = layer_theta_cnt_norm > 0
            per_feature_thetas_norm_layer = torch.full_like(layer_theta_sum_norm, float("inf"))

            if active_mask_layer.any():
                per_feature_thetas_norm_layer[active_mask_layer] = layer_theta_sum_norm[
                    active_mask_layer
                ] / layer_theta_cnt_norm[active_mask_layer].clamp_min(1.0)

            finite_positive_thetas_norm_layer = per_feature_thetas_norm_layer[
                torch.isfinite(per_feature_thetas_norm_layer) & (per_feature_thetas_norm_layer > 0)
            ]

            theta_norm_scalar_for_this_layer: float
            if finite_positive_thetas_norm_layer.numel() > 0:
                theta_norm_scalar_for_this_layer = finite_positive_thetas_norm_layer.mean().item()
                logger.info(
                    f"Rank {self.rank} Layer {l_idx}: Derived normalized theta (scalar, mean of positive active features) = {theta_norm_scalar_for_this_layer:.4e}"
                )
            else:
                theta_norm_scalar_for_this_layer = fallback_norm_theta_value
                logger.warning(
                    f"Rank {self.rank} Layer {l_idx}: No positive, finite per-feature normalized thetas. Using fallback normalized theta = {theta_norm_scalar_for_this_layer:.4e}"
                )

            mu_vec_layer = avg_mus[l_idx]
            sigma_vec_layer = avg_sigmas[l_idx].clamp_min(1e-6)
            theta_raw_vec_for_layer = theta_norm_scalar_for_this_layer * sigma_vec_layer + mu_vec_layer
            theta_raw[l_idx] = theta_raw_vec_for_layer

            if self.rank == 0 and l_idx < 5:
                logger.info(
                    f"Rank 0 Layer {l_idx}: Normalized Theta_scalar={theta_norm_scalar_for_this_layer:.3e}. Mu (sample): {mu_vec_layer[:3].tolist()}. Sigma (sample): {sigma_vec_layer[:3].tolist()}. Raw Theta (sample): {theta_raw_vec_for_layer[:3].tolist()}"
                )
        logger.info(f"Rank {self.rank}: Per-feature raw thresholds computed via unnormalization.")

        num_norm_feat_no_stats = (theta_cnt_norm == 0).sum().item()
        logger.info(
            f"Rank {self.rank}: Number of features that had no BatchTopK/TokenTopK stats (norm counts==0) across all layers: {num_norm_feat_no_stats}"
        )
        if self.rank == 0:
            logger.info(f"Rank {self.rank}: Final RAW Theta stats (per-feature, shape {theta_raw.shape}):")
            for l_idx in range(self.config.num_layers):
                layer_raw_thetas = theta_raw[l_idx]
                logger.info(
                    f"  Layer {l_idx}: min={layer_raw_thetas.min().item():.4e}, mean={layer_raw_thetas.mean().item():.4e}, max={layer_raw_thetas.max().item():.4e}"
                )
            try:
                import wandb

                if wandb.run:
                    for l_idx in range(self.config.num_layers):
                        layer_raw_thetas_for_hist = theta_raw[l_idx].cpu().float()
                        finite_layer_raw_thetas = layer_raw_thetas_for_hist[
                            torch.isfinite(layer_raw_thetas_for_hist) & (layer_raw_thetas_for_hist > 0)
                        ]
                        if finite_layer_raw_thetas.numel() > 0:
                            wandb.log(
                                {
                                    f"debug/theta_layer_{l_idx}_raw_dist_log10": wandb.Histogram(
                                        torch.log10(finite_layer_raw_thetas).tolist()
                                    )
                                },
                                commit=False,
                            )
                        else:
                            logger.debug(
                                f"Rank {self.rank}: Layer {l_idx} had no finite positive raw thetas for histogram."
                            )
                    all_raw_thetas_flat = theta_raw.flatten().cpu().float()
                    finite_all_raw_thetas = all_raw_thetas_flat[
                        torch.isfinite(all_raw_thetas_flat) & (all_raw_thetas_flat > 0)
                    ]
                    if finite_all_raw_thetas.numel() > 0:
                        wandb.log(
                            {
                                "debug/theta_raw_overall_min_log10": torch.log10(finite_all_raw_thetas.min()).item(),
                                "debug/theta_raw_overall_max_log10": torch.log10(finite_all_raw_thetas.max()).item(),
                                "debug/theta_raw_overall_mean_log10": torch.log10(finite_all_raw_thetas.mean()).item(),
                            },
                            commit=False,
                        )
            except ImportError:
                logger.info("WandB not installed, skipping raw theta distribution logging.")
            except (RuntimeError, ValueError) as e:
                logger.error(f"Rank {self.rank}: Error logging raw theta distributions to WandB: {e}")

        min_final_raw_theta = 1e-7
        num_clamped_final = (theta_raw < min_final_raw_theta).sum().item()
        if num_clamped_final > 0:
            logger.warning(
                f"Rank {self.rank}: Clamping {num_clamped_final} final raw theta values below {min_final_raw_theta} to {min_final_raw_theta} before taking log."
            )
            theta_raw.clamp_min_(min_final_raw_theta)

        log_theta_data = torch.log(theta_raw)  # Changed name to avoid conflict with self.log_threshold

        original_activation_fn = self.config.activation_fn
        self.config.activation_fn = "jumprelu"
        self.config.jumprelu_threshold = 0.0  # Mark as effectively superseded

        if original_activation_fn == "batchtopk":
            self.config.batchtopk_k = None
            self.config.batchtopk_straight_through = False
        elif original_activation_fn == "topk":
            if hasattr(self.config, "topk_k"):
                del self.config.topk_k
            if hasattr(self.config, "topk_straight_through"):
                del self.config.topk_straight_through

        if not hasattr(self, "log_threshold") or self.log_threshold is None:
            self.log_threshold = nn.Parameter(log_theta_data.to(device=self.device, dtype=self.dtype))
        else:
            if not isinstance(self.log_threshold, nn.Parameter):
                self.log_threshold = nn.Parameter(
                    log_theta_data.to(device=self.log_threshold.device, dtype=self.log_threshold.dtype)
                )
            else:
                self.log_threshold.data = log_theta_data.to(
                    device=self.log_threshold.device, dtype=self.log_threshold.dtype
                )

        mark_replicated(self.log_threshold)

        logger.info(f"Rank {self.rank}: Model converted to JumpReLU. activation_fn='{self.config.activation_fn}'.")
        if self.rank == 0:
            min_log_thresh = (
                self.log_threshold.data.min().item()
                if self.log_threshold is not None
                and hasattr(self.log_threshold, "data")
                and self.log_threshold.data.numel() > 0
                else float("nan")
            )
            max_log_thresh = (
                self.log_threshold.data.max().item()
                if self.log_threshold is not None
                and hasattr(self.log_threshold, "data")
                and self.log_threshold.data.numel() > 0
                else float("nan")
            )
            mean_log_thresh = (
                self.log_threshold.data.mean().item()
                if self.log_threshold is not None
                and hasattr(self.log_threshold, "data")
                and self.log_threshold.data.numel() > 0
                else float("nan")
            )
            logger.info(
                f"Rank {self.rank}: Final log_threshold stats: min={min_log_thresh:.4f}, max={max_log_thresh:.4f}, mean={mean_log_thresh:.4f}"
            )
