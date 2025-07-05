import torch
from typing import Optional, Tuple, Dict, List, Any
import logging
from clt.config import CLTConfig
from torch.distributed import ProcessGroup
from clt.parallel import ops as dist_ops


class BatchTopK(torch.autograd.Function):
    @staticmethod
    def _compute_mask(x: torch.Tensor, k_per_token: int, x_for_ranking: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Helper static method to compute the BatchTopK mask."""
        B = x.size(0)
        if k_per_token <= 0:
            return torch.zeros_like(x, dtype=torch.bool)

        F_total_batch = x.numel()
        if F_total_batch == 0:
            return torch.zeros_like(x, dtype=torch.bool)

        k_total_batch = min(k_per_token * B, F_total_batch)

        ranking_tensor_to_use = x_for_ranking if x_for_ranking is not None else x
        x_flat = x.reshape(-1)
        ranking_flat = ranking_tensor_to_use.reshape(-1)

        if k_total_batch > 0:
            _, flat_indices = torch.topk(ranking_flat, k_total_batch, sorted=False)
            # Optimized mask creation - avoid individual indexing
            mask = torch.zeros(x_flat.numel(), dtype=torch.bool, device=x.device)
            mask[flat_indices] = True
            mask = mask.view_as(x)
        else:
            mask = torch.zeros_like(x, dtype=torch.bool)

        return mask

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, k: float, straight_through: bool, x_for_ranking: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Applies BatchTopK activation.

        Args:
            x: Input tensor of shape [B, F_total] or [B*S, F_total].
            k: The number of top elements to keep. If k < 1, it's treated as a fraction.
            straight_through: Whether to use straight-through estimator for gradients.
            x_for_ranking: Optional tensor to use for ranking. If None, x is used.

        Returns:
            Output tensor with BatchTopK applied.
        """
        k_per_token = int(k)
        mask = BatchTopK._compute_mask(x, k_per_token, x_for_ranking)

        if straight_through:
            ctx.save_for_backward(mask)
        else:
            # For non-STE, save original input and mask for more complex gradient (e.g. Gumbel-Softmax style, not implemented here)
            # ctx.save_for_backward(x, mask) # Placeholder for more complex backward
            ctx.save_for_backward(mask)  # For now, non-STE will also behave like STE in backward

        ctx.straight_through = straight_through

        return x * mask.to(x.dtype)

    @staticmethod
    def backward(ctx, *args: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None]:
        """Backward pass for BatchTopK.

        Args:
            grad_output: Gradient from the subsequent layer.

        Returns:
            Gradients for input x, k, straight_through, and x_for_ranking.
        """
        # Expecting only one gradient output tensor
        if len(args) != 1:
            raise ValueError(f"BatchTopK backward expected 1 gradient tensor, got {len(args)}")
        grad_output = args[0]

        if ctx.straight_through:
            (mask,) = ctx.saved_tensors
            grad_input = grad_output * mask.to(grad_output.dtype)
        else:
            # Placeholder for a potentially different backward pass if not using STE
            # For now, treat non-STE as STE in backward to avoid errors.
            # A more sophisticated backward (e.g., for Gumbel-Softmax) would go here.
            (mask,) = ctx.saved_tensors
            grad_input = grad_output * mask.to(grad_output.dtype)
            # Example of a different backward (not used now):
            # x_original, mask = ctx.saved_tensors
            # grad_input = grad_output * (some_function_of_x_original_and_mask)
            # This is just illustrative.

        return grad_input, None, None, None  # Gradients for k, straight_through, x_for_ranking are None


class TokenTopK(torch.autograd.Function):
    @staticmethod
    def _compute_mask(x: torch.Tensor, k_float: float, x_for_ranking: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Helper static method to compute the TokenTopK mask."""
        B_tokens, F_total = x.shape

        if F_total == 0:
            return torch.zeros_like(x, dtype=torch.bool)

        k_per_token: int
        if 0 < k_float < 1:
            k_per_token = int(torch.ceil(torch.tensor(k_float * F_total)).item())
        elif k_float >= 1:
            k_per_token = int(k_float)
        else:  # k <= 0
            return torch.zeros_like(x, dtype=torch.bool)

        k_per_token = min(k_per_token, F_total)

        ranking_tensor_to_use = x_for_ranking if x_for_ranking is not None else x

        if k_per_token > 0:
            _, topk_indices_per_row = torch.topk(ranking_tensor_to_use, k_per_token, dim=-1, sorted=False)
            # Use scatter_ for efficient mask creation
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask.scatter_(-1, topk_indices_per_row, True)
        else:
            mask = torch.zeros_like(x, dtype=torch.bool)

        return mask

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, k: float, straight_through: bool, x_for_ranking: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Applies TokenTopK activation (TopK per token/row).

        Args:
            x: Input tensor of shape [B_tokens, F_total].
            k: The number or fraction of top elements to keep per token.
               If k < 1, it's treated as a fraction of F_total.
               If k >= 1, it's treated as an integer count.
            straight_through: Whether to use straight-through estimator for gradients.
            x_for_ranking: Optional tensor to use for ranking. If None, x is used.

        Returns:
            Output tensor with TokenTopK applied.
        """
        mask = TokenTopK._compute_mask(x, k, x_for_ranking)

        if straight_through:
            ctx.save_for_backward(mask)
        else:
            ctx.save_for_backward(mask)  # For now, non-STE also behaves like STE in backward

        ctx.straight_through = straight_through

        return x * mask.to(x.dtype)

    @staticmethod
    def backward(ctx, *args: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None]:
        """Backward pass for TokenTopK (identical to BatchTopK's STE path)."""
        if len(args) != 1:
            raise ValueError(f"TokenTopK backward expected 1 gradient tensor, got {len(args)}")
        grad_output = args[0]

        if ctx.straight_through:
            (mask,) = ctx.saved_tensors
            grad_input = grad_output * mask.to(grad_output.dtype)
        else:
            (mask,) = ctx.saved_tensors
            grad_input = grad_output * mask.to(grad_output.dtype)

        return grad_input, None, None, None


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(input, threshold)
        ctx.bandwidth = bandwidth
        return (input >= threshold).float() * input

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        grad_output = grad_outputs[0]
        input, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        needs_input_grad, needs_threshold_grad, _ = ctx.needs_input_grad

        grad_input = None
        grad_threshold = None

        if needs_input_grad:
            ste_mask = (input >= threshold).type_as(grad_output)
            grad_input = grad_output * ste_mask

        if needs_threshold_grad:
            is_near_threshold = torch.abs(input - threshold) <= (bandwidth / 2.0)
            local_grad_theta = (-input / bandwidth) * is_near_threshold.type_as(input)
            grad_threshold_per_element = grad_output * local_grad_theta

            if grad_threshold_per_element.dim() > threshold.dim():
                # Handles cases like input (B,F), threshold (F) or input (F), threshold (scalar)
                dims_to_sum = tuple(range(grad_threshold_per_element.dim() - threshold.dim()))
                grad_threshold = grad_threshold_per_element.sum(dim=dims_to_sum)
                # Ensure final shape matches threshold, especially if sum squeezed dimensions
                if grad_threshold.shape != threshold.shape:
                    grad_threshold = grad_threshold.reshape(threshold.shape)
            elif grad_threshold_per_element.dim() == threshold.dim():
                # Handles cases like input (F), threshold (F), or input [1], threshold [1]
                grad_threshold = grad_threshold_per_element
                # Defensive reshape, though shapes should ideally match here.
                if grad_threshold.shape != threshold.shape:
                    grad_threshold = grad_threshold.reshape(threshold.shape)
            else:  # grad_threshold_per_element.dim() < threshold.dim()
                # This case is less common (e.g. input scalar, threshold vector - not typical for this op).
                # Defaulting to sum and reshape, primarily for scalar threshold case.
                grad_threshold = grad_threshold_per_element.sum()
                if grad_threshold.shape != threshold.shape:
                    grad_threshold = grad_threshold.reshape(threshold.shape)
        return grad_input, grad_threshold, None


# --- Helper functions for applying BatchTopK/TokenTopK globally --- #
# These were previously in clt.models.encoding.py

logger_helpers = logging.getLogger(__name__ + ".helpers")  # Use a sub-logger


def _apply_batch_topk_helper(
    preactivations_dict: Dict[int, torch.Tensor],
    config: CLTConfig,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
    process_group: Optional[ProcessGroup],
    profiler: Optional[Any] = None,
) -> Dict[int, torch.Tensor]:
    """Helper to apply BatchTopK globally across concatenated layer pre-activations."""

    world_size = dist_ops.get_world_size(process_group)

    if not preactivations_dict:
        logger_helpers.warning(f"Rank {rank}: _apply_batch_topk_helper received empty preactivations_dict.")
        return {}

    ordered_preactivations_original: List[torch.Tensor] = []
    ordered_preactivations_normalized: List[torch.Tensor] = []
    layer_feature_sizes: List[Tuple[int, int]] = []

    first_valid_preact = next((p for p in preactivations_dict.values() if p.numel() > 0), None)
    if first_valid_preact is None:
        logger_helpers.warning(
            f"Rank {rank}: No valid preactivations found in dict for BatchTopK. Returning empty dict."
        )
        return {
            layer_idx: torch.empty((0, config.num_features), device=device, dtype=dtype)
            for layer_idx in preactivations_dict.keys()
        }
    batch_tokens_dim = first_valid_preact.shape[0]

    for layer_idx in range(config.num_layers):
        if layer_idx in preactivations_dict:
            preact_orig = preactivations_dict[layer_idx]
            preact_orig = preact_orig.to(device=device, dtype=dtype)
            current_num_features = preact_orig.shape[1] if preact_orig.numel() > 0 else config.num_features

            if preact_orig.numel() == 0:
                if batch_tokens_dim > 0:
                    zeros_shape = (batch_tokens_dim, current_num_features)
                    ordered_preactivations_original.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
                    ordered_preactivations_normalized.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
            elif preact_orig.shape[0] != batch_tokens_dim:
                logger_helpers.warning(
                    f"Rank {rank} Layer {layer_idx}: Mismatched batch dim ({preact_orig.shape[0]} vs {batch_tokens_dim}). Using zeros."
                )
                zeros_shape = (batch_tokens_dim, current_num_features)
                ordered_preactivations_original.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
                ordered_preactivations_normalized.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
            else:
                ordered_preactivations_original.append(preact_orig)
                mean = preact_orig.mean(dim=0, keepdim=True)
                std = preact_orig.std(dim=0, keepdim=True)
                preact_norm = (preact_orig - mean) / (std + 1e-6)
                ordered_preactivations_normalized.append(preact_norm)
            layer_feature_sizes.append((layer_idx, current_num_features))

    if not ordered_preactivations_original:
        logger_helpers.warning(
            f"Rank {rank}: No tensors collected after iterating layers for BatchTopK. Returning empty activations."
        )
        return {
            layer_idx: torch.empty((batch_tokens_dim, config.num_features), device=device, dtype=dtype)
            for layer_idx in preactivations_dict.keys()
        }

    concatenated_preactivations_original = torch.cat(ordered_preactivations_original, dim=1)
    concatenated_preactivations_normalized = torch.cat(ordered_preactivations_normalized, dim=1)

    k_val: int
    if config.batchtopk_k is not None:
        k_val = int(config.batchtopk_k)
    else:
        k_val = concatenated_preactivations_original.size(1)

    mask_shape = concatenated_preactivations_original.shape
    mask = torch.empty(mask_shape, dtype=torch.bool, device=device)

    if world_size > 1:
        if rank == 0:
            if profiler:
                with profiler.timer("batchtopk_compute_mask") as timer:
                    local_mask = BatchTopK._compute_mask(
                        concatenated_preactivations_original, k_val, concatenated_preactivations_normalized
                    )
                if hasattr(timer, "elapsed"):
                    profiler.record("batchtopk_compute_mask", timer.elapsed)
            else:
                local_mask = BatchTopK._compute_mask(
                    concatenated_preactivations_original, k_val, concatenated_preactivations_normalized
                )
            mask.copy_(local_mask)

            if hasattr(profiler, "dist_profiler") and profiler.dist_profiler:
                with profiler.dist_profiler.profile_op("batchtopk_broadcast"):
                    dist_ops.broadcast(mask, src=0, group=process_group)
            else:
                dist_ops.broadcast(mask, src=0, group=process_group)
        else:
            if hasattr(profiler, "dist_profiler") and profiler.dist_profiler:
                with profiler.dist_profiler.profile_op("batchtopk_broadcast"):
                    dist_ops.broadcast(mask, src=0, group=process_group)
            else:
                dist_ops.broadcast(mask, src=0, group=process_group)
    else:
        if profiler:
            with profiler.timer("batchtopk_compute_mask") as timer:
                mask = BatchTopK._compute_mask(
                    concatenated_preactivations_original, k_val, concatenated_preactivations_normalized
                )
            if hasattr(timer, "elapsed"):
                profiler.record("batchtopk_compute_mask", timer.elapsed)
        else:
            mask = BatchTopK._compute_mask(
                concatenated_preactivations_original, k_val, concatenated_preactivations_normalized
            )

    activated_concatenated = concatenated_preactivations_original * mask.to(dtype)

    activations_dict: Dict[int, torch.Tensor] = {}
    current_total_feature_offset = 0
    for original_layer_idx, num_features_this_layer in layer_feature_sizes:
        activated_segment = activated_concatenated[
            :, current_total_feature_offset : current_total_feature_offset + num_features_this_layer
        ]
        activations_dict[original_layer_idx] = activated_segment
        current_total_feature_offset += num_features_this_layer
    return activations_dict


def _apply_token_topk_helper(
    preactivations_dict: Dict[int, torch.Tensor],
    config: CLTConfig,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
    process_group: Optional[ProcessGroup],
    profiler: Optional[Any] = None,
) -> Dict[int, torch.Tensor]:
    """Helper to apply TokenTopK globally across concatenated layer pre-activations."""
    world_size = dist_ops.get_world_size(process_group)

    if not preactivations_dict:
        logger_helpers.warning(f"Rank {rank}: _apply_token_topk_helper received empty preactivations_dict.")
        return {}

    ordered_preactivations_original: List[torch.Tensor] = []
    ordered_preactivations_normalized: List[torch.Tensor] = []
    layer_feature_sizes: List[Tuple[int, int]] = []

    first_valid_preact = next((p for p in preactivations_dict.values() if p.numel() > 0), None)
    if first_valid_preact is None:
        logger_helpers.warning(
            f"Rank {rank}: No valid preactivations found in dict for TokenTopK. Returning empty dict."
        )
        return {
            layer_idx: torch.empty((0, config.num_features), device=device, dtype=dtype)
            for layer_idx in preactivations_dict.keys()
        }
    batch_tokens_dim = first_valid_preact.shape[0]

    for layer_idx in range(config.num_layers):
        if layer_idx in preactivations_dict:
            preact_orig = preactivations_dict[layer_idx]
            preact_orig = preact_orig.to(device=device, dtype=dtype)
            current_num_features = preact_orig.shape[1] if preact_orig.numel() > 0 else config.num_features

            if preact_orig.numel() == 0:
                if batch_tokens_dim > 0:
                    zeros_shape = (batch_tokens_dim, current_num_features)
                    ordered_preactivations_original.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
                    ordered_preactivations_normalized.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
            elif preact_orig.shape[0] != batch_tokens_dim:
                logger_helpers.warning(
                    f"Rank {rank} Layer {layer_idx}: Mismatched batch dim ({preact_orig.shape[0]} vs {batch_tokens_dim}) for TokenTopK. Using zeros."
                )
                zeros_shape = (batch_tokens_dim, current_num_features)
                ordered_preactivations_original.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
                ordered_preactivations_normalized.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
            else:
                ordered_preactivations_original.append(preact_orig)
                mean = preact_orig.mean(dim=0, keepdim=True)
                std = preact_orig.std(dim=0, keepdim=True)
                preact_norm = (preact_orig - mean) / (std + 1e-6)
                ordered_preactivations_normalized.append(preact_norm)
            layer_feature_sizes.append((layer_idx, current_num_features))

    if not ordered_preactivations_original:
        logger_helpers.warning(
            f"Rank {rank}: No tensors collected after iterating layers for TokenTopK. Returning empty activations."
        )
        return {
            layer_idx: torch.empty((batch_tokens_dim, config.num_features), device=device, dtype=dtype)
            for layer_idx in preactivations_dict.keys()
        }

    concatenated_preactivations_original = torch.cat(ordered_preactivations_original, dim=1)
    concatenated_preactivations_normalized = torch.cat(ordered_preactivations_normalized, dim=1)

    k_val_float: float
    if hasattr(config, "topk_k") and config.topk_k is not None:
        k_val_float = float(config.topk_k)
    else:
        k_val_float = float(concatenated_preactivations_original.size(1))

    mask_shape = concatenated_preactivations_original.shape
    mask = torch.empty(mask_shape, dtype=torch.bool, device=device)

    if world_size > 1:
        if rank == 0:
            if profiler:
                with profiler.timer("topk_compute_mask") as timer:
                    local_mask = TokenTopK._compute_mask(
                        concatenated_preactivations_original,
                        k_val_float,
                        concatenated_preactivations_normalized,
                    )
                if hasattr(timer, "elapsed"):
                    profiler.record("topk_compute_mask", timer.elapsed)
            else:
                local_mask = TokenTopK._compute_mask(
                    concatenated_preactivations_original,
                    k_val_float,
                    concatenated_preactivations_normalized,
                )
            mask.copy_(local_mask)

            if hasattr(profiler, "dist_profiler") and profiler.dist_profiler:
                with profiler.dist_profiler.profile_op("topk_broadcast"):
                    dist_ops.broadcast(mask, src=0, group=process_group)
            else:
                dist_ops.broadcast(mask, src=0, group=process_group)
        else:
            if hasattr(profiler, "dist_profiler") and profiler.dist_profiler:
                with profiler.dist_profiler.profile_op("topk_broadcast"):
                    dist_ops.broadcast(mask, src=0, group=process_group)
            else:
                dist_ops.broadcast(mask, src=0, group=process_group)
    else:
        if profiler:
            with profiler.timer("topk_compute_mask") as timer:
                mask = TokenTopK._compute_mask(
                    concatenated_preactivations_original, k_val_float, concatenated_preactivations_normalized
                )
            if hasattr(timer, "elapsed"):
                profiler.record("topk_compute_mask", timer.elapsed)
        else:
            mask = TokenTopK._compute_mask(
                concatenated_preactivations_original, k_val_float, concatenated_preactivations_normalized
            )

    activated_concatenated = concatenated_preactivations_original * mask.to(dtype)

    activations_dict: Dict[int, torch.Tensor] = {}
    current_total_feature_offset = 0
    for original_layer_idx, num_features_this_layer in layer_feature_sizes:
        activated_segment = activated_concatenated[
            :, current_total_feature_offset : current_total_feature_offset + num_features_this_layer
        ]
        activations_dict[original_layer_idx] = activated_segment
        current_total_feature_offset += num_features_this_layer
    return activations_dict
