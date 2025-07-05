import torch
from typing import Optional, Dict, List, Tuple, Any
import logging
from clt.config import CLTConfig
from torch.distributed import ProcessGroup
from clt.parallel import ops as dist_ops
import math

logger = logging.getLogger(__name__)


def _apply_batch_topk_two_stage(
    preactivations_dict: Dict[int, torch.Tensor],
    config: CLTConfig,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
    process_group: Optional[ProcessGroup],
    profiler: Optional[Any] = None,
) -> Dict[int, torch.Tensor]:
    """Two-stage BatchTopK: per-layer pruning followed by global selection.

    This implementation:
    1. Applies per-layer top-k to reduce candidates (stage 1)
    2. Concatenates pruned activations and applies global top-k (stage 2)
    3. Uses in-place operations where possible to reduce memory usage
    """

    world_size = dist_ops.get_world_size(process_group)

    if not preactivations_dict:
        logger.warning(f"Rank {rank}: _apply_batch_topk_two_stage received empty preactivations_dict.")
        return {}

    # Get batch dimension from first valid tensor
    first_valid_preact = next((p for p in preactivations_dict.values() if p.numel() > 0), None)
    if first_valid_preact is None:
        logger.warning(f"Rank {rank}: No valid preactivations found in dict for BatchTopK. Returning empty dict.")
        return {
            layer_idx: torch.empty((0, config.num_features), device=device, dtype=dtype)
            for layer_idx in preactivations_dict.keys()
        }
    batch_tokens_dim = first_valid_preact.shape[0]

    # --- Determine total k for the *batch* (k_total_batch) ---
    if config.batchtopk_k is not None:
        k_per_token = int(config.batchtopk_k)
        k_total_batch = min(k_per_token * batch_tokens_dim, batch_tokens_dim * config.num_features)
    else:
        # Fall back to keeping all features if k not specified (should not happen)
        k_total_batch = batch_tokens_dim * config.num_features

    # Stage 1: Per-layer pruning target (k1)
    # Heuristic: keep at most 4×(k_total_batch / L) per layer, but at least 64
    num_layers_present = len(
        [layer_idx for layer_idx in preactivations_dict if preactivations_dict[layer_idx].numel() > 0]
    )
    if num_layers_present == 0:
        num_layers_present = 1
    k_per_layer_target = max(64, int((k_total_batch / num_layers_present) * 2))  # keep 2x the share per layer

    pruned_preactivations: List[torch.Tensor] = []
    pruned_indices: List[torch.Tensor] = []
    layer_offsets: List[int] = []
    layer_feature_sizes: List[Tuple[int, int]] = []
    current_offset = 0

    if profiler:
        stage1_timer = profiler.timer("batchtopk_stage1_per_layer")
        stage1_timer.__enter__()

    try:
        for layer_idx in range(config.num_layers):
            if layer_idx not in preactivations_dict:
                continue

            preact_orig = preactivations_dict[layer_idx].to(device=device, dtype=dtype)
            if preact_orig.numel() == 0 or preact_orig.shape[0] != batch_tokens_dim:
                continue

            current_num_features = preact_orig.shape[1]

            # Normalize for ranking (per-layer normalization)
            mean = preact_orig.mean(dim=0, keepdim=True)
            std = preact_orig.std(dim=0, keepdim=True)
            preact_norm = (preact_orig - mean) / (std + 1e-6)

            # Flatten for per-layer top-k
            preact_flat = preact_orig.reshape(-1)
            preact_norm_flat = preact_norm.reshape(-1)

            # Apply per-layer top-k
            k_this_layer = min(k_per_layer_target, preact_flat.numel())
            if k_this_layer > 0:
                # Get top-k values and their indices within this layer
                topk_values, topk_indices_local = torch.topk(preact_norm_flat, k_this_layer, sorted=False)

                # Store pruned values (using original, not normalized)
                pruned_values = preact_flat[topk_indices_local]
                pruned_preactivations.append(pruned_values)

                # Convert local indices to global indices for reconstruction
                global_indices = topk_indices_local + current_offset
                pruned_indices.append(global_indices)

                layer_offsets.append(current_offset)
                layer_feature_sizes.append((layer_idx, current_num_features))
                current_offset += preact_flat.numel()
    finally:
        if profiler and hasattr(stage1_timer, "__exit__"):
            stage1_timer.__exit__(None, None, None)
            if hasattr(stage1_timer, "elapsed"):
                profiler.record("batchtopk_stage1_per_layer", stage1_timer.elapsed)

    if not pruned_preactivations:
        logger.warning(f"Rank {rank}: No tensors collected after stage 1. Returning empty activations.")
        return {
            layer_idx: torch.empty((batch_tokens_dim, config.num_features), device=device, dtype=dtype)
            for layer_idx in preactivations_dict.keys()
        }

    # Concatenate pruned activations
    all_pruned_values = torch.cat(pruned_preactivations)
    all_pruned_indices = torch.cat(pruned_indices)

    # Stage 2: Global top-k selection from pruned candidates
    if profiler:
        stage2_timer = profiler.timer("batchtopk_stage2_global")
        stage2_timer.__enter__()

    try:
        final_k = min(k_total_batch, all_pruned_values.numel())

        if world_size > 1:
            # Distributed: only rank 0 computes, then broadcasts
            if rank == 0:
                # Apply global top-k on pruned values
                _, top_indices_in_pruned = torch.topk(all_pruned_values, final_k, sorted=False)
                final_global_indices = all_pruned_indices[top_indices_in_pruned]
            else:
                final_global_indices = torch.empty(final_k, dtype=torch.long, device=device)

            # Broadcast the selected indices
            if hasattr(profiler, "dist_profiler") and profiler.dist_profiler:
                with profiler.dist_profiler.profile_op("batchtopk_broadcast_indices"):
                    dist_ops.broadcast(final_global_indices, src=0, group=process_group)
            else:
                dist_ops.broadcast(final_global_indices, src=0, group=process_group)
        else:
            # Single GPU: direct computation
            _, top_indices_in_pruned = torch.topk(all_pruned_values, final_k, sorted=False)
            final_global_indices = all_pruned_indices[top_indices_in_pruned]
    finally:
        if profiler and hasattr(stage2_timer, "__exit__"):
            stage2_timer.__exit__(None, None, None)
            if hasattr(stage2_timer, "elapsed"):
                profiler.record("batchtopk_stage2_global", stage2_timer.elapsed)

    # Reconstruct activations with in-place operations
    if profiler:
        reconstruct_timer = profiler.timer("batchtopk_reconstruct")
        reconstruct_timer.__enter__()

    try:
        activations_dict: Dict[int, torch.Tensor] = {}

        # Create a global mask tensor
        total_features = sum(
            preactivations_dict[idx].numel() for idx in preactivations_dict if idx in preactivations_dict
        )
        global_mask = torch.zeros(total_features, dtype=torch.bool, device=device)
        global_mask[final_global_indices] = True

        # Apply mask to each layer with in-place multiplication
        current_position = 0
        for layer_idx in range(config.num_layers):
            if layer_idx not in preactivations_dict:
                continue

            preact = preactivations_dict[layer_idx]
            if preact.numel() == 0:
                continue

            num_elements = preact.numel()
            layer_mask = global_mask[current_position : current_position + num_elements].view_as(preact)

            # In-place multiplication to save memory
            activated = preact.clone()  # Clone to avoid modifying input
            activated.mul_(layer_mask.to(dtype))

            activations_dict[layer_idx] = activated
            current_position += num_elements

    finally:
        if profiler and hasattr(reconstruct_timer, "__exit__"):
            reconstruct_timer.__exit__(None, None, None)
            if hasattr(reconstruct_timer, "elapsed"):
                profiler.record("batchtopk_reconstruct", reconstruct_timer.elapsed)

    return activations_dict


class TwoStageBatchTopK(torch.autograd.Function):
    """Two-stage BatchTopK with optimized forward pass and standard STE backward."""

    @staticmethod
    def forward(
        ctx,
        preactivations_dict: Dict[int, torch.Tensor],
        k: int,
        k_per_layer: int,
        straight_through: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        Apply two-stage BatchTopK selection.

        Args:
            preactivations_dict: Dictionary of pre-activations by layer
            k: Final number of features to keep globally
            k_per_layer: Number of features to keep per layer in stage 1
            straight_through: Whether to use straight-through estimator

        Returns:
            Dictionary of activated tensors by layer
        """
        device = next(iter(preactivations_dict.values())).device
        dtype = next(iter(preactivations_dict.values())).dtype

        # Stage 1: Per-layer selection
        masks_stage1: Dict[int, torch.Tensor] = {}
        pruned_values: List[torch.Tensor] = []
        pruned_indices_map: List[Tuple[int, torch.Tensor]] = []  # (layer_idx, indices)

        for layer_idx, preact in preactivations_dict.items():
            if preact.numel() == 0:
                continue

            # Normalize for ranking
            mean = preact.mean(dim=0, keepdim=True)
            std = preact.std(dim=0, keepdim=True)
            preact_norm = (preact - mean) / (std + 1e-6)

            # Flatten and get top-k per layer
            preact_flat = preact.reshape(-1)
            preact_norm_flat = preact_norm.reshape(-1)

            k_this_layer = min(k_per_layer, preact_flat.numel())
            if k_this_layer > 0:
                _, indices = torch.topk(preact_norm_flat, k_this_layer, sorted=False)

                # Create mask for this layer
                mask = torch.zeros_like(preact_flat, dtype=torch.bool)
                mask[indices] = True
                masks_stage1[layer_idx] = mask.view_as(preact)

                # Store values and indices for stage 2
                pruned_values.append(preact_flat[indices])
                pruned_indices_map.append((layer_idx, indices))

        # Stage 2: Global selection from pruned candidates
        if pruned_values:
            all_pruned = torch.cat(pruned_values)
            final_k = min(k, all_pruned.numel())

            _, global_top_indices = torch.topk(all_pruned, final_k, sorted=False)

            # Create final masks
            final_masks: Dict[int, torch.Tensor] = {}
            current_offset = 0

            for layer_idx, indices in pruned_indices_map:
                layer_size = indices.numel()
                layer_range = torch.arange(current_offset, current_offset + layer_size, device=device)

                # Check which of this layer's candidates made it to final top-k
                selected_in_layer = torch.isin(global_top_indices, layer_range)
                if selected_in_layer.any():
                    # Map back to original positions
                    selected_positions = global_top_indices[selected_in_layer] - current_offset
                    original_indices = indices[selected_positions]

                    # Create final mask for this layer
                    preact_shape = preactivations_dict[layer_idx].shape
                    final_mask = torch.zeros(preact_shape[0] * preact_shape[1], dtype=torch.bool, device=device)
                    final_mask[original_indices] = True
                    final_masks[layer_idx] = final_mask.view(preact_shape)
                else:
                    final_masks[layer_idx] = torch.zeros_like(preactivations_dict[layer_idx], dtype=torch.bool)

                current_offset += layer_size
        else:
            final_masks = {
                idx: torch.zeros_like(preact, dtype=torch.bool) for idx, preact in preactivations_dict.items()
            }

        # Apply masks with in-place operations
        activations: Dict[int, torch.Tensor] = {}
        for layer_idx, preact in preactivations_dict.items():
            if layer_idx in final_masks:
                # Clone to avoid modifying input, then multiply in-place
                activated = preact.clone()
                activated.mul_(final_masks[layer_idx].to(dtype))
                activations[layer_idx] = activated
            else:
                activations[layer_idx] = torch.zeros_like(preact)

        # Save masks for backward
        if straight_through:
            ctx.save_for_backward(*[final_masks.get(i, None) for i in sorted(preactivations_dict.keys())])

        return activations

    @staticmethod
    def backward(ctx, *grad_outputs_dict):
        """Standard straight-through estimator backward pass."""
        if ctx.saved_tensors:
            masks = ctx.saved_tensors

            grad_inputs = {}
            for i, (layer_idx, grad_out) in enumerate(grad_outputs_dict.items()):
                if i < len(masks) and masks[i] is not None:
                    grad_inputs[layer_idx] = grad_out * masks[i].to(grad_out.dtype)
                else:
                    grad_inputs[layer_idx] = torch.zeros_like(grad_out)

            return grad_inputs, None, None, None
        else:
            # No gradients if not using STE
            return ({idx: torch.zeros_like(g) for idx, g in grad_outputs_dict.items()}, None, None, None)


def _apply_token_topk_two_stage(
    preactivations_dict: Dict[int, torch.Tensor],
    config: CLTConfig,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
    process_group: Optional[ProcessGroup],
    profiler: Optional[Any] = None,
) -> Dict[int, torch.Tensor]:
    """Two-stage TokenTopK with vectorized reconstruction for performance."""
    world_size = dist_ops.get_world_size(process_group)

    if not preactivations_dict:
        logger.warning(f"Rank {rank}: _apply_token_topk_two_stage received empty preactivations_dict.")
        return {}

    first_valid_preact = next((p for p in preactivations_dict.values() if p.numel() > 0), None)
    if first_valid_preact is None:
        logger.warning(f"Rank {rank}: No valid preactivations found for TokenTopK. Returning empty dict.")
        return {
            layer_idx: torch.empty((0, config.num_features), device=device, dtype=dtype)
            for layer_idx in preactivations_dict.keys()
        }
    batch_tokens_dim = first_valid_preact.shape[0]

    k_val_float = float(config.topk_k) if config.topk_k is not None else float(config.num_features)

    # --- Smarter oversampling for Stage 1 ---
    num_layers_present = len([p for p in preactivations_dict.values() if p.numel() > 0])
    if num_layers_present > 1:
        # This formula calculates k for stage1 on a per-layer basis to meet a total budget
        total_candidate_budget = (
            k_val_float * math.ceil(math.log2(num_layers_present)) * 1.2
        )  # budget with 20% safety margin
        k_stage1 = max(8, int(total_candidate_budget / num_layers_present))  # distribute budget, with a floor of 8
    else:
        k_stage1 = int(k_val_float * 2.0)  # Fallback for single layer case

    # --- Stage 1: Per-layer token-wise pruning ---
    pruned_values_per_layer: List[torch.Tensor] = []
    pruned_indices_info: List[Tuple[int, torch.Tensor]] = []

    if profiler:
        stage1_timer = profiler.timer("tokentopk_stage1_per_layer")
        stage1_timer.__enter__()

    try:
        for layer_idx in range(config.num_layers):
            if layer_idx not in preactivations_dict or preactivations_dict[layer_idx].numel() == 0:
                continue

            preact_orig = preactivations_dict[layer_idx].to(device=device, dtype=dtype)
            current_num_features = preact_orig.shape[1]

            mean = preact_orig.mean(dim=0, keepdim=True)
            std = preact_orig.std(dim=0, keepdim=True)
            preact_norm = (preact_orig - mean) / (std + 1e-6)

            k_this_layer = min(k_stage1, current_num_features)
            if k_this_layer > 0:
                _, topk_indices = torch.topk(preact_norm, k_this_layer, dim=-1, sorted=False)

                batch_indices = torch.arange(batch_tokens_dim, device=device).unsqueeze(1).expand(-1, k_this_layer)
                topk_values_orig = preact_orig[batch_indices, topk_indices]

                pruned_values_per_layer.append(topk_values_orig)
                pruned_indices_info.append((layer_idx, topk_indices))
    finally:
        if profiler and hasattr(stage1_timer, "__exit__"):
            stage1_timer.__exit__(None, None, None)

    if not pruned_values_per_layer:
        return {idx: torch.zeros_like(p) for idx, p in preactivations_dict.items() if p.numel() > 0}

    concatenated_pruned = torch.cat(pruned_values_per_layer, dim=1)

    # --- Stage 2: Global token-wise top-k on pruned candidates ---
    final_k = int(k_val_float) if k_val_float >= 1 else int(k_val_float * concatenated_pruned.shape[1])
    final_k = min(final_k, concatenated_pruned.shape[1])

    if profiler:
        stage2_timer = profiler.timer("tokentopk_stage2_global")
        stage2_timer.__enter__()

    try:
        if world_size > 1:
            if rank == 0:
                _, global_topk_indices = torch.topk(concatenated_pruned, final_k, dim=-1, sorted=False)
            else:
                global_topk_indices = torch.empty((batch_tokens_dim, final_k), dtype=torch.long, device=device)
            dist_ops.broadcast(global_topk_indices, src=0, group=process_group)
        else:
            _, global_topk_indices = torch.topk(concatenated_pruned, final_k, dim=-1, sorted=False)
    finally:
        if profiler and hasattr(stage2_timer, "__exit__"):
            stage2_timer.__exit__(None, None, None)

    # --- Vectorized Reconstruction ---
    if profiler:
        reconstruct_timer = profiler.timer("tokentopk_reconstruct_vectorized")
        reconstruct_timer.__enter__()
    try:
        total_pruned_features = concatenated_pruned.shape[1]
        original_feature_indices_lookup = torch.empty(
            batch_tokens_dim, total_pruned_features, dtype=torch.long, device=device
        )
        original_layer_indices_lookup = torch.empty(
            batch_tokens_dim, total_pruned_features, dtype=torch.long, device=device
        )

        offset = 0
        for layer_idx, stage1_indices_tensor in pruned_indices_info:
            num_pruned_this_layer = stage1_indices_tensor.shape[1]
            end_offset = offset + num_pruned_this_layer
            original_layer_indices_lookup[:, offset:end_offset] = layer_idx
            original_feature_indices_lookup[:, offset:end_offset] = stage1_indices_tensor
            offset = end_offset

        final_selected_layers = original_layer_indices_lookup.gather(dim=1, index=global_topk_indices)
        final_selected_features = original_feature_indices_lookup.gather(dim=1, index=global_topk_indices)
        final_selected_values = concatenated_pruned.gather(dim=1, index=global_topk_indices)

        final_output_flat = torch.zeros(
            batch_tokens_dim, config.num_layers * config.num_features, device=device, dtype=dtype
        )
        scatter_indices = final_selected_layers * config.num_features + final_selected_features
        final_output_flat.scatter_(dim=1, index=scatter_indices, src=final_selected_values)

        activations_dict = {}
        original_layer_indices_present = {info[0] for info in pruned_indices_info}
        for i in original_layer_indices_present:
            start = i * config.num_features
            end = (i + 1) * config.num_features
            activations_dict[i] = final_output_flat[:, start:end]

        for layer_idx in preactivations_dict:
            if layer_idx not in activations_dict:
                activations_dict[layer_idx] = torch.zeros_like(preactivations_dict[layer_idx])
    finally:
        if profiler and hasattr(reconstruct_timer, "__exit__"):
            reconstruct_timer.__exit__(None, None, None)

    return activations_dict
