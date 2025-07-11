import torch
from pathlib import Path
import numpy as np

from clt.training.data.local_activation_store import LocalActivationStore


class TestLocalActivationStore:
    def test_initialization(self, tmp_local_dataset: Path):
        """Test that the store initializes correctly with a local dataset."""
        store = LocalActivationStore(
            dataset_path=str(tmp_local_dataset),
            train_batch_size_tokens=16,
            rank=0,
            world=1,
        )
        assert store.num_layers == 2
        assert store.d_model == 8
        assert store.total_tokens == 64
        assert len(store.sampler) == 64 // 16

    def test_get_batch_unsharded(self, tmp_local_dataset: Path):
        """Test fetching a single batch from the store without sharding."""
        store = LocalActivationStore(
            dataset_path=str(tmp_local_dataset),
            train_batch_size_tokens=16,
            dtype="float16",  # Match the dtype of the data in the fixture
            rank=0,
            world=1,
        )
        inputs, targets = store.get_batch()

        assert isinstance(inputs, dict)
        assert isinstance(targets, dict)
        assert len(inputs) == store.num_layers
        assert len(targets) == store.num_layers

        for i in range(store.num_layers):
            assert i in inputs
            assert i in targets
            assert inputs[i].shape == (16, store.d_model)
            assert targets[i].shape == (16, store.d_model)
            assert inputs[i].dtype == torch.float16
            assert targets[i].device.type == store.device.type

    def test_iteration_unsharded(self, tmp_local_dataset: Path):
        """Test iterating through the entire dataset without sharding."""
        store = LocalActivationStore(
            dataset_path=str(tmp_local_dataset),
            train_batch_size_tokens=16,
            rank=0,
            world=1,
            dtype="float16",
        )

        num_batches = 0
        total_tokens = 0
        for inputs, targets in store:
            num_batches += 1
            # Get token count from the first layer's input tensor
            batch_tokens = next(iter(inputs.values())).shape[0]
            total_tokens += batch_tokens

        assert num_batches == len(store.sampler)
        assert total_tokens == store.total_tokens

    def test_sharding_iteration(self, tmp_local_dataset: Path):
        """Test that sharding splits the data correctly across ranks."""
        world_size = 2
        total_tokens_processed = 0

        for rank in range(world_size):
            store = LocalActivationStore(
                dataset_path=str(tmp_local_dataset),
                train_batch_size_tokens=16,
                rank=rank,
                world=world_size,
                shard_data=True,
                dtype="float16",
            )

            rank_tokens = 0
            for inputs, _ in store:
                rank_tokens += next(iter(inputs.values())).shape[0]

            # Each rank should process roughly half the tokens
            assert rank_tokens == store.total_tokens // world_size
            total_tokens_processed += rank_tokens

        # The sum of tokens processed by all ranks should equal the total tokens
        assert total_tokens_processed == store.total_tokens

    def test_state_dict_roundtrip(self, tmp_local_dataset: Path):
        """Test that the store can be resumed from a saved state."""
        store1 = LocalActivationStore(dataset_path=str(tmp_local_dataset), train_batch_size_tokens=16, dtype="float16")

        # Get first batch
        batch1_inputs, _ = store1.get_batch()

        # Save state
        state = store1.state_dict()

        # Create a new store and load state
        store2 = LocalActivationStore(dataset_path=str(tmp_local_dataset), train_batch_size_tokens=16, dtype="float16")
        store2.load_state_dict(state)

        # Get next batch from the new store
        batch2_inputs, _ = store2.get_batch()

        # Get the second batch from the original store for comparison
        expected_batch2_inputs, _ = store1.get_batch()

        # The batch from the resumed store should match the next batch from the original
        for i in store1.layer_indices:
            torch.testing.assert_close(batch2_inputs[i], expected_batch2_inputs[i])

    def test_layer_data_integrity(self, tmp_local_dataset: Path):
        """Test that each layer's data is distinct and not mixed up."""
        store = LocalActivationStore(
            dataset_path=str(tmp_local_dataset),
            train_batch_size_tokens=16,
            dtype="float16",
        )

        inputs, targets = store.get_batch()

        # Verify each layer has distinct data
        # The fixture creates random data: inputs in [0, 10), targets in [0, 5)
        for layer_id in range(store.num_layers):
            layer_inputs = inputs[layer_id].cpu().numpy()
            layer_targets = targets[layer_id].cpu().numpy()

            # Basic sanity checks on the data
            assert layer_inputs.shape == (16, store.d_model), f"Layer {layer_id} inputs have wrong shape"
            assert layer_targets.shape == (16, store.d_model), f"Layer {layer_id} targets have wrong shape"

            # Check that values are in reasonable ranges (inputs: [0, 10), targets: [0, 5))
            assert layer_inputs.min() >= 0, f"Layer {layer_id} inputs have negative values"
            assert layer_inputs.max() < 15, f"Layer {layer_id} inputs have unexpectedly large values"
            assert layer_targets.min() >= 0, f"Layer {layer_id} targets have negative values"
            assert layer_targets.max() < 10, f"Layer {layer_id} targets have unexpectedly large values"

            # Verify data is different between layers (statistically very unlikely to be identical)
            if layer_id > 0:
                prev_layer_inputs = inputs[layer_id - 1].cpu().numpy()
                # Check that at least some values differ between layers
                assert not np.array_equal(
                    layer_inputs, prev_layer_inputs
                ), f"Layer {layer_id} data identical to layer {layer_id - 1}"

                # Additional check: mean values should differ (statistically)
                mean_diff = abs(layer_inputs.mean() - prev_layer_inputs.mean())
                assert mean_diff > 0.01, f"Layer {layer_id} and {layer_id - 1} have suspiciously similar means"
