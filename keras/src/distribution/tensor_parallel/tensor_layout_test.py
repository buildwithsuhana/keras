import keras
import collections
import numpy as np

# Import the classes from your file
# Note: LayoutAction is removed, and LayoutMap is a namedtuple
from keras.src.distribution.tensor_parallel.tensor_layout import Split, LayoutMap
from keras.src import testing


class LayoutTest(testing.TestCase):
    """Test suite for tensor layout actions and mappings."""

    # --- Split Action Tests ---

    def test_split_with_even_division(self):
        """Tests splitting a tensor that divides evenly among workers."""
        device_count = 4
        # Create a tensor of shape (8, 2)
        tensor = keras.ops.reshape(keras.ops.arange(16, dtype="float32"), (8, 2))
        action = Split(device_count=device_count, dim=0)

        # Expected shard for rank 0 has shape (2, 2)
        expected_shard_0 = keras.ops.array([[0.0, 1.0], [2.0, 3.0]])
        # Expected shard for rank 2 has shape (2, 2)
        expected_shard_2 = keras.ops.array([[8.0, 9.0], [10.0, 11.0]])

        shard_0 = action(tensor, rank=0)
        shard_2 = action(tensor, rank=2)

        self.assertAllClose(shard_0, expected_shard_0)
        self.assertAllClose(shard_2, expected_shard_2)
        self.assertEqual(shard_0.shape, (2, 2))

    def test_split_with_uneven_division(self):
        """Tests splitting a tensor where the remainder is distributed correctly."""
        device_count = 3
        # Create a tensor of shape (10, 1). 10 / 3 = 3 with remainder 1.
        tensor = keras.ops.reshape(keras.ops.arange(10, dtype="float32"), (10, 1))
        action = Split(device_count=device_count, dim=0)

        # Rank 0 should get 3 + 1 = 4 rows (remainder goes to the first 'remainder' ranks).
        shard_0 = action(tensor, rank=0)
        self.assertEqual(shard_0.shape, (4, 1))
        self.assertAllClose(shard_0, keras.ops.array([[0.0], [1.0], [2.0], [3.0]]))

        # Rank 1 should get 3 rows.
        shard_1 = action(tensor, rank=1)
        self.assertEqual(shard_1.shape, (3, 1))
        self.assertAllClose(shard_1, keras.ops.array([[4.0], [5.0], [6.0]]))

        # Rank 2 should get 3 rows.
        shard_2 = action(tensor, rank=2)
        self.assertEqual(shard_2.shape, (3, 1))
        self.assertAllClose(shard_2, keras.ops.array([[7.0], [8.0], [9.0]]))

    def test_split_and_undo_cycle_even_removed(self):
        """
        Confirms that the original tensor can be reconstructed (conceptually).
        NOTE: Since 'undo' is removed, this test reconstructs manually.
        """
        device_count = 2
        original_tensor = keras.ops.reshape(keras.ops.arange(12, dtype="float32"), (6, 2))
        action = Split(device_count=device_count, dim=0)

        # Manually perform the split
        shards = [action(original_tensor, rank=i) for i in range(device_count)]
        
        # Manually reconstruct the tensor (equivalent to keras.ops.concatenate)
        reconstructed_tensor = keras.ops.concatenate(shards, axis=action.dim)

        self.assertAllClose(original_tensor, reconstructed_tensor)

    def test_split_and_undo_cycle_uneven_removed(self):
        """
        Confirms that the original tensor can be reconstructed even with uneven split.
        NOTE: Since 'undo' is removed, this test reconstructs manually.
        """
        device_count = 4
        # 11 / 4 = 2 with a remainder of 3.
        original_tensor = keras.ops.reshape(keras.ops.arange(22, dtype="float32"), (11, 2))
        action = Split(device_count=device_count, dim=0)

        shards = [action(original_tensor, rank=i) for i in range(device_count)]
        
        # Verify shard shapes
        self.assertEqual(shards[0].shape, (3, 2)) # 2 + 1
        self.assertEqual(shards[1].shape, (3, 2)) # 2 + 1
        self.assertEqual(shards[2].shape, (3, 2)) # 2 + 1
        self.assertEqual(shards[3].shape, (2, 2)) # 2

        # Manually reconstruct the tensor (equivalent to keras.ops.concatenate)
        reconstructed_tensor = keras.ops.concatenate(shards, axis=action.dim)
        self.assertAllClose(original_tensor, reconstructed_tensor)

    def test_split_last_dimension(self):
        """Tests splitting on the last dimension using dim=-1."""
        device_count = 3
        original_tensor = keras.ops.reshape(keras.ops.arange(30, dtype="float32"), (2, 5, 3))
        action = Split(device_count=device_count, dim=-1)

        shards = [action(original_tensor, rank=i) for i in range(device_count)]
        
        # Each shard should have the last dimension split, with size 1 (3/3=1).
        self.assertEqual(shards[0].shape, (2, 5, 1))
        self.assertEqual(shards[1].shape, (2, 5, 1))
        self.assertEqual(shards[2].shape, (2, 5, 1))

    def test_split_with_sharding_type_hint(self):
        """Tests using 'row' and 'column' sharding hints for 2D tensors."""
        device_count = 2
        tensor = keras.ops.reshape(keras.ops.arange(16, dtype="float32"), (4, 4))

        # **Row sharding** sets dim=0
        action_row = Split(device_count=device_count, dim=-1, sharding_type="row")
        shard_row_0 = action_row(tensor, rank=0)
        self.assertAllClose(shard_row_0, tensor[:2, :])
        self.assertEqual(action_row.dim, 0) # Check if hint correctly set the dim

        # **Column sharding** sets dim=1
        action_col = Split(device_count=device_count, dim=-1, sharding_type="column")
        shard_col_0 = action_col(tensor, rank=0)
        self.assertAllClose(shard_col_0, tensor[:, :2])
        self.assertEqual(action_col.dim, 1) # Check if hint correctly set the dim
    
    # --- LayoutMap Tests (using namedtuple) ---

    def test_layout_map_namedtuple_behavior(self):
        """Tests basic behavior of the LayoutMap namedtuple."""
        state_rules = {"kernel": Split(device_count=2, dim=0)}
        output_rules = {"output": Split(device_count=2, dim=-1)}

        layout_map = LayoutMap(state_rules=state_rules, output_rules=output_rules)

        # 1. Access via attribute name
        self.assertIs(layout_map.state_rules, state_rules)
        self.assertIs(layout_map.output_rules, output_rules)

        # 2. Access via index
        self.assertIs(layout_map[0], state_rules)
        self.assertIs(layout_map[1], output_rules)
        
        # 3. Immutability (namedtuple behavior)
        with self.assertRaises(AttributeError):
            layout_map.state_rules = {}

        # 4. Check contents
        self.assertIsInstance(layout_map.state_rules["kernel"], Split)