import os
from unittest.mock import patch

# Mock torch.distributed.init_process_group
with patch("torch.distributed.init_process_group") as mock_init:
    with patch("torch.distributed.is_initialized", return_value=False):
        from keras.src.backend.torch.distribution_lib import initialize

        print("--- Testing initialize with explicit arguments ---")
        initialize(num_processes=4, process_id=2)
        mock_init.assert_called_once()
        args, kwargs = mock_init.call_args
        print(
            f"Called with rank={kwargs.get('rank')}, world_size={kwargs.get('world_size')}"
        )
        mock_init.reset_mock()

        print("\n--- Testing initialize with environment variables ---")
        os.environ["WORLD_SIZE"] = "8"
        os.environ["RANK"] = "3"
        initialize()
        mock_init.assert_called_once()
        args, kwargs = mock_init.call_args
        print(
            f"Called with rank={kwargs.get('rank')}, world_size={kwargs.get('world_size')}"
        )
        mock_init.reset_mock()
        del os.environ["WORLD_SIZE"]
        del os.environ["RANK"]

        print(
            "\n--- Testing initialize with no arguments and no environment (should not init) ---"
        )
        initialize()
        mock_init.assert_not_called()
        print("Success: init_process_group not called when world_size=1")
