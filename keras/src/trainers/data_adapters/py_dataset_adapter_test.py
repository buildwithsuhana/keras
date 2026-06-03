import math
import time
from unittest.mock import patch

import jax
import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.distribution import distribution_lib as dist_lib
from keras.src.testing.test_utils import named_product
from keras.src.trainers.data_adapters import py_dataset_adapter
from keras.src.utils.rng_utils import set_random_seed


class ExamplePyDataset(py_dataset_adapter.PyDataset):
    def __init__(self, x, y, batch_size, workers=1, use_multiprocessing=False):
        super().__init__(
            workers=workers, use_multiprocessing=use_multiprocessing
        )
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __getitem__(self, index):
        return (
            self.x[index * self.batch_size : (index + 1) * self.batch_size],
            self.y[index * self.batch_size : (index + 1) * self.batch_size],
        )

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


class PyDatasetAdapterTest(testing.TestCase):
    def test_basic_flow(self):
        x = np.random.random((16, 4))
        y = np.random.random((16, 2))
        py_dataset = ExamplePyDataset(x, y, batch_size=2)
        adapter = py_dataset_adapter.PyDatasetAdapter(py_dataset)
        self.assertEqual(adapter.num_batches, 8)
        self.assertEqual(adapter.batch_size, 2)
        self.assertEqual(adapter.has_partial_batch, False)

        gen = adapter.get_numpy_iterator()
        for i, batch in enumerate(gen):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertAllClose(bx, x[i * 2 : (i + 1) * 2])
            self.assertAllClose(by, y[i * 2 : (i + 1) * 2])

    def test_multiprocessing_flow(self):
        x = np.random.random((16, 4))
        y = np.random.random((16, 2))
        py_dataset = ExamplePyDataset(
            x, y, batch_size=2, workers=2, use_multiprocessing=True
        )
        adapter = py_dataset_adapter.PyDatasetAdapter(py_dataset)
        gen = adapter.get_numpy_iterator()
        for i, batch in enumerate(gen):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertAllClose(bx, x[i * 2 : (i + 1) * 2])
            self.assertAllClose(by, y[i * 2 : (i + 1) * 2])

    def test_shuffle(self):
        set_random_seed(1337)
        x = np.random.random((16, 4))
        y = np.random.random((16, 2))
        py_dataset = ExamplePyDataset(x, y, batch_size=2)
        adapter = py_dataset_adapter.PyDatasetAdapter(py_dataset, shuffle=True)
        gen = adapter.get_numpy_iterator()
        batches = []
        for batch in gen:
            batches.append(batch)
        self.assertEqual(len(batches), 8)

        # Verify that we got all the data
        all_x = np.concatenate([b[0] for b in batches], axis=0)
        all_y = np.concatenate([b[1] for b in batches], axis=0)
        self.assertNotAllClose(all_x, x)
        self.assertNotAllClose(all_y, y)
        self.assertAllClose(np.sort(all_x, axis=0), np.sort(x, axis=0))

    def test_exceptions(self):
        class BadDataset(ExamplePyDataset):
            def __getitem__(self, index):
                if index == 2:
                    raise ValueError("Intentional error")
                return super().__getitem__(index)

        x = np.random.random((16, 4))
        y = np.random.random((16, 2))
        py_dataset = BadDataset(x, y, batch_size=2)
        adapter = py_dataset_adapter.PyDatasetAdapter(py_dataset)
        gen = adapter.get_numpy_iterator()
        with self.assertRaisesRegex(ValueError, "Intentional error"):
            for _ in gen:
                pass

    def test_multiprocessing_exceptions(self):
        class BadDataset(ExamplePyDataset):
            def __getitem__(self, index):
                if index == 2:
                    raise ValueError("Intentional error")
                return super().__getitem__(index)

        x = np.random.random((16, 4))
        y = np.random.random((16, 2))
        py_dataset = BadDataset(
            x, y, batch_size=2, workers=2, use_multiprocessing=True
        )
        adapter = py_dataset_adapter.PyDatasetAdapter(py_dataset)
        gen = adapter.get_numpy_iterator()
        with self.assertRaisesRegex(ValueError, "Intentional error"):
            for _ in gen:
                pass

    def test_multiprocessing_hang_on_exit(self):
        # Test that we don't hang if the iterator is not fully consumed
        x = np.random.random((100, 4))
        y = np.random.random((100, 2))
        py_dataset = ExamplePyDataset(
            x, y, batch_size=2, workers=2, use_multiprocessing=True
        )
        adapter = py_dataset_adapter.PyDatasetAdapter(py_dataset)
        gen = adapter.get_numpy_iterator()
        next(gen)
        # Explicitly delete the iterator to trigger cleanup
        del gen

    def test_worker_timeout(self):
        class SlowDataset(ExamplePyDataset):
            def __getitem__(self, index):
                time.sleep(0.5)
                return super().__getitem__(index)

        x = np.random.random((4, 4))
        y = np.random.random((4, 2))
        py_dataset = SlowDataset(
            x, y, batch_size=2, workers=1, use_multiprocessing=True
        )
        # No easy way to test the timeout directly, but we can verify it doesn't crash
        adapter = py_dataset_adapter.PyDatasetAdapter(py_dataset)
        gen = adapter.get_numpy_iterator()
        list(gen)

    def test_invalid_arguments(self):
        x = np.random.random((16, 4))
        y = np.random.random((16, 2))
        py_dataset = ExamplePyDataset(x, y, batch_size=2)
        with self.assertRaisesRegex(ValueError, "Expected x to be a PyDataset"):
            py_dataset_adapter.PyDatasetAdapter(x)

    def test_no_len_dataset(self):
        class NoLenDataset(py_dataset_adapter.PyDataset):
            def __getitem__(self, index):
                return np.array([index]), np.array([index])

        py_dataset = NoLenDataset()
        adapter = py_dataset_adapter.PyDatasetAdapter(py_dataset)
        self.assertEqual(adapter.num_batches, None)
        self.assertEqual(adapter.has_partial_batch, None)

        gen = adapter.get_numpy_iterator()
        for index, _ in enumerate(NoLenDataset()):
            if index >= 10:
                break

    @parameterized.named_parameters(
        ("dataparallel", "dp", 4, 1, 4, 1),
        ("modelparallel", "mp", 8, 5, 8, 5),
    )
    @patch("keras.src.distribution.distribution_lib.distribution")
    def test_sharding(
        self,
        dist_type,
        world_size,
        rank,
        expected_num_processes,
        expected_process_id,
        mock_distribution,
    ):
        if backend.backend() not in ("jax"):
            pytest.skip("Distribution support is only available for jax.")
        from keras.src.backend import distribution_lib as backend_dist_lib

        with (
            patch.object(
                backend_dist_lib, "num_processes", return_value=world_size
            ),
            patch.object(backend_dist_lib, "process_id", return_value=rank),
        ):
            if dist_type == "dp":
                dist = dist_lib.DataParallel(devices=["cpu:0"] * world_size)
            else:
                device_mesh = dist_lib.DeviceMesh(
                    shape=(world_size,),
                    axis_names=("data",),
                    devices=["cpu:0"] * world_size,
                )
                dist = dist_lib.ModelParallel(
                    device_mesh=device_mesh,
                    layout_map=dist_lib.LayoutMap(device_mesh),
                    batch_dim_name="data",
                )
            dist.auto_shard_dataset = True
            mock_distribution.return_value = dist

            x = np.random.random((16, 4)).astype("float32")
            y = np.random.random((16, 2)).astype("float32")
            py_dataset = ExamplePyDataset(x, y, batch_size=2)
            adapter = py_dataset_adapter.PyDatasetAdapter(
                py_dataset, shuffle=False
            )

            self.assertEqual(adapter._num_processes, expected_num_processes)
            self.assertEqual(adapter._process_id, expected_process_id)

            it = adapter._get_iterator()
            batches = list(it)
            expected_num_batches = 8 // expected_num_processes
            self.assertEqual(len(batches), expected_num_batches)

    def test_deterministic_shuffle(self):
        x = np.arange(16).reshape((8, 2)).astype("float32")
        y = np.arange(8).reshape((8, 1)).astype("float32")
        py_dataset = ExamplePyDataset(x, y, batch_size=2)

        # Two adapters with same epoch should have same shuffle
        adapter1 = py_dataset_adapter.PyDatasetAdapter(py_dataset, shuffle=True)
        adapter1._epoch = 1
        it1 = adapter1._get_iterator()
        batches1 = list(it1)

        adapter2 = py_dataset_adapter.PyDatasetAdapter(py_dataset, shuffle=True)
        adapter2._epoch = 1
        it2 = adapter2._get_iterator()
        batches2 = list(it2)

        for b1, b2 in zip(batches1, batches2):
            self.assertAllClose(b1[0], b2[0])

        # Different epochs should have different shuffle
        adapter3 = py_dataset_adapter.PyDatasetAdapter(py_dataset, shuffle=True)
        adapter3._epoch = 2
        it3 = adapter3._get_iterator()
        batches3 = list(it3)

        different = False
        for b1, b3 in zip(batches1, batches3):
            if not np.allclose(b1[0], b3[0]):
                different = True
                break
        self.assertTrue(different)

    def test_enqueuer_is_running_and_start(self):
        x = np.random.random((16, 4))
        y = np.random.random((16, 2))
        py_dataset = ExamplePyDataset(x, y, batch_size=2)
        enqueuer = py_dataset_adapter.OrderedEnqueuer(
            py_dataset, workers=2, use_multiprocessing=False
        )
        try:
            self.assertFalse(enqueuer.is_running())
            enqueuer.start()
            self.assertTrue(enqueuer.is_running())
            enqueuer.start()
            self.assertTrue(enqueuer.is_running())
        finally:
            enqueuer.stop()
        self.assertFalse(enqueuer.is_running())
