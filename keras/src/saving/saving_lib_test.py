import json
import os
import pathlib
import zipfile
from io import BytesIO
from unittest import mock

import h5py
import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.src import backend
from keras.src import testing
from keras.src.saving import saving_lib


class SavingLibTest(testing.TestCase):
    def test_save_load_model(self):
        model = keras.Sequential(
            [
                keras.layers.Dense(3, input_shape=(4,)),
                keras.layers.BatchNormalization(),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        filepath = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(filepath)
        reloaded_model = keras.models.load_model(filepath)
        self.assertEqual(len(model.weights), len(reloaded_model.weights))
        for w1, w2 in zip(model.weights, reloaded_model.weights):
            self.assertAllClose(w1, w2)

    def test_save_load_weights_only(self):
        model = keras.Sequential([keras.layers.Dense(3, input_shape=(4,))])
        filepath = os.path.join(self.get_temp_dir(), "weights.weights.h5")
        model.save_weights(filepath)
        new_model = keras.Sequential([keras.layers.Dense(3, input_shape=(4,))])
        new_model.load_weights(filepath)
        for w1, w2 in zip(model.weights, new_model.weights):
            self.assertAllClose(w1, w2)

    def test_save_load_model_zipped_false(self):
        model = keras.Sequential([keras.layers.Dense(3, input_shape=(4,))])
        dirpath = os.path.join(self.get_temp_dir(), "model_dir")
        model.save(dirpath, zipped=False)
        reloaded_model = keras.models.load_model(dirpath)
        for w1, w2 in zip(model.weights, reloaded_model.weights):
            self.assertAllClose(w1, w2)

    def test_save_load_sharded_weights(self):
        model = keras.Sequential([keras.layers.Dense(3, input_shape=(4,))])
        filepath = os.path.join(self.get_temp_dir(), "sharded.weights.json")
        model.save_weights(filepath, max_shard_size=0.0001)
        new_model = keras.Sequential([keras.layers.Dense(3, input_shape=(4,))])
        new_model.load_weights(filepath)
        for w1, w2 in zip(model.weights, new_model.weights):
            self.assertAllClose(w1, w2)


class SafeZipReadTest(testing.TestCase):
    def _zip_with_member(self, name, data, compression=zipfile.ZIP_DEFLATED):
        path = os.path.join(self.get_temp_dir(), "a.zip")
        with zipfile.ZipFile(path, "w", compression=compression) as zf:
            zf.writestr(name, data)
        return path

    def test_rejects_decompression_bomb(self):
        # Highly compressible member: large declared size, ~nothing on disk.
        path = self._zip_with_member("config.json", b"A" * 100_000)
        with (
            mock.patch.object(saving_lib, "_ZIP_MEMBER_BOMB_FLOOR_BYTES", 64),
            mock.patch.object(saving_lib, "_ZIP_MEMBER_MAX_EXPANSION", 10),
        ):
            with zipfile.ZipFile(path, "r") as zf:
                with self.assertRaisesRegex(ValueError, "decompression bomb"):
                    saving_lib._safe_zip_read(zf, "config.json")

    def test_allows_incompressible_member(self):
        # Stored (uncompressed) member: declared size == stored size.
        data = os.urandom(100_000)
        path = self._zip_with_member("w", data, compression=zipfile.ZIP_STORED)
        with mock.patch.object(saving_lib, "_ZIP_MEMBER_BOMB_FLOOR_BYTES", 64):
            with zipfile.ZipFile(path, "r") as zf:
                self.assertEqual(saving_lib._safe_zip_read(zf, "w"), data)

    def test_load_model_rejects_bomb_config(self):
        # End-to-end: a tiny `.keras` whose config.json decompresses huge is
        # rejected before the read allocates, with safe_mode=True.
        path = os.path.join(self.get_temp_dir(), "bomb.keras")
        payload = b'{"x":"' + b" " * 200_000 + b'"}'
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", b'{"keras_version":"3"}')
            zf.writestr("config.json", payload)
        self.assertLess(os.path.getsize(path), 1 << 16)  # tiny file on disk
        with (
            mock.patch.object(saving_lib, "_ZIP_MEMBER_BOMB_FLOOR_BYTES", 64),
            mock.patch.object(saving_lib, "_ZIP_MEMBER_MAX_EXPANSION", 10),
        ):
            with self.assertRaisesRegex(ValueError, "decompression bomb"):
                saving_lib.load_model(path)

    def test_load_model_rejects_bomb_weights(self):
        # A bomb `model.weights.h5` member must be rejected up front, before the
        # in-memory / extract-to-disk / on-the-fly read paths (and not swallowed
        # by the on-the-fly fallback's bare `except`).
        import keras

        model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(3)])
        good = os.path.join(self.get_temp_dir(), "good.keras")
        model.save(good)
        with zipfile.ZipFile(good) as zf:
            cfg = zf.read("config.json")
            meta = zf.read("metadata.json")

        evil = os.path.join(self.get_temp_dir(), "evil.keras")
        with zipfile.ZipFile(
            evil, "w"
        ) as zf:  # config/metadata stored (ratio 1)
            zf.writestr("metadata.json", meta)
            zf.writestr("config.json", cfg)
            info = zipfile.ZipInfo("model.weights.h5")
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, b"\x00" * 200_000)  # deflated bomb member

        with (
            mock.patch.object(saving_lib, "_ZIP_MEMBER_BOMB_FLOOR_BYTES", 64),
            mock.patch.object(saving_lib, "_ZIP_MEMBER_MAX_EXPANSION", 10),
        ):
            with self.assertRaisesRegex(ValueError, "decompression bomb"):
                saving_lib.load_model(evil)


class SafeGetH5DatasetTest(testing.TestCase):
    def _shape_bomb_file(self):
        """An HDF5 file with a dataset declaring ~8 PiB but storing ~nothing."""
        path = os.path.join(self.get_temp_dir(), "bomb.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "d",
                shape=(2**50,),
                dtype="float64",
                chunks=(1024,),
                compression="gzip",
                fillvalue=0.0,
            )
        return path

    def test_rejects_shape_bomb(self):
        path = self._shape_bomb_file()
        self.assertLess(os.path.getsize(path), 1 << 20)  # tiny file on disk
        with h5py.File(path, "r") as f:
            with self.assertRaisesRegex(ValueError, "shape bomb"):
                saving_lib.safe_get_h5_dataset(f, "d")

    def test_load_weights_rejects_shape_bomb(self):
        model = keras.Sequential(
            [keras.Input((4,)), keras.layers.Dense(3, name="d")]
        )
        good_path = os.path.join(self.get_temp_dir(), "good.weights.h5")
        model.save_weights(good_path)

        # Replace a real weight dataset with a shape bomb at the same path.
        datasets = []
        with h5py.File(good_path, "r") as f:

            def collect(name, obj):
                if isinstance(obj, h5py.Dataset) and "/vars/" in "/" + name:
                    datasets.append(name)

            f.visititems(collect)
        with h5py.File(good_path, "r+") as f:
            del f[datasets[0]]
            f.create_dataset(
                datasets[0],
                shape=(2**50,),
                dtype="float64",
                chunks=(1024,),
                compression="gzip",
                fillvalue=0.0,
            )

        reloaded = keras.Sequential(
            [keras.Input((4,)), keras.layers.Dense(3, name="d")]
        )
        with self.assertRaises(ValueError):
            reloaded.load_weights(good_path)


class SavingDiskIOStoreTest(testing.TestCase):
    def test_disk_io_store_rejects_path_traversal(self):
        store = saving_lib.DiskIOStore("assets", archive=None, mode="w")
        working_dir = os.path.realpath(store.working_dir)
        for bad in ["../escape", os.path.join("a", "..", "..", "escape"), ".."]:
            with self.assertRaisesRegex(ValueError, "Invalid asset path"):
                store.make(bad)
            with self.assertRaisesRegex(ValueError, "Invalid asset path"):
                store.get(bad)
            self.assertFalse(store.has_path(bad))
        # Nothing was created outside the working directory.
        self.assertFalse(
            os.path.exists(os.path.join(os.path.dirname(working_dir), "escape"))
        )
        # Normal nested asset paths still work.
        made = store.make(os.path.join("layers", "dense"))
        self.assertTrue(os.path.isdir(made))
        self.assertTrue(os.path.realpath(made).startswith(working_dir + os.sep))
        self.assertIsNotNone(store.get(os.path.join("layers", "dense")))
        store.close()

    def test_disk_io_store_rejects_backslash_traversal(self):
        store = saving_lib.DiskIOStore("assets", archive=None, mode="w")
        for bad in ["..\\escape", "a\\..\\..\\escape"]:
            with self.assertRaisesRegex(ValueError, "Invalid asset path"):
                store.make(bad)
        store.close()

    def test_disk_io_store_remote_working_dir_is_preserved(self):
        store = saving_lib.DiskIOStore(
            "gs://bucket/model", archive=None, mode="r"
        )
        resolved = store._full_path(os.path.join("layers", "dense"))
        self.assertTrue(resolved.startswith("gs://bucket/model/"))
        self.assertTrue(resolved.endswith("layers/dense"))
        for bad in ["../escape", "/abs", os.path.join("x", "..", "..", "y")]:
            with self.assertRaisesRegex(ValueError, "Invalid asset path"):
                store._full_path(bad)


class SavingNpzIOStoreTest(testing.TestCase):
    def _write_npz_member(self, path, name, shape, descr="<f8", data=b""):
        """Write an npz `name` member declaring `shape` but storing no `data`.

        Used to craft a member whose `.npy` header declares a huge array while
        almost nothing is stored on disk (a shape/decompression bomb).
        """
        header = BytesIO()
        npy_format.write_array_header_1_0(
            header,
            {"descr": descr, "fortran_order": False, "shape": shape},
        )
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{name}.npy", header.getvalue() + data)

    def test_npz_io_store_rejects_shape_bomb(self):
        # Member declares an 8 PiB array but stores only its header.
        temp_filepath = os.path.join(self.get_temp_dir(), "bomb.npz")
        self._write_npz_member(temp_filepath, "w", shape=(2**50,))
        store = saving_lib.NpzIOStore(temp_filepath, mode="r")
        with self.assertRaisesRegex(
            ValueError, r"Refusing to load npz weight 'w'"
        ):
            store.get("w")

    def test_npz_io_store_loads_normal_array(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "store.npz")
        a = np.arange(6, dtype="float32").reshape(2, 3)
        np.savez(temp_filepath, w=a)
        store = saving_lib.NpzIOStore(temp_filepath, mode="r")
        self.assertAllClose(store.get("w"), a.tolist())
