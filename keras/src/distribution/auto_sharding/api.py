"""The user-facing API for the AutoShard distribution strategy."""

import importlib

import jax

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.distribution import distribution_lib
from keras.src.distribution.auto_sharding import interfaces


@keras_export("keras.distribution.experimental.AutoShardDistribution")
class AutoShardDistribution(distribution_lib.Distribution):
    """A distribution strategy that automatically shards models and data."""

    def __init__(
        self,
        device_mesh: distribution_lib.DeviceMesh = None,
        min_shard_size: int = 0,
        auto_shard_dataset=True,
    ):
        if device_mesh is None:
            print(
                "AutoShardDistribution: No device mesh provided. "
                "Auto-detecting hardware"
            )
            devices = jax.devices()
            if not devices:
                raise RuntimeError(
                    "No visible JAX devices found for auto-detection."
                )

            num_devices = len(devices)
            device_mesh = distribution_lib.DeviceMesh(
                shape=(num_devices,), axis_names=["data"]
            )
            print(
                f"AutoShardDistribution: Detected {num_devices} devices. "
                f"Created default mesh with shape=({num_devices},) and "
                "axis_names=['data']."
            )

        batch_dim_name = device_mesh.axis_names[0]
        super().__init__(
            device_mesh,
            batch_dim_name=batch_dim_name,
            auto_shard_dataset=auto_shard_dataset,
        )
        self.min_shard_size = min_shard_size
        self._sharding_plan = None
        self._model = None

        backend_name = backend.backend()
        try:
            graph_module = importlib.import_module(
                f"keras.src.backend.{backend_name}.auto_sharding.graph"
            )
            planner_module = importlib.import_module(
                f"keras.src.backend.{backend_name}.auto_sharding.planner"
            )
            applier_module = importlib.import_module(
                f"keras.src.backend.{backend_name}.auto_sharding.applier"
            )
            self._graph_impl = next(
                c
                for c in vars(graph_module).values()
                if isinstance(c, type) and issubclass(c, interfaces.IKerasGraph)
            )
            self._planner_impl = next(
                c
                for c in vars(planner_module).values()
                if isinstance(c, type)
                and issubclass(c, interfaces.IShardingPlanner)
            )
            self._applier_impl = next(
                c
                for c in vars(applier_module).values()
                if isinstance(c, type)
                and issubclass(c, interfaces.IShardApplier)
            )
        except (ImportError, StopIteration):
            raise ImportError(
                f"AutoShardDistribution is not supported for '{backend_name}'"
            )

    def _plan_and_apply(self, model, inputs_spec):
        """Helper function containing the core planning logic."""
        print(
            "AutoShardDistribution: Analyzing model and applying sharding plan"
        )
        dummy_inputs = [
            jax.ShapeDtypeStruct(spec.shape, spec.dtype.as_numpy_dtype)
            for spec in inputs_spec
        ]
        dummy_inputs = [
            jax.ShapeDtypeStruct(spec.shape, spec.dtype.as_numpy_dtype)
            for spec in inputs_spec
        ]

        graph_repr = self._graph_impl(model.call, *dummy_inputs)
        planner = self._planner_impl()
        self._sharding_plan = planner.plan(
            graph_repr, self.device_mesh, self.min_shard_size
        )
        applier = self._applier_impl()
        self._model = applier.apply(model, self._sharding_plan)
        print("AutoShardDistribution: Sharding plan applied successfully.")

    def build(self, model):
        """Attempts to eagerly build the sharding plan at compile time."""
        self._model = model
        inputs_spec = model.input_spec
        if not isinstance(inputs_spec, (list, tuple)):
            inputs_spec = [inputs_spec]

        is_fully_defined = all(
            spec.shape and None not in spec.shape for spec in inputs_spec
        )

        if is_fully_defined:
            self._plan_and_apply(model, inputs_spec)

        return self._model

    def distribute_dataset(self, dataset):
        """Distributes the dataset and triggers deferred planning if needed."""
        if self._sharding_plan is None:
            element_spec = dataset.element_spec
            inputs_spec = element_spec[0]
            if not isinstance(inputs_spec, (list, tuple)):
                inputs_spec = [inputs_spec]
            self._plan_and_apply(self._model, inputs_spec)

        return super().distribute_dataset(dataset)

    def get_variable_layout(self, variable):
        if self._sharding_plan is None:
            return distribution_lib.TensorLayout(
                [None] * len(variable.shape), self.device_mesh
            )
        layout = self._sharding_plan.layout_map.get(variable.path)
        if layout:
            return layout
        return distribution_lib.TensorLayout(
            [None] * len(variable.shape), self.device_mesh
        )

    def get_tensor_layout(self, tensor_path):
        if self._sharding_plan is None:
            return None
        layout = self._sharding_plan.layout_map.get(tensor_path)
        return layout
