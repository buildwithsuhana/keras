from typing import Any

from keras.src.distribution.auto_sharding import interfaces
from keras.src.distribution.auto_sharding.core_components import ShardingPlan


class JaxShardApplier(interfaces.IShardApplier):
    """
    Applies a ShardingPlan to a Keras model using JAX APIs.
    """

    def apply(self, model: Any, plan: ShardingPlan) -> Any:
        """
        Applies the sharding plan to the model's parameters.

        A full implementation would wrap the model's `call` method or
        use other mechanisms to apply `with_sharding_constraint` to the
        weights as they are used.
        """
        print("JaxShardApplier: Applying sharding plan to the model...")

        if not plan.layout_map.is_empty():
            print("JaxShardApplier: Sharding rules found in plan.")
        else:
            print("JaxShardApplier: No sharding rules to apply.")

        return model
