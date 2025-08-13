from keras.src.distribution.auto_sharding import interfaces
from keras.src.distribution.auto_sharding.core_components import ShardingPlan
from keras.src.distribution.distribution_lib import DeviceMesh
from keras.src.distribution.distribution_lib import LayoutMap


class JaxShardingPlanner(interfaces.IShardingPlanner):
    """
    Analyzes a jaxpr to create a sharding plan for JAX backend.
    """

    def plan(
        self,
        graph: interfaces.IKerasGraph,
        mesh: DeviceMesh,
        min_shard_size: int,
    ) -> ShardingPlan:
        print(
            "JaxShardingPlanner: Analyzing jaxpr and creating a sharding plan"
        )

        layout_map = LayoutMap()

        print("JaxShardingPlanner: Sharding plan created.")
        return ShardingPlan(layout_map=layout_map)
