"""Abstract interfaces for auto-sharding components."""

from abc import ABC
from abc import abstractmethod
from typing import Any

from keras.src.distribution.auto_sharding.core_components import ShardingPlan
from keras.src.distribution.distribution_lib import DeviceMesh


class IKerasGraph(ABC):
    """Abstract interface for representing a model's computation graph."""

    @abstractmethod
    def __init__(self, model_fn, *args, **kwargs):
        """Initializes the graph representation from a model function."""
        pass

    @abstractmethod
    def analyze(self) -> Any:
        """Analyzes the graph to extract structure for the planner."""
        pass


class IShardingPlanner(ABC):
    """Abstract interface for a sharding planner."""

    @abstractmethod
    def plan(
        self, graph: IKerasGraph, mesh: DeviceMesh, min_shard_size: int
    ) -> ShardingPlan:
        """Creates a sharding plan based on the graph and device mesh."""
        pass


class IShardApplier(ABC):
    """Abstract interface for a shard applier."""

    @abstractmethod
    def apply(self, model: Any, plan: ShardingPlan) -> Any:
        """Applies a `ShardingPlan` to a model."""
        pass
