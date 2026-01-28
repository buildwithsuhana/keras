import logging
import torch

from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.ops.operation import Operation

# Set up logging for torch layer
logger = logging.getLogger(__name__)


class TorchLayer(torch.nn.Module):
    @property
    def torch_params(self):
        if not hasattr(self, "_torch_params"):
            self._track_variables()
        return self._torch_params

    def _post_build(self):
        # Do not track variables when in a stateless scope.
        # The variables are not initialized.
        if in_stateless_scope():
            return
        self._track_variables()

    def _track_variables(self):
        # set torch_params attribute will have module automatically track
        # parameters.
        logger.info(f"Tracking variables for layer: {self.__class__.__name__}")
        self._torch_params = torch.nn.ParameterDict(
            {variable.path: variable.value for variable in self.variables}
        )
        logger.info(f"  Found {len(self.variables)} variables")
        for i, variable in enumerate(self.variables):
            logger.info(f"    [{i}] {variable.path}: shape={variable.shape}, dtype={variable.dtype}")

    def named_parameters(
        self,
        prefix="",
        recurse=True,
        remove_duplicate=True,
    ):
        if not hasattr(self, "_torch_params"):
            self._track_variables()
        return torch.nn.Module.named_parameters(
            self, prefix, recurse, remove_duplicate
        )

    def forward(self, *args, **kwargs):
        return Operation.__call__(self, *args, **kwargs)

    def _setattr_hook(self, name, value):
        from keras.src.layers import Layer

        if (
            isinstance(value, torch.nn.Module)
            and not isinstance(value, Layer)
            and not name == "_torch_params"
        ):
            from keras.src.utils.torch_utils import TorchModuleWrapper

            if not isinstance(self, TorchModuleWrapper):
                value = TorchModuleWrapper(value)
        return name, value

    def _post_track_variable(self, variable):
        logger.debug(f"Tracking new variable: {variable.path}")
        if hasattr(self, "_torch_params"):
            if variable.path not in self.torch_params:
                logger.info(f"  Adding variable to torch_params: {variable.path}")
                self.torch_params[variable.path] = variable.value

    def _post_untrack_variable(self, variable):
        logger.debug(f"Untracking variable: {variable.path}")
        if hasattr(self, "_torch_params"):
            if variable.path in self.torch_params:
                logger.info(f"  Removing variable from torch_params: {variable.path}")
                self.torch_params.pop(variable.path)
