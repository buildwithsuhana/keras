import os
import torch

from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.ops.operation import Operation

# Debug flag for distributed training issues
DEBUG_LAYER = os.environ.get("KERAS_TORCH_LAYER_DEBUG", "0") == "1"


def _make_torch_param(value):
    """Convert a tensor to torch.nn.Parameter if possible.

    Only floating point and complex tensors can be wrapped as
    torch.nn.Parameter with requires_grad=True. Non-floating point
    tensors (like int32) must be stored as regular tensors.

    Also preserves existing torch.nn.Parameter instances to avoid
    double-wrapping which can cause errors with non-floating dtypes.
    """
    # If already a Parameter, return as-is to avoid double-wrapping
    if isinstance(value, torch.nn.Parameter):
        if DEBUG_LAYER:
            print(f"[DEBUG _make_torch_param] Skipping double-wrap for existing Parameter")
        return value

    if hasattr(value, 'dtype'):
        is_float_or_complex = value.dtype.is_floating_point or value.dtype.is_complex
        if DEBUG_LAYER:
            print(f"[DEBUG _make_torch_param] value.dtype={value.dtype}, is_float_or_complex={is_float_or_complex}")
        if is_float_or_complex:
            requires_grad = getattr(value, 'requires_grad', True)
            if DEBUG_LAYER:
                print(f"[DEBUG _make_torch_param] Creating Parameter with requires_grad={requires_grad}")
            return torch.nn.Parameter(value, requires_grad=requires_grad)
    if DEBUG_LAYER:
        print(f"[DEBUG _make_torch_param] Returning value as-is (non-floating dtype)")
    return value


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
        # Use a regular dict instead of ParameterDict to avoid automatic
        # wrapping of non-floating point tensors as Parameters (which would
        # cause RuntimeError: Only Tensors of floating point and complex
        # dtype can require gradients).
        # We manually register parameters with the Module to ensure they
        # are tracked correctly.
        self._torch_params = {}
        for variable in self.variables:
            param = _make_torch_param(variable.value)
            self._torch_params[variable.path] = param
            # Register with PyTorch module for proper tracking
            # Only register if it's a Parameter (to avoid duplicate registration)
            if isinstance(param, torch.nn.Parameter) and not hasattr(self, variable.path):
                self.register_parameter(variable.path, param)

    def named_parameters(
        self,
        prefix="",
        recurse=True,
        remove_duplicate=True,
    ):
        if not hasattr(self, "_torch_params"):
            self._track_variables()
        # Use Module's named_parameters to get all registered parameters
        # This now includes parameters we registered via register_parameter
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
        if hasattr(self, "_torch_params"):
            if variable.path not in self.torch_params:
                param = _make_torch_param(variable.value)
                self.torch_params[variable.path] = param
                # Register with PyTorch module for proper tracking
                if isinstance(param, torch.nn.Parameter) and not hasattr(self, variable.path):
                    self.register_parameter(variable.path, param)

    def _post_untrack_variable(self, variable):
        if hasattr(self, "_torch_params"):
            if variable.path in self.torch_params:
                # Unregister from PyTorch module if it was registered
                if hasattr(self, variable.path):
                    self._parameters.pop(variable.path, None)
                self.torch_params.pop(variable.path)
