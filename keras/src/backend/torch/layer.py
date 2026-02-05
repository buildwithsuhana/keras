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
        # set torch_params attribute will have module automatically track
        # parameters.
        self._torch_params = torch.nn.ParameterDict(
            {variable.path: _make_torch_param(variable.value) for variable in self.variables}
        )

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
        if hasattr(self, "_torch_params"):
            if variable.path not in self.torch_params:
                self.torch_params[variable.path] = _make_torch_param(variable.value)

    def _post_untrack_variable(self, variable):
        if hasattr(self, "_torch_params"):
            if variable.path in self.torch_params:
                self.torch_params.pop(variable.path)
