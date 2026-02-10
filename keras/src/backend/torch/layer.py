import torch

from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.ops.operation import Operation


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
        # Only add floating-point or complex tensors as Parameters
        # Integer tensors (like padding masks) cannot be Parameters
        params_dict = {}
        for variable in self.variables:
            value = variable.value
            # Check if value can be wrapped in Parameter
            # PyTorch only allows floating point or complex dtypes for requires_grad
            
            # Check if value is already a DTensor (from distributed training)
            is_dtensor = hasattr(torch, 'distributed') and hasattr(value, 'to_local')
            
            if is_dtensor:
                # DTensor - check the local tensor's dtype
                local_tensor = value.to_local()
                is_float_or_complex = (local_tensor.dtype.is_floating_point or 
                                       local_tensor.dtype.is_complex)
            elif isinstance(value, torch.Tensor):
                is_float_or_complex = value.dtype.is_floating_point or value.dtype.is_complex
            else:
                is_float_or_complex = False
            
            if is_float_or_complex:
                # Wrap in Parameter for gradient tracking
                if isinstance(value, torch.nn.Parameter):
                    # Already a Parameter, just use it
                    pass
                elif is_dtensor:
                    # For DTensors, wrap the DTensor in Parameter
                    # PyTorch Parameter can wrap DTensor if local tensor is float/complex
                    value = torch.nn.Parameter(value, requires_grad=variable.trainable)
                else:
                    # Regular tensor - wrap in Parameter
                    value = torch.nn.Parameter(value, requires_grad=variable.trainable)
            else:
                # Integer/bool tensors cannot be Parameters, store as-is
                # These are typically not trainable weights (e.g., padding masks)
                # Don't wrap in Parameter
                pass
            
            params_dict[variable.path] = value
        self._torch_params = torch.nn.ParameterDict(params_dict)

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
                value = variable.value
                # Skip adding non-floating point tensors to ParameterDict
                # PyTorch's ParameterDict will try to wrap values in Parameter()
                # which fails for integer tensors (can't require gradients)
                
                # Check if value is a DTensor (from distributed training)
                is_dtensor = hasattr(torch, 'distributed') and hasattr(value, 'to_local')
                
                if is_dtensor:
                    # DTensor - check the local tensor's dtype
                    local_tensor = value.to_local()
                    is_float_or_complex = (local_tensor.dtype.is_floating_point or 
                                           local_tensor.dtype.is_complex)
                elif isinstance(value, torch.Tensor):
                    is_float_or_complex = value.dtype.is_floating_point or value.dtype.is_complex
                else:
                    is_float_or_complex = False
                
                if not is_float_or_complex:
                    # Skip non-floating point tensors (e.g., padding masks)
                    # Don't try to wrap them in Parameter
                    return
                
                # Add to ParameterDict - handle DTensors and regular tensors
                if isinstance(value, torch.nn.Parameter):
                    self.torch_params[variable.path] = value
                elif is_dtensor:
                    # Wrap DTensor in Parameter if local tensor is float/complex
                    self.torch_params[variable.path] = torch.nn.Parameter(
                        value, requires_grad=variable.trainable
                    )
                else:
                    self.torch_params[variable.path] = torch.nn.Parameter(
                        value, requires_grad=variable.trainable
                    )

    def _post_untrack_variable(self, variable):
        if hasattr(self, "_torch_params"):
            if variable.path in self.torch_params:
                self.torch_params.pop(variable.path)
