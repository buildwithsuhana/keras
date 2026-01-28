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
        # Import path adapter here to avoid circular imports
        try:
            from keras.src.backend.torch.distribution_lib import TorchPathAdapter
            use_adapter = True
            print(f"[Torch Layer] Using path adapter for variable tracking")
        except ImportError:
            use_adapter = False
            print(f"[Torch Layer] Path adapter not available, using raw paths")
        
        # Create ParameterDict with both Keras-style and PyTorch-style keys
        param_dict = {}
        for variable in self.variables:
            keras_path = variable.path
            torch_path = TorchPathAdapter.keras_to_torch(keras_path) if use_adapter else keras_path
            
            # Store with both keys for compatibility
            param_dict[keras_path] = variable.value
            if torch_path != keras_path:
                param_dict[torch_path] = variable.value
                print(f"[Torch Layer] Tracked variable: keras_path='{keras_path}', torch_path='{torch_path}'")
            else:
                print(f"[Torch Layer] Tracked variable: path='{keras_path}'")
        
        self._torch_params = torch.nn.ParameterDict(param_dict)
        print(f"[Torch Layer] Total tracked parameters: {len(self._torch_params)}")

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
            keras_path = variable.path
            try:
                from keras.src.backend.torch.distribution_lib import TorchPathAdapter
                torch_path = TorchPathAdapter.keras_to_torch(keras_path)
                self.torch_params[keras_path] = variable.value
                if torch_path != keras_path:
                    self.torch_params[torch_path] = variable.value
                print(f"[Torch Layer] Post-track variable: {keras_path}")
            except ImportError:
                if keras_path not in self.torch_params:
                    self.torch_params[keras_path] = variable.value

    def _post_untrack_variable(self, variable):
        if hasattr(self, "_torch_params"):
            keras_path = variable.path
            try:
                from keras.src.backend.torch.distribution_lib import TorchPathAdapter
                torch_path = TorchPathAdapter.keras_to_torch(keras_path)
                if keras_path in self.torch_params:
                    self.torch_params.pop(keras_path)
                if torch_path in self.torch_params:
                    self.torch_params.pop(torch_path)
                print(f"[Torch Layer] Post-untrack variable: {keras_path}")
            except ImportError:
                if keras_path in self.torch_params:
                    self.torch_params.pop(keras_path)
