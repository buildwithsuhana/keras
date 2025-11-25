import contextlib
import keras
from keras.src import backend
from keras.src import ops
import numpy as np

class GhostVariable:
    """
    A Fake Variable class that mimics Keras Variable but allocates 0 bytes.
    
    This class mimics the API of backend.Variable so that Keras internals
    (tracking, building, etc.) think it is a real variable.
    """
    def __init__(self, initializer, shape=None, dtype=None, trainable=True, name=None, **kwargs):
        self.name = name
        self._shape = shape
        self._dtype = dtype or "float32"
        self.trainable = trainable
        self.initializer = initializer
        
        # Internal Keras attributes usually needed
        self.path = name if name else "ghost_var"
        self._ndim = len(shape) if shape else 0

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype
    
    @property
    def ndim(self):
        return self._ndim

    def numpy(self):
        # Return a tiny placeholder if accessed.
        # This prevents OOM if Keras tries to inspect the value.
        print(f"⚠️  Warning: Accessing numpy value of GhostVariable {self.name} (Returning 1-element zero)")
        return np.zeros((1,), dtype=self._dtype)

    def __repr__(self):
        return f"<GhostVariable name={self.name} shape={self.shape} (0 bytes)>"

    # --- API Mocks to fool Keras ---
    
    def assign(self, value):
        pass 
    
    def experimental_ref(self):
        return self
    
    def value(self):
        return self

    def __array__(self):
        return self.numpy()

@contextlib.contextmanager
def lazy_init_scope():
    """
    Context manager that intercepts Keras Variable creation.
    We swap the backend.Variable CLASS with our GhostVariable CLASS.
    """
    # 1. Capture the original class
    original_var_cls = backend.Variable
    
    # 2. Swap it with our Ghost Class
    # This ensures isinstance(x, backend.Variable) passes when x is a GhostVariable
    backend.Variable = GhostVariable
    
    try:
        yield
    finally:
        # 3. Restore original class
        backend.Variable = original_var_cls