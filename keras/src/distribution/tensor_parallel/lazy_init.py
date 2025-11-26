import contextlib
from keras.src import backend

# DO NOT capture OriginalVariable here globally. 
# It might capture the wrong class if the backend isn't initialized yet.

class LazyVariable(backend.Variable):
    """
    A placeholder for a Keras Variable that holds no physical memory.
    It inherits from the original backend.Variable to satisfy type checks.
    """
    def __init__(self, initializer, shape=None, dtype=None, trainable=True, name=None, **kwargs):
        # FIX: Accept **kwargs to handle 'autocast', 'aggregation', etc.
        # CRITICAL: Do NOT call super().__init__(). We bypass memory allocation.
        
        self._initializer = initializer
        self._shape = shape
        self._dtype = dtype
        self._trainable = trainable
        self._name = name
        self._kwargs = kwargs

        # Mirror key Variable attributes expected by `Variable` methods.
        # Provide sensible defaults based on kwargs when available.
        self._autocast = bool(kwargs.get("autocast", True))
        self._aggregation = kwargs.get("aggregation", "none")
        self._synchronization = kwargs.get("synchronization", "auto")
        self._ndim = len(shape) if shape is not None else 0
        parent_path = None
        if parent_path:
            self._path = f"{parent_path}/{name}"
        else:
            self._path = name or "lazy_variable"

        # JAX backend specific attribute / placeholder for materialized value
        self._value = None
        # Layout may be queried by backend-specific Variable subclasses
        self._layout = None

    # --- Properties to override parent class read-only attributes ---

    @property
    def path(self):
        # Override path to avoid "no setter" error.
        return self._name if self._name else "lazy_variable"

    @property
    def ndim(self):
        # Override ndim to avoid "no setter" error.
        return len(self._shape) if self._shape else 0

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype
    
    @property
    def name(self):
        return self._name
    
    @property
    def trainable(self):
        return self._trainable
    
    def numpy(self):
        raise ValueError(
            f"Cannot convert LazyVariable '{self.name}' to numpy. "
            "It has not been materialized yet. This error usually occurs "
            "if you try to access weights before the distribution strategy "
            "has processed them."
        )

    def __repr__(self):
        return f"<LazyVariable name='{self.name}' shape={self.shape} (Unmaterialized)>"


@contextlib.contextmanager
def lazy_init_scope():
    """
    Context manager to skip actual memory allocation during model definition.
    It patches backend.Variable with the LazyVariable class.
    """
    # 1. Capture the original Variable class NOW, ensuring we get the JAX/Torch one
    original_variable_class = backend.Variable
    
    try:
        print("   [LazyInit] 👻 Entering Ghost Mode (No memory allocation)...")
        # 2. Patch the backend.Variable with our Class
        backend.Variable = LazyVariable
        yield
    finally:
        # 3. Restore the specific class we captured
        backend.Variable = original_variable_class
        print("   [LazyInit] 👻 Exiting Ghost Mode.")