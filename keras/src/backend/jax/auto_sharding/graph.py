import jax

from keras.src.distribution.auto_sharding import interfaces


class JaxGraph(interfaces.IKerasGraph):
    """
    A representation of the Keras model's computation graph using JAX's jaxpr.
    """

    def __init__(self, model_fn, *args, **kwargs):
        self._jaxpr = jax.make_jaxpr(model_fn)(*args, **kwargs)
        print("JaxGraph: Successfully generated jaxpr.")

    def analyze(self) -> jax.core.Jaxpr:
        return self._jaxpr
