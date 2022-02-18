import pytest
from jax.interpreters import xla

@pytest.fixture(autouse=False)  # TODO: set to true once cache_clear is found
def cleanup():
    """Clears XLA cache after every test."""
    yield  # run the test
    # clear XLA cache to prevent OOM
    print("\nclearing XLA cache")
    # TODO: can't find this function in new JAX version
    xla._xla_callable.cache_clear()  