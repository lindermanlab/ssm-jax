import pytest
from jax.interpreters import xla

@pytest.fixture(autouse=True)
def cleanup():
    """Clears XLA cache after every test."""
    yield  # run the test
    # clear XLA cache to prevent OOM
    print("\nclearing XLA cache")
    xla._xla_callable.cache_clear()