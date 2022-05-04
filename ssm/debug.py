import jax
from jax import lax, jit

DEBUG = False
AUTO_DEBUG = False

def _index_array(i, aval, x):
    if aval is jax.core.abstract_unit:
        return jax.core.unit
    else:
        return jax.lax.index_in_dim(x, i, keepdims=False)

# Code taken from jax._src.lax.control_flow
def _debug_scan(f, init, xs):
    xs_flat, xs_tree = jax.tree_flatten(xs)
    carry = init
    ys = []
    for i in range(xs_flat[0].shape[0]):
        xs_slice = [_index_array(i, jax.core.get_aval(x), x) for x in xs_flat]
        carry, y = f(carry, jax.tree_unflatten(xs_tree, xs_slice))
        ys.append(y)
    stack = lambda y, *ys: (y if jax.core.get_aval(y) is jax.core.abstract_unit
                            else jax.numpy.stack((y, *ys)))
    out = jax.tree_multimap(stack, *ys)
    return carry, out

def scan(f, init, xs):
    if DEBUG:
        return _debug_scan(f, init, xs)
    else:
        return lax.scan(f, init, xs)

def debug_jit(f):
    def wrapper(*args, **kwargs):
        if not DEBUG:
            if AUTO_DEBUG:
                print(
"Auto-debug mode on, when the model crashes the last iteration will be re-run in debug mode.")
            func = jit(f)
        else:
            print("Debug mode on, all jit-compiled functions will be decomplied.")
            func = f

        if (not DEBUG) and AUTO_DEBUG:
            try:
                return func(*args, **kwargs)
            except:
                print(
"Some jit-compiled code crashed. Running the last iteration in debug mode.")
                with Debug():
                    # Use the unjitted version
                    return f(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

class Debug(object):
    def __enter__(self):
        global DEBUG
        DEBUG = True
    
    def __exit__(self, type, value, traceback):
        global DEBUG
        DEBUG = False

class AutoDebug(object):
    def __enter__(self):
        global AUTO_DEBUG
        AUTO_DEBUG = True
    
    def __exit__(self, type, value, traceback):
        global AUTO_DEBUG
        AUTO_DEBUG = False