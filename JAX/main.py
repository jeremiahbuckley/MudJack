from fastapi import FastAPI
from fastapi.responses import JSONResponse
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import time
import traceback

app = FastAPI()

# Test functions
def test_functions():
    results = {}
    functions_to_test = {
        'jax.named_call': test_named_call,
        'jax.numpy.array': test_array,
        'jax.numpy.zeros': test_zeros,
        'jax.lax.select': test_select,
        'jax._src.interpreters.partial_eval': test_partial_eval,
        'jax.core.eval_context': test_eval_context,
        'jax.lax.all_gather': test_all_gather,
        'jax.lax.integer_pow': test_integer_pow,
        'jax.numpy.size': test_size,
        'jax.tree_util.Partial': test_partial,
        'jax.make_jaxpr': test_make_jaxpr,
        'jax.numpy.log': test_log,
        'jax.numpy.isscalar': test_isscalar,
        'jax.tree_util.tree_unflatten': test_tree_unflatten,
        'jax.vjp': test_vjp,
        'jax.numpy.einsum_path': test_einsum_path,
        'jax.numpy.delete': test_delete,
        'jax._src.interpreters.partial_eval.trace_to_jax_pr_dynamic': test_trace_to_jax_pr_dynamic,
        'jax.scipy.stats.norm.cdf': test_norm_cdf,
        'jax.lax.stop_gradient': test_stop_gradient,
        'jax.numpy.reshape': test_reshape,
        'jax.numpy.average': test_average,
        'jax.disable_jit': test_disable_jit,
        'jax.tree_util.tree_map': test_tree_map,
        'jax._src.core.get_aval': test_get_aval,
        'jax.scipy.signal.convolve2d': test_convolve2d,
        'jax.lax.erf': test_erf,
        'jax.scipy.special.ndtr': test_ndtr,
        'jax.numpy.convolve': test_convolve,
        'jax.numpy.linalg.svd': test_svd,
        'jax.numpy.compress': test_compress,
        'jax.numpy.stack': test_stack,
        'jax.scipy.special.i0': test_i0,
        'jax.numpy.var': test_var,
        'jax.numpy.tril': test_tril,
        'jax.numpy.sum': test_sum,
        'jax.numpy.triu_indices': test_triu_indices,
        'jax.numpy.power': test_power,
        'jax.numpy.ones': test_ones,
        'jax.lax.pmax': test_pmax,
        'jax.numpy.max': test_max,
        'jax.scipy.linalg.lu': test_lu,
        'jax.numpy.prod': test_prod,
        'jax.lax.slice_in_dim': test_slice_in_dim,
        'jax.lax.bitwise_and': test_bitwise_and,
        'jax.numpy.tril_indices_from': test_tril_indices_from,
        'jax.numpy.arange': test_arange,
        'jax.numpy.add': test_add,
        'jax.numpy.all': test_all,
        'jax.scipy.special.gammaln': test_gammaln,
        'jax.numpy.mean': test_mean,
        'jax.numpy.flip': test_flip,
        'jax.numpy.split': test_split,
        'jax.numpy.fliplr': test_fliplr,
        'jax.lax.top_k': test_top_k,
        'jax.numpy.exp': test_exp,
        'jax.lax.ge': test_ge,
        'jax.nn.one_hot': test_one_hot,
        'jax.random.PRNGKey': test_prngkey,
        'jax.numpy.cos': test_cos,
        'jax.numpy.sqrt': test_sqrt
    }

    for func_name, func in functions_to_test.items():
        func_result = {'error': None, 'average_time': None}
        try:
            # Warm-up run
            func()

            # Timing over 10 runs
            total_time = 0.0
            for _ in range(10):
                start_time = time.time()
                func()
                end_time = time.time()
                total_time += (end_time - start_time)
            func_result['average_time'] = total_time / 10
        except Exception as e:
            func_result['error'] = traceback.format_exc()
        results[func_name] = func_result

    return results

# Individual test implementations for each JAX function
def test_named_call():
    @jax.named_call
    def f(x):
        return x + 1
    f(jnp.array(1))

def test_array():
    jnp.array([1, 2, 3])

def test_zeros():
    jnp.zeros((10, 10))

def test_select():
    cond = jnp.array([True, False])
    x = jnp.array([1, 2])
    y = jnp.array([3, 4])
    jax.lax.select(cond, x, y)

def test_partial_eval():
    pass # Placeholder for partial evaluation example

def test_eval_context():
    with jax.core.eval_context():
        jnp.dot(jnp.array([1, 2]), jnp.array([3, 4]))

def test_all_gather():
    x = jnp.array([1, 2, 3, 4])
    jax.lax.all_gather(x, axis=0)

def test_integer_pow():
    jax.lax.integer_pow(jnp.array(2), 3)

def test_size():
    jnp.size(jnp.array([1, 2, 3]))

def test_partial():
    pass # Placeholder for partial utility

def test_make_jaxpr():
    def f(x):
        return x + 1
    jax.make_jaxpr(f)(1)

def test_log():
    jnp.log(jnp.array([1.0, 2.0, 3.0]))

def test_isscalar():
    jnp.isscalar(3)

def test_tree_unflatten():
    pass # Placeholder for tree unflattening

def test_vjp():
    def f(x):
        return x ** 2
    jax.vjp(f, 1.0)

def test_einsum_path():
    jnp.einsum_path('i,j->ij', jnp.arange(3), jnp.arange(3))

def test_delete():
    jnp.delete(jnp.array([1, 2, 3]), 1)

def test_trace_to_jax_pr_dynamic():
    pass # Placeholder for dynamic tracing

def test_norm_cdf():
    jsp.stats.norm.cdf(0.5)

def test_stop_gradient():
    jax.lax.stop_gradient(jnp.array([1.0, 2.0, 3.0]))

def test_reshape():
    jnp.reshape(jnp.array([1, 2, 3, 4]), (2, 2))

def test_average():
    jnp.average(jnp.array([1, 2, 3, 4]))

def test_disable_jit():
    jax.disable_jit(True)

def test_tree_map():
    pass # Placeholder for tree map example

def test_get_aval():
    pass # Placeholder for get_aval example

def test_convolve2d():
    jsp.signal.convolve2d(jnp.ones((5, 5)), jnp.ones((3, 3)))

def test_erf():
    jax.lax.erf(jnp.array(0.5))

def test_ndtr():
    jsp.special.ndtr(jnp.array(0.5))

def test_convolve():
    jnp.convolve(jnp.array([1, 2, 3]), jnp.array([0, 1, 0.5]))

def test_svd():
    jnp.linalg.svd(jnp.ones((3, 3)))

def test_compress():
    jnp.compress([0, 1], jnp.array([1, 2]))

def test_stack():
    jnp.stack([jnp.array([1, 2]), jnp.array([3, 4])])

def test_i0():
    jsp.special.i0(jnp.array([1.0, 2.0]))

def test_var():
    jnp.var(jnp.array([1, 2, 3]))

def test_tril():
    jnp.tril(jnp.ones((3, 3)))

def test_sum():
    jnp.sum(jnp.array([1, 2, 3]))

def test_triu_indices():
    jnp.triu_indices(3)

def test_power():
    jnp.power(jnp.array([1, 2, 3]), 2)

def test_ones():
    jnp.ones((3, 3))

def test_pmax():
    x = jnp.array([1, 2, 3])
    jax.lax.pmax(x, axis_name='i')

def test_max():
    jnp.max(jnp.array([1, 2, 3]))

def test_lu():
    jsp.linalg.lu(jnp.array([[1, 2], [3, 4]]))

def test_prod():
    jnp.prod(jnp.array([1, 2, 3]))

def test_slice_in_dim():
    x = jnp.array([1, 2, 3])
    jax.lax.slice_in_dim(x, start_index=0, limit_index=2)

def test_bitwise_and():
    jax.lax.bitwise_and(jnp.array([1, 2]), jnp.array([2, 3]))

def test_tril_indices_from():
    jnp.tril_indices_from(jnp.ones((3, 3)))

def test_arange():
    jnp.arange(10)

def test_add():
    jnp.add(1, 2)

def test_all():
    jnp.all(jnp.array([True, False]))

def test_gammaln():
    jsp.special.gammaln(jnp.array([1.0, 2.0]))

def test_mean():
    jnp.mean(jnp.array([1, 2, 3]))

def test_flip():
    jnp.flip(jnp.array([1, 2, 3]))

def test_split():
    jnp.split(jnp.array([1, 2, 3, 4]), 2)

def test_fliplr():
    jnp.fliplr(jnp.array([[1, 2], [3, 4]]))

def test_top_k():
    jax.lax.top_k(jnp.array([1, 3, 2]), 2)

def test_exp():
    jnp.exp(jnp.array([1.0, 2.0]))

def test_ge():
    jax.lax.ge(jnp.array([1, 2]), jnp.array([2, 1]))

def test_one_hot():
    jax.nn.one_hot(jnp.array([0, 1, 2]), 3)

def test_prngkey():
    jax.random.PRNGKey(0)

def test_cos():
    jnp.cos(jnp.array([1.0, 2.0]))

def test_sqrt():
    jnp.sqrt(jnp.array([4.0, 9.0]))

@app.get("/test")
async def run_tests():
    results = test_functions()
    return JSONResponse(content=results)

