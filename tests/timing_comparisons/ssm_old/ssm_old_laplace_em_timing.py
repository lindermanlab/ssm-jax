from time import time

import matplotlib.pyplot as plt
import numpy as np
import ssm
from ssm import LDS
from tqdm.auto import trange


def timer(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        t_elapsed = t2-t1
        print(f'Function {func.__name__!r} executed in {t_elapsed:.4f}s')
        return result, t_elapsed
    return wrap_func

def sample_lds(true_lds, num_trials):
    all_states, all_data = [], []
    for i in range(num_trials):
        states, data = true_lds.sample(T=200)
        all_states.append(states)
        all_data.append(data)
    return all_states, all_data

def test_laplace_em(N=10, D=2, T=200, num_trials=1):
    true_lds = LDS(N, D, dynamics="gaussian", emissions="poisson")
    all_states, all_data = sample_lds(true_lds, num_trials=num_trials)
    test_lds = LDS(N, D, dynamics="gaussian", emissions="poisson")
    (q_elbos_lem_train, q_lem_train), time_elapsed = timer(test_lds.fit)(
        datas=all_data,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_init_iters=0, num_iters=100, verbose=False
    )
    return (q_elbos_lem_train, q_lem_train), time_elapsed

def run_test():
    times = dict()
    for t in trange(0, 31, 5):
        if t==0: t+=1
        _, time_elapsed = test_laplace_em(num_trials=t)
        times[t] = time_elapsed
    x = list(times.keys())
    y = list(times.values())
    out = np.array([x, y])
    np.save("../data/ssm_old/laplace_em_num_trials", out)
    print(out)
    return out

if __name__ == "__main__":
    run_test()
