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

def sample_lds(true_lds, num_trials, time_bins):
    all_states, all_data = [], []
    for i in range(num_trials):
        states, data = true_lds.sample(T=time_bins)
        all_states.append(states)
        all_data.append(data)
    return all_states, all_data

def time_lds_em(emission_dim=10, latent_dim=2, num_trials=5, time_bins=200, num_iters=100):

    N = emission_dim
    D = latent_dim
    
    true_lds = LDS(N, D, dynamics="gaussian", emissions="gaussian")
    all_states, all_data = sample_lds(true_lds, num_trials=num_trials, time_bins=time_bins)
    test_lds = LDS(N, D, dynamics="gaussian", emissions="gaussian")
    (q_elbos_lem_train, q_lem_train), time_elapsed = timer(test_lds.fit)(
        datas=all_data,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_init_iters=0, num_iters=num_iters, verbose=False
    )
    return (q_elbos_lem_train, q_lem_train), time_elapsed

