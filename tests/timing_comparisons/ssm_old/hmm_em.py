from time import time

import matplotlib.pyplot as plt
import numpy as np
import ssm
from ssm import HMM
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

def sample_hmm(true_hmm, num_trials, time_bins):
    all_states, all_data = [], []
    for i in range(num_trials):
        states, data = true_hmm.sample(T=time_bins)
        all_states.append(states)
        all_data.append(data)
    return all_states, all_data

def time_hmm_em(emission_dim=10, latent_dim=2, num_trials=5, time_bins=200, num_iters=100):
    true_hmm = HMM(emission_dim, latent_dim, dynamics="gaussian", emissions="gaussian")
    all_states, all_data = sample_hmm(true_hmm, num_trials=num_trials, time_bins=time_bins)
    test_hmm = HMM(emission_dim, latent_dim, dynamics="gaussian", emissions="gaussian")
    (lls), time_elapsed = timer(test_hmm.fit)(
        datas=all_data,
        method="em",
        num_init_iters=0, 
        num_iters=num_iters,
        tolerance=-1,
    )
    return (lls), time_elapsed

