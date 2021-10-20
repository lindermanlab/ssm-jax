from tqdm.auto import tqdm
import numpy as np
import importlib
import functools
import time
import json
import glob

TIMING_TESTS = dict()

def get_test_func(mode, test_file, test_name):
    """Test function should return _, elapsed_time for a given trial.
    """
    module = importlib.import_module(f"{mode}.{test_file}")
    test_func = getattr(module, test_name)
    return test_func


def make_time_test(test_file, test_name, parameter_name):
    """Fancy decorator to define the testing harness.

    Runs the test_func for various parameters and collects and 
    writes outputs to a json file.

    Parameters to the decorator specify the test_func that should be
    imported in.

    Decorator should decorate a generator object that generates
    (params: dict<str,value>, time_elapsed: float) by running the 
    test_func over various parameters.
    """
    def Inner(my_generator):
        def wrapper(mode, write_to_file):
            # import the appropriate test function (params => output, elapsed_time)
            test_func = get_test_func(mode, test_file, test_name)

            # get name for data output file
            outfile_base = f"data/{mode}.{test_file}.{test_name}.{parameter_name}"
            num_existing = len(glob.glob(f"{outfile_base}*.json"))
            outfile = f"{outfile_base}-{num_existing+1}.json"
            
            results = []
            # run through params in test
            for (params, elapsed_time) in tqdm(my_generator(test_func, mode, write_to_file)):
                res = dict(params=params, time=elapsed_time)
                results.append(res)
                print(res)
                # dump to file each iteration to save progress
                if write_to_file:
                    with open(outfile, "w") as f:
                        json.dump(results, f, indent=4, sort_keys=True)
            return results
        # register wrappered functions using generator's name
        TIMING_TESTS[my_generator.__name__] = wrapper
        return wrapper
    return Inner

#### Define Tests

@make_time_test("laplace_em", "time_laplace_em", "num_trials")
def test_laplace_em_num_trials(test_fn, mode, write_to_file=True):
    for num_trials in range(0, 250, 25):
        if num_trials == 0: num_trials += 1
        _, elapsed_time = test_fn(num_trials=num_trials)
        params = dict(num_trials=num_trials)
        yield params, elapsed_time

@make_time_test("lds_em", "time_lds_em", "num_trials")
def test_lds_em_num_trials(test_fn, mode, write_to_file=True):
    for num_trials in range(0, 500, 25):
        if num_trials == 0: num_trials += 1
        _, elapsed_time = test_fn(num_trials=num_trials)
        params = dict(num_trials=num_trials)
        yield params, elapsed_time

@make_time_test("lds_em", "time_lds_em", "num_timesteps")
def test_lds_em_num_timesteps(test_fn, mode, write_to_file=True): 
    for num_timesteps in range(100, 100000, 1000):
        _, elapsed_time = test_fn(num_trials=5, time_bins=num_timesteps)
        params = dict(num_timesteps=num_timesteps)
        yield params, elapsed_time

@make_time_test("hmm_em", "time_hmm_em", "num_trials")
def test_hmm_em_num_trials(test_fn, mode, write_to_file=True):
    for num_trials in range(0, 500, 25):
        if num_trials == 0: num_trials += 1
        _, elapsed_time = test_fn(num_trials=num_trials)
        params = dict(num_trials=num_trials)
        yield params, elapsed_time

@make_time_test("hmm_em", "time_hmm_em", "num_timesteps")
def test_hmm_em_num_timesteps(test_fn, mode="ssm_jax", write_to_file=True):
    for num_timesteps in range(100, 100000, 1000):
        _, elapsed_time = test_fn(num_trials=5, time_bins=num_timesteps)
        params = dict(num_timesteps=num_timesteps)
        yield params, elapsed_time

@make_time_test("hmm_em", "time_hmm_em", "latent_dim")
def test_hmm_em_latent_dim(test_fn, mode="ssm_jax", write_to_file=True):
    for latent_dim in range(1, 128, 5):
        _, elapsed_time = test_fn(latent_dim=latent_dim)
        params = dict(latent_dim=latent_dim)
        yield params, elapsed_time

@make_time_test("lds_em", "time_lds_em", "latent_dim")
def test_lds_em_latent_dim(test_fn, mode="ssm_jax", write_to_file=True):
    for latent_dim in range(1, 128, 5):
        _, elapsed_time = test_fn(latent_dim=latent_dim)
        params = dict(latent_dim=latent_dim)
        yield params, elapsed_time

@make_time_test("laplace_em", "time_laplace_em", "num_timesteps")
def test_laplace_em_num_timesteps(test_fn, mode="ssm_jax", write_to_file=True):
    for num_timesteps in range(100, 100000, 1000):
        _, elapsed_time = test_fn(num_trials=5, time_bins=num_timesteps)
        params = dict(num_timesteps=num_timesteps)
        yield params, elapsed_time

@make_time_test("laplace_em", "time_laplace_em", "latent_dim")
def test_laplace_em_latent_dim(test_fn, mode="ssm_jax", write_to_file=True):
    for latent_dim in range(1, 128, 5):
        _, elapsed_time = test_fn(latent_dim=latent_dim)
        params = dict(latent_dim=latent_dim)
        yield params, elapsed_time


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="ssm_jax", choices=["ssm_jax", "ssm_old"])
    parser.add_argument("--name", type=str, default=["test_num_trials"], choices=TIMING_TESTS, nargs="+")
    parser.add_argument('--save', default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # run tests
    for test_name in args.name:
        print("="*5, "Running timing test: ", test_name, "="*5)
        results = TIMING_TESTS[test_name](mode=args.mode, write_to_file=args.save)
        print(f"{'Parameters':<17} =", [i["params"] for i in results])
        print(f"{'Time Elapsed (s)':<17} =", [i["time"] for i in results])




    





