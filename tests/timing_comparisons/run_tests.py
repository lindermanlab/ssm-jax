from tqdm.auto import trange
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

def register(func):
    """Register a timing test function as a test"""
    TIMING_TESTS[func.__name__] = func
    return func

# TODO: make decorator?
# def TimeTest(test_file, test_name):
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(mode, test_func):
#             test_func = get_test_func(mode, test_file, test_name)
#             times = func(mode, test_func)
#             x = list(times.keys())
#             y = list(times.values())
#             out = np.array([x, y])
#             np.save("../data/{self.mode}/{self.test_name}", out)
#             return out
#         return wrapper
#     return decorator

#### Define Tests
@register
def test_laplace_em_num_trials(mode, write_to_file=True):
    """Test different trials for Laplace EM
    """
    test_file = "laplace_em"
    test_name = "time_laplace_em"
    parameter_name = "num_trials"
    laplace_em_func = get_test_func(mode, test_file, test_name)

    outfile_base = f"data/{mode}/{test_file}.{test_name}.{parameter_name}"
    num_existing = len(glob.glob(f"{outfile_base}*.json"))
    outfile = f"{outfile_base}-{num_existing+1}.json"

    results = []
    for num_trials in trange(0, 250, 25):
        if num_trials == 0: num_trials += 1
        _, elapsed_time = laplace_em_func(num_trials=num_trials)
        
        results.append(dict(
            params={"num_trials": num_trials},
            time=elapsed_time)                  
        )
        
        # dump to file each iteration to save progress
        if write_to_file:
            with open(outfile, "w") as f:
                json.dump(results, f, indent=4, sort_keys=True)

    return results

@register
def test_lds_em_num_trials(mode, write_to_file=True):
    """Test different trials for LDS EM
    """
    test_file = "lds_em"
    test_name = "time_lds_em"
    parameter_name = "num_trials"
    lds_em_func = get_test_func(mode, test_file, test_name)

    outfile_base = f"data/{mode}/{test_file}.{test_name}.{parameter_name}"
    num_existing = len(glob.glob(f"{outfile_base}*.json"))
    outfile = f"{outfile_base}-{num_existing+1}.json"

    results = []
    for num_trials in range(0, 500, 25):
        if num_trials == 0: num_trials += 1
        _, elapsed_time = lds_em_func(num_trials=num_trials)
        
        results.append(dict(
            params={"num_trials": num_trials},
            time=elapsed_time)                  
        )
        
        # dump to file each iteration to save progress
        if write_to_file:
            with open(outfile, "w") as f:
                json.dump(results, f, indent=4, sort_keys=True)

    return results

@register
def test_lds_em_num_timesteps(mode, write_to_file=True):
    """Test different trials for LDS EM
    """
    test_file = "lds_em"
    test_name = "time_lds_em"
    parameter_name = "num_timesteps"
    lds_em_func = get_test_func(mode, test_file, test_name)

    outfile_base = f"data/{mode}/{test_file}.{test_name}.{parameter_name}"
    num_existing = len(glob.glob(f"{outfile_base}*.json"))
    outfile = f"{outfile_base}-{num_existing+1}.json"

    results = []
    for num_timesteps in range(100, 100000, 1000):
        _, elapsed_time = lds_em_func(num_trials=5, time_bins=num_timesteps)
        
        result = dict(
            params={"num_timesteps": num_timesteps},
            time=elapsed_time
        )   
        print(result)
        results.append(result)
        
        # dump to file each iteration to save progress
        if write_to_file:
            with open(outfile, "w") as f:
                json.dump(results, f, indent=4, sort_keys=True)

    return results


@register
def test_hmm_em_num_trials(mode, write_to_file=True):
    """Test different trials for LDS EM
    """
    test_file = "hmm_em"
    test_name = "time_hmm_em"
    parameter_name = "num_trials"
    hmm_em_func = get_test_func(mode, test_file, test_name)

    outfile_base = f"data/{mode}/{test_file}.{test_name}.{parameter_name}"
    num_existing = len(glob.glob(f"{outfile_base}*.json"))
    outfile = f"{outfile_base}-{num_existing+1}.json"

    results = []
    for num_trials in trange(0, 500, 25):
        if num_trials == 0: num_trials += 1
        _, elapsed_time = hmm_em_func(num_trials=num_trials)
        
        results.append(dict(
            params={"num_trials": num_trials},
            time=elapsed_time)                  
        )
        
        # dump to file each iteration to save progress
        if write_to_file:
            with open(outfile, "w") as f:
                json.dump(results, f, indent=4, sort_keys=True)

    return results

@register
def test_hmm_em_num_timesteps(mode, write_to_file=True):
    """Test different timestep lengths for HMM EM
    """
    test_file = "hmm_em"
    test_name = "time_hmm_em"
    parameter_name = "num_timesteps"
    hmm_em_func = get_test_func(mode, test_file, test_name)

    outfile_base = f"data/{mode}/{test_file}.{test_name}.{parameter_name}"
    num_existing = len(glob.glob(f"{outfile_base}*.json"))
    outfile = f"{outfile_base}-{num_existing+1}.json"

    results = []
    for num_timesteps in range(100, 100000, 1000):
        _, elapsed_time = hmm_em_func(num_trials=5, time_bins=num_timesteps)
        
        results.append(dict(
            params={"num_timesteps": num_timesteps},
            time=elapsed_time)                  
        )
        
        # dump to file each iteration to save progress
        if write_to_file:
            with open(outfile, "w") as f:
                json.dump(results, f, indent=4, sort_keys=True)

    return results


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




    





