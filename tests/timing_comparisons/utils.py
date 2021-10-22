from tqdm.auto import tqdm
import numpy as np
import importlib
import traceback
import json
import glob
import sys

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

    Gracefully handles errors in the test function as well.

    Parameters to the decorator specify the test_func that should be
    imported in.

    Decorator should decorate a generator object that generates
    (params: dict<str,value>).
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
            for params in my_generator(mode, write_to_file):

                # catch any errors during execution 
                try:
                    print("-"*15)
                    print(f"Running {test_func.__name__}({params})...")
                    _, elapsed_time = test_func(**params)
                    res = dict(params=params, time=elapsed_time)
                except Exception as e:
                    exc_info = sys.exc_info()
                    error_traceback_str = ''.join(traceback.format_exception(*exc_info))
                    print("CAUGHT ERROR in test_func:", e)
                    res = dict(params=params, time=-1, exception=error_traceback_str)
                print(res)
                results.append(res)
                # dump to file each iteration to save progress
                if write_to_file:
                    with open(outfile, "w") as f:
                        json.dump(results, f, indent=4, sort_keys=True)
            return results
        # register wrappered functions using generator's name
        TIMING_TESTS[my_generator.__name__] = wrapper
        return wrapper
    return Inner