from utils import make_time_test, TIMING_TESTS

#### TEST HMM EM
@make_time_test("hmm_em", "time_hmm_em", "num_trials")
def test_hmm_em_num_trials(mode, write_to_file=True):
    for num_trials in range(0, 500, 25):
        if num_trials == 0: num_trials += 1
        params = dict(num_trials=num_trials)
        yield params

@make_time_test("hmm_em", "time_hmm_em", "num_timesteps")
def test_hmm_em_num_timesteps(mode, write_to_file=True):
    for num_timesteps in range(100, 100000, 1000):
        params = dict(num_timesteps=num_timesteps)
        yield params

@make_time_test("hmm_em", "time_hmm_em", "latent_dim")
def test_hmm_em_latent_dim(mode, write_to_file=True):
    for latent_dim in range(1, 128, 5):
        params = dict(latent_dim=latent_dim)
        yield params

@make_time_test("hmm_em", "time_hmm_em", "emission_dim")
def test_hmm_em_emission_dim(mode, write_to_file=True):
    for emission_dim in range(1, 128, 5):
        params = dict(emission_dim=emission_dim)
        yield params


#### TEST LDS EM
@make_time_test("lds_em", "time_lds_em", "num_trials")
def test_lds_em_num_trials(mode, write_to_file=True):
    for num_trials in range(0, 500, 25):
        if num_trials == 0: num_trials += 1
        params = dict(num_trials=num_trials)
        yield params

@make_time_test("lds_em", "time_lds_em", "num_timesteps")
def test_lds_em_num_timesteps(mode, write_to_file=True): 
    for num_timesteps in range(100, 100000, 1000):
        params = dict(num_timesteps=num_timesteps)
        yield params

@make_time_test("lds_em", "time_lds_em", "latent_dim")
def test_lds_em_latent_dim(mode, write_to_file=True):
    for latent_dim in range(1, 128, 5):
        params = dict(latent_dim=latent_dim)
        yield params

@make_time_test("lds_em", "time_lds_em", "emission_dim")
def test_lds_em_emission_dim(mode, write_to_file=True):
    for emission_dim in range(1, 128, 5):
        params = dict(emission_dim=emission_dim)
        yield params

#### TEST Laplace EM
@make_time_test("laplace_em", "time_laplace_em", "num_trials")
def test_laplace_em_num_trials(test_fn, mode, write_to_file=True):
    for num_trials in range(0, 250, 25):
        if num_trials == 0: num_trials += 1
        params = dict(num_trials=num_trials)
        yield params

@make_time_test("laplace_em", "time_laplace_em", "num_timesteps")
def test_laplace_em_num_timesteps(mode, write_to_file=True):
    for num_timesteps in range(100, 100000, 1000):
        params = dict(num_timesteps=num_timesteps)
        yield params

@make_time_test("laplace_em", "time_laplace_em", "latent_dim")
def test_laplace_em_latent_dim(mode, write_to_file=True):
    for latent_dim in range(1, 128, 5):
        params = dict(latent_dim=latent_dim)
        yield params

@make_time_test("laplace_em", "time_laplace_em", "emission_dim")
def test_laplace_em_emission_dim(mode, write_to_file=True):
    for emission_dim in range(1, 128, 5):
        params = dict(emission_dim=emission_dim)
        yield params


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




    





