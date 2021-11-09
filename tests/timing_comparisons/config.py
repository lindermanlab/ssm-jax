"""
Default values for benchmark tests.
"""

NUM_ROUNDS = 1
NUM_TRIALS = 5
NUM_TIMESTEPS = 200
LATENT_DIM = 3
EMISSIONS_DIM = 10
NUM_ITERS = 100

NUM_TRIALS_SWEEP = range(1, 202, 100)
NUM_TIMESTEPS_SWEEP = range(10, 20011, 10000)
LATENT_DIM_SWEEP = range(2, 13, 2)
EMISSIONS_DIM_SWEEP = range(2, 13, 2)

# fewer timesteps for Laplace EM for SSM-V0 
# to be reasonable with time durations
NUM_TIMESTEPS_SWEEP_LDS_SSM_V0 = range(10, 1011, 250)
