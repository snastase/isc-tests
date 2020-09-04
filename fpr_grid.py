from os.path import join
from sys import argv
import json
import numpy as np
from scipy.stats import ttest_1samp
from brainiak.isc import (isc, permutation_isc, bootstrap_isc,
                          phaseshift_isc, timeshift_isc,
                          compute_summary_statistic,
                          _check_timeseries_input, p_from_null)
from timeflip_isc import timeflip_isc
from simulate_data import correlated_data


# Get random seed for data generation
seed = argv[1]


# Set paths for saving outputs
base_dir = '/jukebox/hasson/snastase/isc-test'
sim_dir = join(base_dir, 'simulations')


# Function for computing p-value on simulated data for different tests
def isc_test(data, test, pairwise=False, n_randomizations=1000):

    if test == bootstrap_isc:
        iscs = isc(data, pairwise=pairwise)
        _, _, p, _ = test(iscs, n_bootstraps=n_randomizations)

    elif test == permutation_isc:
        iscs = isc(data, pairwise=pairwise)
        _, p, _ = test(iscs, n_permutations=n_randomizations)

    elif test in (phaseshift_isc, timeshift_isc):
        _, p, _ = test(data, pairwise=pairwise,
                       n_shifts=n_randomizations)

    elif test == timeflip_isc:
         _, p, _ = test(data, pairwise=pairwise,
                        n_flips=n_randomizations)

    elif test == ttest_1samp:
        iscs = isc(data, pairwise=pairwise)
        _, p = ttest_1samp(iscs, popmean=0, axis=0)

    return p


# Set grid parameters for simulated data and tests
r = 0
pairwise = False
n_randomizations = 1000
tests = [ttest_1samp, permutation_isc, bootstrap_isc,
         phaseshift_isc, timeshift_isc, timeflip_isc]
n_subjects = [10, 20, 30, 50, 100, 200, 500, 1000]
n_trs = [50, 100, 300, 500, 1000, 2000]

results = {'test': [], 'subjects (N)': [],
           'duration (TRs)': [],'p-values': []}
for n in n_subjects:
    for t in n_trs:

        # Create simulated data for this parameter set
        data = correlated_data(t, n, r, seed=seed)

        # Compute p-value for each type of test
        for test in tests:
            p = isc_test(data, test, pairwise=pairwise,
                         n_randomizations=n_randomizations)

            # Compile results
            results['test'].append(test.__name__)
            results['subjects (N)'].append(n)
            results['duration (TRs)'].append(t)
            results['p-values'].append(p[0])

            print(f"Finished simulation for {test.__name__} "
                  f"with {n} subjects ({t} TRs)")


# Save results for this simulation
results_fn = join(sim_dir, f'isc-loo_sim-{seed}_pvals.json')
with open(results_fn, 'w') as f:
    json.dump(results, f, indent=2)
