import numpy as np
from brainiak.isc import (isc, compute_summary_statistic,
                          _check_timeseries_input, p_from_null)

MAX_RANDOM_SEED = 2**32 - 1


def timeflip_isc(data, pairwise=False, summary_statistic='median',
                 n_flips=1000, tolerate_nans=True, random_state=None):

    """Time-series flipping randomization for one-sample ISC
    
    For each voxel or ROI, compute the actual ISC and p-values
    from a null distribution of ISCs where response time series
    are first randomly flipped around zero (i.e. multiplied by 1 or -1).
    Input time-series are mean-centered prior to computing ISC and test.
    If pairwise, apply random time-series flipping to each subject and
    compute pairwise ISCs. If leave-one-out approach is used (pairwise=False),
    apply the random time-series flipping to only the left-out subject in each
    iteration of the leave-one-out procedure. Input data should be a list where
    each item is a time-points by voxels ndarray for a given subject.
    Multiple input ndarrays must be the same shape. If a single ndarray is
    supplied, the last dimension is assumed to correspond to subjects.
    When using leave-one-out approach, NaNs are ignored when computing mean
    time series of N-1 subjects (default: tolerate_nans=True). Alternatively,
    you may supply a float between 0 and 1 indicating a threshold proportion
    of N subjects with non-NaN values required when computing the average time
    series for a given voxel. For example, if tolerate_nans=.8, ISCs will be
    computed for any voxel where >= 80% of subjects have non-NaN values,
    while voxels with < 80% non-NaN values will be assigned NaNs. If set to
    False, NaNs are not tolerated and voxels with one or more NaNs among the
    N-1 subjects will be assigned NaN. Setting tolerate_nans to True or False
    will not affect the pairwise approach; however, if a threshold float is
    provided, voxels that do not reach this threshold will be excluded. Note
    that accommodating NaNs may be notably slower than setting tolerate_nans to
    False. Returns the observed ISC and p-values (two-tailed test), as well as
    the null distribution of ISCs computed on randomly flipped time-series
    data.
    
    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data for which to compute ISFC

    pairwise : bool, default: False
        Whether to use pairwise (True) or leave-one-out (False) approach

    summary_statistic : str, default: 'median'
        Summary statistic, either 'median' (default) or 'mean'

    n_flips : int, default: 1000
        Number of randomly flipped samples

    tolerate_nans : bool or float, default: True
        Accommodate NaNs (when averaging in leave-one-out approach)

    random_state = int, None, or np.random.RandomState, default: None
        Initial random seed

    Returns
    -------
    observed : float, observed ISC (without time-series flipping)
        Actual ISCs

    p : float, p-value
        p-value based on randomized time-series flipping

    distribution : ndarray, n_flips by voxels
        Time-series flipped null distribution
    """
    
    # Check response time series input format
    data, n_TRs, n_voxels, n_subjects = _check_timeseries_input(data)

    # Mean-center time series for flipping around zero
    data -= np.mean(data, axis=0)

    # Get actual observed ISC
    observed = isc(data, pairwise=pairwise,
                   summary_statistic=summary_statistic,
                   tolerate_nans=tolerate_nans)

    # Roll axis to get subjects in first dimension for loop
    if pairwise:
        data = np.rollaxis(data, 2, 0)

    # Iterate through randomized flips to create null distribution
    distribution = []
    for i in np.arange(n_flips):

        # Random seed to be deterministically re-randomized at each iteration
        if isinstance(random_state, np.random.RandomState):
            prng = random_state
        else:
            prng = np.random.RandomState(random_state)

        # Get a random set of flips based on number of subjects
        flips = prng.choice([-1, 1], size=n_subjects, replace=True)

        # In pairwise approach, apply all flips then compute pairwise ISCs
        if pairwise:

            # Apply flips to each subject's time series
            flipped_data = data * flips

            # Compute null ISC on shifted data for pairwise approach
            flipped_isc = isc(flipped_data, pairwise=pairwise,
                              summary_statistic=summary_statistic,
                              tolerate_nans=tolerate_nans)

        # In leave-one-out, apply flips only to each left-out participant
        elif not pairwise:

            flipped_isc = []
            for s, flip in enumerate(flips):
                flipped_subject = data[..., s] * flip
                nonflipped_mean = np.mean(np.delete(data, s, 2), axis=2)
                loo_isc = isc(np.dstack((flipped_subject, nonflipped_mean)),
                              pairwise=False,
                              summary_statistic=None,
                              tolerate_nans=tolerate_nans)
                flipped_isc.append(loo_isc)

            # Get summary statistics across left-out subjects
            flipped_isc = compute_summary_statistic(
                              np.dstack(flipped_isc),
                              summary_statistic=summary_statistic,
                              axis=2)

        distribution.append(flipped_isc)

        # Update random state for next iteration
        random_state = np.random.RandomState(prng.randint(0, MAX_RANDOM_SEED))

    # Convert distribution to numpy array
    distribution = np.vstack(distribution)

    # Get p-value for actual median from shifted distribution
    p = p_from_null(observed, distribution,
                    side='two-sided', exact=False,
                    axis=0)

    return observed, p, distribution
