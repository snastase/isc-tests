import numpy as np


# Simulate correlated data by sampling from multivariate normal
def correlated_data(n_trs, n_subjects, r, mean=0, var=1, seed=None):
    
    mean = np.full(n_subjects, mean)
    cov = np.full((n_subjects, n_subjects), r)
    np.fill_diagonal(cov, var)
    
    np.random.seed(seed)
    data = np.random.multivariate_normal(mean, cov, size=n_trs)
    
    return data
