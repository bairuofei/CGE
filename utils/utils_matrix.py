import scipy
import numpy as np
import networkx as nx

from utils.constants import EIG_TH

def get_d_opt(A):
    # Solves a standard or generalized eigenvalue problem for a complex Hermitian or real symmetric matrix.
    eigv2 = scipy.linalg.eigvalsh(A)
    if np.iscomplex(eigv2.any()):
        print("Error: Complex Root")
    n = np.size(A, 0)
    eigv = eigv2[eigv2 > EIG_TH] # Only select eigenvalues larger than EIG_TH (1e-6)
    return np.exp(np.sum(np.log(eigv)) / n) # Avoid overflow and do normalization

