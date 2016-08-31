"""Useful kernels for molecular comparisons.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import cython
cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def tanimoto_kernel(X, Y):
    """Tanimoto kernel for use in kernel methods.

    Data must be binary. This is not checked. Sparse matrices are converted
    to dense arrays before computation.

    Parameters
    ----------
    X : ndarray or csr_matrix of np.float64
        MxP bitvector array for M mols and P bits
    Y : ndarray or csr_matrix of np.float64
        NxP bitvector array for N mols and P bits

    Returns
    ----------
    ndarray of np.float64
        Tanimoto similarity between X and Y fingerprints

    References
    ----------
    ..[1] L. Ralaivola, S.J. Swamidass, H. Saigo, P. Baldi."Graph kernels for
          chemical informatics." Neural Networks. 2005. 18(8): 1093-1110.
          doi: 10.1.1.92.483
    """
    try:
        X = X.toarray()
        Y = Y.toarray()
    except AttributeError:
        tanimoto = tanimoto_kernel_dense_(np.asarray(X, dtype=DTYPE),
                                          np.asarray(Y, dtype=DTYPE))

    return np.asarray(tanimoto, dtype=DTYPE)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef DTYPE_t[:, :] tanimoto_kernel_dense_(np.ndarray[DTYPE_t, ndim=2] X,
                                           np.ndarray[DTYPE_t, ndim=2] Y):
    cdef:
        np.ndarray[DTYPE_t, ndim=2] tanimoto, Xbits, Ybits, XYbits
        int i, j, k
        
    Xbits = X.sum(axis=1, keepdims=True)
    Ybits = Y.sum(axis=1, keepdims=True)
    XYbits = X.dot(Y.T)
    tanimoto = XYbits / (Xbits + Ybits.T - XYbits)

    return tanimoto
