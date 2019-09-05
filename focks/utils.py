from qutip import Qobj
import numpy as np
from numpy import dot, exp, conj, log, zeros
from numpy.linalg import eigh, eigvalsh
import matplotlib.pyplot as plt
from typing import Sequence

def expmh(herm_mat, coef=1, e_vals=None, e_vecs=None):
    """
    Return the exponential of a Hermitian operator (scaled by a complex number)
    Parameters
    ----------
    herm_mat : 2darray
        A Hermitian matrix.
    coef :
        A complex number.

    Returns
    -------
    u : Qobj
        exp(coef * herm_oper)

    """
    if e_vecs is None:
        e_vals, e_vecs = eigh(herm_mat)
    elif e_vals is None:
        e_vals = eigvalsh(herm_mat)
    u = dot(e_vecs, (exp(conj(coef)*e_vals) * e_vecs).T.conj())
    return u

def visualize_matrix(mat, log_scale=False):
    """
    Visualise a 2d matrix

    Parameters
    ----------
    mat: 2d array

    """
    if mat.dtype == complex:
        mat = np.abs(mat)
    if log_scale:
        mat = log(mat)
    plt.matshow(mat)
    plt.colorbar()
    plt.show()

def expand_state(state: Qobj, dims):
    num_focks = dims[0][1]
    old_num_focks = state.dims[0][1]
    new_full = zeros((2 * num_focks, 1), dtype=complex)
    new_full[0:old_num_focks, 0] = state.full()[0:old_num_focks, 0]
    new_full[num_focks:num_focks + old_num_focks, 0] = state.full()[old_num_focks: , 0]
    return Qobj(new_full, dims=dims)

def transform(U, H):
    return dot(U, dot(H, U.conj().T))

def relu(x, shift=0):
    return (x-shift) * (x-shift > 0)

def relu_der(x, shift=0):
    return x-shift > 0