#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:33:28 2019

@author: kin
"""
from qutip import basis, tensor
from numpy import dot, vectorize, abs, linspace, pi, ceil, mod, zeros, arange, log
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_ion_state_generators(num_focks):
    """
    Return a 2-tuple of two functions (ground, excited).
    The excited function generates an excited state in the n-th Fock state.
    The ground function generates an ground state in the n-th Fock state.

    It's cutomary to name the excited and ground functions e and g respectively
    in actual use, for instance:
    e, g = get_ion_state_generators(5)
    To get a |g2> state use g(2)
    To get a |e3> state use e(3)
    """
    return (lambda n: tensor(basis(2, 1), basis(num_focks, n)),
            lambda n: tensor(basis(2, 0), basis(num_focks, n)))


def transform(U, H):
    return dot(U, dot(H, U.conj().T))
