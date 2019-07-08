#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:33:28 2019

@author: kin
"""

from qutip import basis, tensor

GROUND = 1
EXCITED = 0

def ion_state(n_internal, n_fock, num_focks):
    return tensor(basis(2, n_internal), basis(num_focks, n_fock))

# alias
ionst = ion_state
