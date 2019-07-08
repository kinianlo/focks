#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:33:28 2019

@author: kin
"""

from qutip import basis, tensor

class IonStateFactory:
    """
    Provide a concise way to generate ion trap states. 
    """
    def __init__(self, num_focks):
        self.num_focks = num_focks
        
    def exd(self, n_fock):
        """
        Return an excited state in the n_fock Fock state 
        """
        self._verify_n_fock(n_fock)
        return tensor(basis(2, 0), basis(self.num_focks, n_fock))
    
    def gnd(self, n_internal, n_fock):
        """
        Return a ground state in the n_fock Fock state 
        """
        self._verify_n_fock(n_fock)
        return tensor(basis(2, 1), basis(self.num_focks, n_fock))
    
    def _verify_n_fock(self, n_fock):
        if n_fock >= self.num_focks:
            raise ValueError('n_fock must be less then num_focks')
        