#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:42:46 2019

@author: kin
"""    

class Parameterization:
    def __init__(self):
        pass

class FourierParameterization:
    def __init__(self, num_terms):
        self.num_terms = num_terms
    
    def coef_functions(self, p):
        """
        Return three functions fc, fr, fb based on the p given 
        """
        pass
    
    def init_parameters(self):
        """
        Return initial parameters for optimization
        """
        pass

class Fock:
    def __init__(self, target_state, num_focks, rabi_freq, lamb_dicke, atom_freq, cavity_freq, gate_time):
        self.num_focks = num_focks
        self.target_state = target_state
        self.init_state = tensor(basis(2,1), basis(num_focks, 0))
        self.rabi_freq = rabi_freq
        self.lamb_dicke = lamb_dicke
        self.atom_freq = atom_freq
        self.cavity_freq = cavity_freq
        self.gate_time = gate_time

    def optimize(self, parameterization):
        """
        Optimize the fidelity given the parameterization
        """
    
    def _evolve(self, p):
        """
        Evolve the initial state according to the parameters p
        """