#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:01:37 2019

@author: kin
"""

import numpy as np
from numpy.linalg import eigh, eig
from scipy.sparse.linalg import eigs
from scipy.optimize import minimize, approx_fprime
from qutip import tensor, basis, qeye, sigmap, sigmam, create, destroy, mesolve, Qobj, expect
import matplotlib.pyplot as plt
from utils import get_ion_state_generators, transform
from math import factorial

class Parameterization:
    """
    The abstract class for parameterizations of coefficient functions.
    Coefficient functions refers to the three time-dependent functions
    which control the Hamiltonian
    H = fc*sigam_+ + fr*sigma_+ a^dagger + fb*sigma_+ a + h.c..
    """
    def __init__(self, num_focks):
        self.num_focks = num_focks

        Sp = tensor(sigmap(), qeye(num_focks))
        Sm = tensor(sigmam(), qeye(num_focks))
        a = tensor(qeye(2), destroy(num_focks))
        adag = tensor(qeye(2), create(num_focks))

        self._ctrl_hamils = [Sp+Sm, Sp*a+Sm*adag, Sp*adag+Sm*a,
                             1j*(Sp-Sm), 1j*(Sp*a-Sm*adag), 1j*(Sp*adag-Sm*a)]
        self._adaga = adag*a

    def init_param_vec(self):
        """
        Return an initial parameter vector for optimization.
        """
        raise NotImplementedError

    def unpack_param_vec(self, param_vec):
        """
        Unpack a parameter vector into a more manageable form
        """
        raise NotImplementedError

    def strengths(self, param_vec):
        """
        Return a tuple of three functions (fc, fr, fb) based on the parameter
        vector param_vec.
        The functions must take an t argument and a list of arguments *arg,
        e.g. fc = lambda t, *arg : ***function_output***.
        The *arg is required by QuTip for additional arguments (if any).
        """
        raise NotImplementedError

    def evolve(self, init_state, param_vec):
        """
        Return the evolved state.
        """
        raise NotImplementedError

    def observe(self, init_state, param_vec, times, e_ops=[]):
        """
        Evolve the init_state given the param_vec for a duration of gate_time.
        Return a qutip.solver.Result objection
        """
        raise NotImplementedError

    def parse_parameters(self, param_vec):
        """
        Print the parameters in a human readable form.
        """
        raise NotImplementedError

class PiecewiseConstant(Parameterization):
    def __init__(self, num_focks, num_steps):
        Parameterization.__init__(self, num_focks)
        self.num_steps = num_steps

    def unpack_param_vec(self, param_vec):
        """
        param_vec is a 1-d array of all the parameters.
        This method seperate the param_vec into num_steps rows and each row
        contains the 6 control amplitdues of a time step
        """
        return param_vec.reshape((self.num_steps,-1))

    def init_param_vec(self):
        """
        Return an arrany of 6*num_steps randum numbers in [-0.5, 0.5]
        """
        return (np.random.rand(6*self.num_steps)-0.5)

    def get_complex_strengths_func(self, param_vec):
        """
        Return three functions of time (fc, fr, fb). Each function returns the
        complex amplitude of a colour (carrier, red or blue) at time t.
        """
        amps = self.unpack_param_vec(param_vec)
        fc_list = amps[:, 0] + 1j * amps[:, 3]
        fr_list = amps[:, 1] + 1j * amps[:, 4]
        fb_list = amps[:, 2] + 1j * amps[:, 5]

        fc = lambda t, *_: fc_list[np.mod(int(t*self.num_steps), self.num_steps)]
        fr = lambda t, *_: fr_list[np.mod(int(t*self.num_steps), self.num_steps)]
        fb = lambda t, *_: fb_list[np.mod(int(t*self.num_steps), self.num_steps)]

        return fc, fr, fb

    def get_amps_func(self, param_vec):
        amps = self.unpack_param_vec(param_vec)

        func_list = []
        for i in range(6):
            func_list.append(lambda t, *_, ii=i: amps.T[ii][np.mod(int(t*self.num_steps), \
                                                                   self.num_steps)])

        return func_list


    def evolve(self, init_state, param_vec):
        """
        Return the state after one unit of time
        """
        amps = self.unpack_param_vec(param_vec)

        state = init_state
        for j in range(self.num_steps):
            H = 0
            for i in range(6):
                H += amps[j, i] * self._ctrl_hamils[i]
            state = ((-1j/self.num_steps)*H).expm() * state
        return state

    def observe(self, init_state, param_vec, times, e_ops=[]):
        amps_func = self.get_amps_func(param_vec)
        H = []
        for i in range(6):
            H.append([self._ctrl_hamils[i], amps_func[i]])
        result = mesolve(H, init_state, times, [], e_ops)

        if e_ops == []:
            return result.states
        else:
            return result.expect

    def target_func(self, param_vec, init_state, target_state):

        evolved_state = self.evolve(init_state, param_vec)
        Phi0 = 1 - abs(target_state.overlap(evolved_state))**2
        Phi1 = (expect(self._adaga, evolved_state)-expect(self._adaga, target_state))**2
        return Phi0 + Phi1

    def gradients(self, param_vec, init_state, target_state, quick=False):
        amps = self.unpack_param_vec(param_vec)
        evolved_state = self.evolve(init_state, param_vec)
        overlap = evolved_state.overlap(target_state)
        left = target_state
        right = evolved_state
        grads = np.empty((self.num_steps, 6))

        DT = 1/self.num_steps
        for j in reversed(range(self.num_steps)):
            H = 0
            for i in range(6):
                H += amps[j, i] * self._ctrl_hamils[i]
            Udag = ((1j*DT)*H).expm()

            if not quick:
                # Diagonalise H
                w, v = eigh(H.full())
                # Evaluate the integral for "average" Hi on a "for each matrix element fashion"
                L = w.reshape((-1, 1)) - w
                E = 0
                for k in range(20):
                    E += 1/factorial(k+1) * (-1j*L*DT)**k
                E *= DT

            for i in range(6):
                # prepare the "average" control Hamil Hi
                Hi = self._ctrl_hamils[i]
                if not quick:
                    exp_Hi = Qobj(transform(v, E*transform(v.conj().T, Hi.full())), dims=Hi.dims)
                else:
                    exp_Hi = Hi

                grads[j, i] = -2*np.imag(left.overlap(exp_Hi*right) * overlap)

            left = Udag * left
            right = Udag * right

        return grads.reshape((-1))

    def parse_parameters(self, param_vec):
        # unpack the parameter vector
        amps = self.unpack_param_vec(param_vec)
        Ac = amps[:, 0] + 1j * amps[:, 3]
        Ar = amps[:, 1] + 1j * amps[:, 4]
        Ab = amps[:, 2] + 1j * amps[:, 5]

        output = '\nPiecewiseParameterization\n'
        output = output + 'fc(t) = \\sum_j Ac_j * Heaviside(t-t_j) * Heaviside(t_{j+1}-t)\n'
        output = output + 'fr(t) = \\sum_j Ar_j * Heaviside(t-t_j) * Heaviside(t_{j+1}-t)\n'
        output = output + 'fb(t) = \\sum_j Ab_j * Heaviside(t-t_j) * Heaviside(t_{j+1}-t)\n'

        output = output + '\ncarrier transition:\n'
        output = output + 't_j\tt_{j+1}\tAc\t\tAr\t\tAb\n'
        for j in range(self.num_steps):
            t_j = j/self.num_steps
            t_jp1 = (j+1)/self.num_steps
            output = output + '{:.3}\t{:.3}\t{:.3e}\t{:.3e}\t{:.3e}\n'.format(t_j, t_jp1, Ac[j], Ar[j], Ab[j])


def plot_dynamics(param_vec, parameterization, init_state, target_state, \
                  num_focks):
    """
    Plot the fidelity and coefficient functions during the gate time.
    """
    fc, fr, fb = parameterization.get_complex_strengths_func(param_vec)

    e_ops = []
    for n in range(num_focks):
        e_ops.append(tensor(basis(2,1), basis(num_focks, n)).proj()) # |0>
        e_ops.append(tensor(basis(2,0), basis(num_focks, n)).proj()) # |1>
    e_ops.append(target_state.proj())

    times = np.linspace(0, 1, 500)
    expect = parameterization.observe(init_state, param_vec, times, e_ops)

    fig, axes = plt.subplots(num_focks+2, 1, sharex=True)

    for n in range(num_focks):
        axes[num_focks-1-n].plot(times, expect[2*n], alpha=0.7, label='ground')
        axes[num_focks-1-n].plot(times, expect[2*n+1], alpha=0.7, label='excited')

        axes[num_focks-1-n].set_ylabel('|{}⟩ '.format(n), rotation='horizontal')
        axes[num_focks-1-n].set_ylim(-0.1, 1.1)
        axes[num_focks-1-n].grid()

    axes[0].legend()

    axes[-2].plot(times, np.abs(np.vectorize(fc)(times))/np.pi, 'g', alpha=0.7)
    axes[-2].plot(times, np.abs(np.vectorize(fr)(times))/np.pi, 'r', alpha=0.7)
    axes[-2].plot(times, np.abs(np.vectorize(fb)(times))/np.pi, 'b', alpha=0.7)

    axes[-2].set_ylabel('strength/pi')
    axes[-2].grid()

    axes[-1].semilogy(times, 1-expect[-1])
    axes[-1].set_ylabel('target_func')
    axes[-1].set_xlabel('time')
    axes[-1].grid()

    if isinstance(parameterization, PiecewiseConstant):
        axes[-1].xaxis.set_ticks(np.linspace(0, 1, parameterization.num_steps+1))

def plot_fidelity_num_steps(target_state, num_steps_range, num_repeats):
    num_focks = max(num_steps_range)+1
    g, e = get_ion_state_generators(num_focks)
    init_state = g(0)

    all_errors = []

    for num_steps in num_steps_range:
        errors = []
        parameterization = PiecewiseConstant(num_focks, num_steps)
        for i in range(num_repeats):
            def dist(param_vec):
                evolved_state = parameterization.evolve(init_state, param_vec)
                return 1-np.abs(target_state.overlap(evolved_state))**2
            opt_result = minimize(dist, parameterization.init_param_vec(),
                                      options={'disp':True})
            opt_param_vec = opt_result.x
            evolved_state = parameterization.evolve(init_state, opt_param_vec)
            errors.append(1-np.abs(target_state.overlap(evolved_state))**2)
        all_errors.append(errors)

    fig, ax = plt.subplots(1,1)
    for i in range(len(num_steps_range)):
        ax.scatter([num_steps_range[i]]*num_repeats, all_errors[i])

def grad_error(num_steps=4, num_focks=20, eps=1e-8):
    parameterization = PiecewiseConstant(num_focks, num_steps)
    param_vec = parameterization.init_param_vec()

    g, e = get_ion_state_generators(num_focks)
    init_state = g(0)
    target_state = (e(0)+g(0)).unit()

    grad = parameterization.gradients(param_vec, init_state, target_state)
    approx = approx_fprime(param_vec, parameterization.target_func, eps, init_state, target_state)
    grad = parameterization.gradients(param_vec, init_state, target_state)
#    print('gradients')
#    print(grad.reshape(-1, 6))
#    print('approx')
#    print(approx.reshape(-1,6))
    return np.sum((grad-approx)**2)**0.5/np.sum((approx)**2)**0.5

def plot_grad_error():
    num_steps_list = np.arange(1, 5)
    num_repeats = 8
    error_mean_list = np.empty(len(num_steps_list))
    error_std_list = np.empty(len(num_steps_list))

    for j in range(len(num_steps_list)):
        errors = np.empty(num_repeats)
        for i in range(num_repeats):
            errors[i] = grad_error(num_steps_list[j], num_steps_list[j]*2+10)
        error_mean_list[j] = np.mean(errors)
        error_std_list[j] = np.std(errors)

    fig, ax = plt.subplots(1, 1)
    error_std_list = np.empty(len(num_steps_list))
    ax.errorbar(num_steps_list, error_mean_list, yerr=error_std_list)
    ax.set_yscale('log')

def plot_init_grad():
    num_repeats = 100
    grad_mag_mean = []
    grad_mag_std = []

    n_range = range(1, 15)
    for n in n_range:
        num_focks = n*2+5
        num_steps = n + np.mod(n, 2)
        g, e = get_ion_state_generators(num_focks)
        init_state = g(0)
        target_state = g(n)
        param = PiecewiseConstant(num_focks, num_steps)
        grad_mag_list = []
        for i in range(num_repeats):
            grad_mag = np.sum(param.gradients(param.init_param_vec(), init_state, target_state)**2)**0.5
            grad_mag_list.append(grad_mag)
        grad_mag_mean.append(np.mean(grad_mag_list))
        grad_mag_std.append(np.std(grad_mag_list))
    fig, axes = plt.subplots(2, 1, sharex=True)
    ax1, ax2 = axes

    ax1.errorbar(n_range, grad_mag_mean, yerr=grad_mag_std)
    ax2.errorbar(n_range, grad_mag_mean, yerr=grad_mag_std)
    ax1.set_yscale('log')
    ax1.set_ylabel('length of gradient')
    ax2.set_ylabel('length of gradient')
    ax2.set_xlabel('|gn⟩')
    ax1.grid()
    ax2.grid()

def plot_grad_diff():
    """
    Compare the analytic graidents calcuated in paramterization.gradients()
    to the numberical gradients given by scipy (approx_fprime)
    """
    num_repeats = 10
    err_mag_mean = []
    err_mag_std = []
    appr_eps = 1e-8

    n_range = range(1, 10)
    for n in n_range:
        num_focks = n*2+5
        num_steps = n + np.mod(n, 2)
        g, e = get_ion_state_generators(num_focks)
        init_state = g(0)
        target_state = g(n)
        param = PiecewiseConstant(num_focks, num_steps)
        err_mag_list = []
        for i in range(num_repeats):
            param_vec = param.init_param_vec()
            grad_anal = param.gradients(param_vec, init_state, target_state)
            grad_appr = approx_fprime(param_vec, param.target_func,
                                      appr_eps, init_state, target_state)
            err_mag = np.sum((grad_anal-grad_appr)**2)**0.5
            err_mag_list.append(err_mag)
        err_mag_mean.append(np.mean(err_mag_list))
        err_mag_std.append(np.std(err_mag_list))
    fig, axes = plt.subplots(2, 1, sharex=True)
    ax1, ax2 = axes

    ax1.errorbar(n_range, err_mag_mean, yerr=err_mag_std)
    ax2.errorbar(n_range, err_mag_mean, yerr=err_mag_std)
    ax1.set_yscale('log')
    ax1.set_ylabel('length of gradient error')
    ax2.set_ylabel('length of gradient error')
    ax2.set_xlabel('|gn⟩')
    ax1.grid()
    ax2.grid()

##

from time import process_time
NELDER_MEAD = 'nelder-mead'
BFGS = 'BFGS'
POWERLL = 'Powell'

num_focks = 30
g, e = get_ion_state_generators(num_focks)

init_state = g(0)
target_state = (g(5)+g(2)).unit()
num_steps = 7

parameterization = PiecewiseConstant(num_focks, num_steps)

def dist(param_vec):
    evolved_state = parameterization.evolve(init_state, param_vec)
    return 1-np.abs(target_state.overlap(evolved_state))**2

start = process_time()
opt_result = minimize(parameterization.target_func, parameterization.init_param_vec(),
                        jac=parameterization.gradients,
                        args=(init_state, target_state),
                        options={'disp':True}, method=BFGS)
end = process_time()

opt_param_vec = opt_result.x # optimized parameter vector

print('CPU time used: {}'.format(end-start))
parameterization.parse_parameters(opt_param_vec)

#plot_dynamics(opt_param_vec, parameterization, init_state, target_state, num_focks)
