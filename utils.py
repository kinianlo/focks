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


def plot_dynamics(param_vec, parameterization, init_state, target_state,
                  num_focks):
    """
    Plot the fidelity and coefficient functions during the gate time.
    """
    fc, fr, fb = parameterization.get_complex_strengths_func(param_vec)

    e_ops = []
    for n in range(num_focks):
        e_ops.append(tensor(basis(2, 1), basis(num_focks, n)).proj())  # |0>
        e_ops.append(tensor(basis(2, 0), basis(num_focks, n)).proj())  # |1>
    e_ops.append(target_state.proj())

    times = linspace(0, 1, 500)
    expect = parameterization.observe(init_state, param_vec, times, e_ops)

    old_num_focks = num_focks
    for i in range(num_focks):
        if max(max(expect[2*(num_focks-1)]), max(expect[2*(num_focks-1)+1])) > 1e-4:
            break
        num_focks -= 1

    print('num_focks truncated from {} to {}'.format(old_num_focks, num_focks))

    gs = gridspec.GridSpec(5, 1)
    fig = plt.figure(figsize=(8, 6))

    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1:4]), fig.add_subplot(gs[4])]
    axes[2].get_shared_x_axes().join(axes[2], axes[0])
    axes[0].set_xticklabels([])
    axes[2].get_shared_x_axes().join(axes[2], axes[1])
    axes[1].set_xticklabels([])

    # infidelity (log)
    axes[0].plot(times, log(1-expect[-1]))
    axes[0].set_xlim(min(times), max(times))
    axes[0].set_ylabel('log infidelity ')

    g_c = 'tab:blue'
    e_c = 'tab:orange'
    b_c = 'gray'
    for n in range(num_focks):
        axes[1].axhline(n, color='k')

        axes[1].plot(times, expect[2 * n]+n, color=g_c, alpha=1.0, label='ground')
        axes[1].plot(times, n+1-expect[2 * n + 1], color=e_c, alpha=1.0, label='excited')

        axes[1].fill_between(times, expect[2 * n]+n, zeros(len(times))+n,
                             facecolor=g_c, alpha=0.7, label='ground')
        axes[1].fill_between(times, n+1-expect[2 * n + 1], zeros(len(times))+n+1,
                             facecolor=e_c, alpha=0.7, label='excited')
        if n % 2 == 0:
            axes[1].fill_between(times, zeros(len(times)) + n, zeros(len(times)) + n + 1,
                                 facecolor=b_c, alpha=0.15)

        axes[1].set_ylim(0, num_focks)
        axes[1].grid()

    # Ket notation tick labels
    axes[1].set_yticks(arange(0,num_focks)+0.5)
    axes[1].set_yticklabels(['|{}⟩'.format(n) for n in range(num_focks)])

    # laser strengths
    axes[2].plot(times, abs(vectorize(fc)(times)) / pi, 'g', alpha=0.7)
    axes[2].plot(times, abs(vectorize(fr)(times)) / pi, 'r', alpha=0.7)
    axes[2].plot(times, abs(vectorize(fb)(times)) / pi, 'b', alpha=0.7)
    axes[2].set_ylabel('strengths/pi')

    # sticks at step boundaries
    for i in range(len(axes)):
        axes[i].set_xticks(arange(min(times), max(times), 1/parameterization.num_steps))

    # grids
    axes[0].grid()
    axes[1].grid()
    axes[2].grid()

    fig.show()
