import matplotlib.pyplot as plt
from focks.interaction import LaserInteraction
from typing import List
from qutip import Qobj
from numpy import linspace, zeros, vectorize, arange, pi
import numpy as np
import matplotlib.gridspec as gridspec

EPS = np.finfo(float).eps
SQEPS = np.finfo(float).eps ** 0.5

def piecewise_constant_dynamics(interaction: LaserInteraction,
                                pulse_sequence: List,
                                init_state: Qobj,
                                target_state: Qobj=None):
    """
    Plot the fidelity and coefficient functions during the gate time.
    """
    e, g = interaction.eigenstate_generators
    num_focks = interaction.num_focks
    num_steps = len(pulse_sequence)

    fc_list = [pulse[0] for pulse in reversed(pulse_sequence)]
    fr_list = [pulse[1] for pulse in reversed(pulse_sequence)]
    fb_list = [pulse[2] for pulse in reversed(pulse_sequence)]

    fc = lambda t: fc_list[np.mod(int(t), num_steps)]
    fr = lambda t: fr_list[np.mod(int(t), num_steps)]
    fb = lambda t: fb_list[np.mod(int(t), num_steps)]

    amps = [fc, fr, fb]

    e_ops = []
    for n in range(num_focks):
        e_ops.append(g(n).proj())  # projection operator for |gn>
        e_ops.append(e(n).proj())  # projection operator for |en>
    e_ops.append(target_state.proj())  # projection operator for fidelity

    times = linspace(0, num_steps, 1000)
    expect = interaction.monitor(init_state, amps, times, e_ops)

    old_num_focks = num_focks
    for i in range(num_focks):
        if max(max(expect[2 * (num_focks - 1)]), max(expect[2 * (num_focks - 1) + 1])) > SQEPS:
            break
        num_focks -= 1

    print('num_focks truncated from {} to {}'.format(old_num_focks, num_focks))

    gs = gridspec.GridSpec(5, 1)
    fig = plt.figure()

    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1:4]), fig.add_subplot(gs[4])]
    axes[2].get_shared_x_axes().join(axes[2], axes[0])
    axes[0].set_xticklabels([])
    axes[2].get_shared_x_axes().join(axes[2], axes[1])
    axes[1].set_xticklabels([])

    # infidelity (log)
    axes[0].plot(times, 1 - expect[-1])
    axes[0].set_yscale('log')
    axes[0].set_xlim(min(times), max(times))
    axes[0].set_ylabel('infidelity')

    g_c = 'tab:blue'
    e_c = 'tab:orange'
    b_c = 'gray'
    for n in range(num_focks):

        axes[1].plot(times, expect[2 * n] + n, color=g_c, alpha=1.0, label='ground')
        axes[1].plot(times, n + 1 - expect[2 * n + 1], color=e_c, alpha=1.0, label='excited')

        axes[1].fill_between(times, expect[2 * n] + n, zeros(len(times)) + n,
                             facecolor=g_c, alpha=0.7, label='ground')
        axes[1].fill_between(times, n + 1 - expect[2 * n + 1], zeros(len(times)) + n + 1,
                             facecolor=e_c, alpha=0.7, label='excited')
        if n % 2 == 0:
            axes[1].fill_between(times, zeros(len(times)) + n, zeros(len(times)) + n + 1,
                                 facecolor=b_c, alpha=0.15)

        axes[1].axhline(n, color='k')

        axes[1].set_ylim(0, num_focks)

    # Ket notation tick labels
    axes[1].set_yticks(arange(0, num_focks) + 0.5)
    axes[1].set_yticklabels(['|{}⟩'.format(n) for n in range(num_focks)])

    # laser strengths
    time_list = range(0, num_steps + 1)
    fc_plot = [0]
    fr_plot = [0]
    fb_plot = [0]
    time_plot = [0]

    for i in range(num_steps):
        fc_plot.append(abs(fc_list[i]))
        fc_plot.append(abs(fc_list[i]))
        fr_plot.append(abs(fr_list[i]))
        fr_plot.append(abs(fr_list[i]))
        fb_plot.append(abs(fb_list[i]))
        fb_plot.append(abs(fb_list[i]))
        time_plot.append(i)
        time_plot.append(i+1)

    fc_plot.append(0)
    fr_plot.append(0)
    fb_plot.append(0)
    time_plot.append(num_steps)

    axes[2].plot(time_plot, fc_plot, 'g', alpha=0.7)
    axes[2].plot(time_plot, fr_plot, 'r', alpha=0.7)
    axes[2].plot(time_plot, fb_plot, 'b', alpha=0.7)

    axes[2].fill_between(time_plot, fc_plot, zeros(len(time_plot)),
                         facecolor='g', alpha=0.2)
    axes[2].fill_between(time_plot, fr_plot, zeros(len(time_plot)),
                         facecolor='r', alpha=0.2)
    axes[2].fill_between(time_plot, fb_plot, zeros(len(time_plot)),
                         facecolor='b', alpha=0.2)

    axes[2].set_ylabel('strengths/pi')
    axes[2].set_ylim((0, None))

    # sticks at step boundaries
    for i in range(len(axes)):
        axes[i].set_xticks(arange(min(times), max(times)+1))

    # grids
    axes[0].grid()
    axes[1].grid(axis='x')
    axes[2].grid()

    show_plot = True
    save_plot = False
    if show_plot:
        fig.show()

    if save_plot:
        fig.tight_layout()
        fig.savefig(os.path.join(self._dirpath, 'dynamics.pdf'))
