import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from numpy import linspace, arange, log, vectorize, pi, zeros
from qutip import tensor, basis
from utils import get_ion_state_generators
from numpy.linalg import norm


class OptimizationRecord:
    """
    Storage for an optimzation result
    """

    def __init__(self, setup, optim_result):
        self.setup = setup
        self.optim_result = optim_result

    def plot_final_dynamics(self):
        self.plot_dynamics(self.optim_result.allvecs[-1])

    def plot_dynamics(self, param_vec):
        """
        Plot the fidelity and coefficient functions during the gate time.
        """
        setup = self.setup
        init_state = setup.init_state
        target_state = setup.target_state
        num_focks = setup.num_focks

        fc, fr, fb = setup.get_complex_strengths_func(param_vec)

        g, e = get_ion_state_generators(num_focks)
        e_ops = []
        for n in range(num_focks):
            e_ops.append(g(n).proj())  # projection operator for |gn>
            e_ops.append(e(n).proj())  # projection operator for |en>
        e_ops.append(target_state.proj())  # projection operator for fidelity

        times = linspace(0, 1, 500)
        expect = setup.observe(init_state, param_vec, times, e_ops)

        old_num_focks = num_focks
        for i in range(num_focks):
            if max(max(expect[2 * (num_focks - 1)]), max(expect[2 * (num_focks - 1) + 1])) > 1e-4:
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
            axes[1].grid()

        # Ket notation tick labels
        axes[1].set_yticks(arange(0, num_focks) + 0.5)
        axes[1].set_yticklabels(['|{}⟩'.format(n) for n in range(num_focks)])

        # laser strengths
        axes[2].plot(times, abs(vectorize(fc)(times)) / pi, 'g', alpha=0.7)
        axes[2].plot(times, abs(vectorize(fr)(times)) / pi, 'r', alpha=0.7)
        axes[2].plot(times, abs(vectorize(fb)(times)) / pi, 'b', alpha=0.7)

        axes[2].fill_between(times, abs(vectorize(fc)(times)) / pi, zeros(len(times)),
                             facecolor='g', alpha=0.2)
        axes[2].fill_between(times, abs(vectorize(fr)(times)) / pi, zeros(len(times)),
                             facecolor='r', alpha=0.2)
        axes[2].fill_between(times, abs(vectorize(fb)(times)) / pi, zeros(len(times)),
                             facecolor='b', alpha=0.2)

        axes[2].set_ylabel('strengths/pi')

        # sticks at step boundaries
        for i in range(len(axes)):
            axes[i].set_xticks(arange(min(times), max(times), 1 / setup.num_steps))

        # grids
        axes[0].grid()
        axes[1].grid()
        axes[2].grid()

        fig.show()

    def plot_history(self, plot_grad=True):
        setup = self.setup
        func_history = []
        grad_history = []

        for param_vec in self.optim_result.allvecs:
            func_history.append(setup.target_func(param_vec, setup.init_state, setup.target_state))
            grad_history.append(norm(setup.gradient(param_vec, setup.init_state, setup.target_state)))

        fig, axes = plt.subplots(2, 1, sharex='col')
        ax1, ax2 = axes

        ax1.plot(range(len(func_history)), func_history)
        ax1.scatter(range(len(func_history)), func_history, marker='x')
        ax2.plot(range(len(grad_history)), grad_history)
        ax2.scatter(range(len(grad_history)), grad_history, marker='x')

        ax1.set_ylabel('target function')
        ax2.set_ylabel('gradient norm')
        ax2.set_xlabel('iteration')
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax1.grid()
        ax2.grid()
        fig.show()
