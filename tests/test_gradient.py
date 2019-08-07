from fock import *

"""
Compare the analytic graidents calculated in paramterization.gradient()
to the numerical gradient given by scipy (approx_fprime).
Consider the target state |gn>, where n goes from 1 to 10.
The fock space is truncated at the 2n+5 level.
Number of steps is n for even n, and n+1 for odd n.
This is the minimum steps for the analytic algorithm.
"""
num_repeats = 10
err_mag_mean = []
err_mag_std = []
frac_err_mean = []
frac_err_std = []

n_range = range(1, 7)
for n in n_range:
    num_focks = n * 2 + 5
    num_steps = n + np.mod(n, 2)
    g, e = get_ion_state_generators(num_focks)
    init_state = g(0)
    target_state = g(n)
    param = PiecewiseConstant(num_focks, num_steps)
    err_mag_list = []
    frac_err_list = []
    for i in range(num_repeats):
        param_vec = param.init_param_vec()
        grad_anal = param.gradient(param_vec, init_state, target_state)
        grad_appr = approx_fprime(param_vec, param.target_func,
                                  appr_eps, init_state, target_state)
        err_mag = np.sum((grad_anal - grad_appr) ** 2) ** 0.5
        frac_err_list.append(err_mag/np.sum(grad_appr**2)**0.5)
        err_mag_list.append(err_mag)
    err_mag_mean.append(np.mean(err_mag_list))
    err_mag_std.append(np.std(err_mag_list))
    frac_err_mean.append(np.mean(frac_err_list))
    frac_err_std.append(np.std(frac_err_list))
fig, axes = plt.subplots(2, 1, sharex=True)
ax1, ax2 = axes

ax1.errorbar(n_range, err_mag_mean, yerr=err_mag_std)
ax2.errorbar(n_range, frac_err_mean, yerr=frac_err_std)

ax1.set_yscale('log')
ax2.set_yscale('log')

ax1.set_ylabel('norm of gradient error')
ax2.set_ylabel('fraction of gradient error')
ax2.set_xlabel('|gn⟩')
ax1.grid()
ax2.grid()

fig.show()