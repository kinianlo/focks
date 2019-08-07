from fock import *

appr_eps = (np.finfo(float).eps)**0.5

num_repeats = 10
grad_mag_mean = []
grad_mag_std = []

n_range = range(1, 9)
for n in n_range:
    num_focks = n * 2 + 5
    num_steps = n + np.mod(n, 2)
    g, e = get_ion_state_generators(num_focks)
    init_state = g(0)
    target_state = g(n)
    param = PiecewiseConstant(num_focks, num_steps, 1e-1)
    grad_mag_list = []
    for i in range(num_repeats):
        #grad_mag = np.sum(param.gradient(param.init_param_vec(), init_state, target_state) ** 2) ** 0.5
        grad_mag = np.sum(approx_fprime(param.init_param_vec(), param.target_func, appr_eps,
                                        init_state, target_state) ** 2) ** 0.5
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

fig.show()
