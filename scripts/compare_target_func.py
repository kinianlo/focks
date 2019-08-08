from fock import *

num_repeats = 5

infil_0_mean = []
infil_0_std = []
infil_1_mean = []
infil_1_std = []

n_range = range(1, 5)
for n in n_range:
    num_focks = n * 2 + 5
    num_steps = n + np.mod(n, 2)
    g, e = get_ion_state_generators(num_focks)
    init_state = g(0)
    target_state = g(n)

    param = IonTrapSetup(num_focks, num_steps, 0)
    infil_list = []
    for i in range(num_repeats):
        opt_result = minimize(param.target_func, param.init_param_vec(),
                              args=(init_state, target_state),
                              options={'disp': True})
        infil_list.append(opt_result.fun)
    infil_0_mean.append(np.mean(infil_list))
    infil_0_std.append(np.std(infil_list))

    param = IonTrapSetup(num_focks, num_steps, 1e-1)
    infil_list = []
    for i in range(num_repeats):
        opt_result = minimize(param.target_func, param.init_param_vec(),
                              args=(init_state, target_state),
                              options={'disp': True})
        infil_list.append(opt_result.fun)
    infil_1_mean.append(np.mean(infil_list))
    infil_1_std.append(np.std(infil_list))


fig, axes = plt.subplots(2, 1, sharex=True)
ax1, ax2 = axes

ax1.errorbar(n_range, infil_0_mean, yerr=infil_0_std, label='Phi0')
ax1.errorbar(n_range, infil_1_mean, yerr=infil_1_std, label='Phi0+0.1*Phi1')
ax2.errorbar(n_range, infil_0_mean, yerr=infil_0_std)
ax2.errorbar(n_range, infil_1_mean, yerr=infil_1_std)
ax1.set_yscale('log')
ax1.set_ylabel('final infidelity')
ax2.set_ylabel('final infidelity')
ax2.set_xlabel('|gn⟩')
ax1.grid()
ax2.grid()

fig.show()
