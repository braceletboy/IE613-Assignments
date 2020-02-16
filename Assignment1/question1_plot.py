# !/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


# ########################## Plotting function ##########################
def regret_plotting(regret, cases, plotting_info):
    # ### Average Regret ### 
    avg_regret = np.mean(regret, axis=1)
    # print avg_regret

    # ### Confidence interval ### 
    error = []
    freedom_degree = len(regret[0]) - 1
    for c in range(len(cases)):
        error.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(np.array(regret[c])))
    # print error

    # ### Plotting ### 
    plt.errorbar(cases, avg_regret, error, color='g')
    plt.plot(cases, avg_regret, 'g')

    # ### Plotting details ### 
    plt.xlabel(plotting_info[0], fontsize=15)
    plt.ylabel(plotting_info[1], fontsize=15)
    plt.title(plotting_info[2], fontsize=15)

    #  Saving file
    plt.savefig(plotting_info[3], bbox_inches='tight')
    plt.close()


# ############################## Algorithm ##############################
def online_dummy_algo(parameters):
    loss_vector     = parameters[0]
    optimal         = parameters[1]
    c               = parameters[2]

    T = len(loss_vector)
    K = len(loss_vector[0])

    instantaneous_regret = []
    base_arm = int(c)
    arms_list = [base_arm-1, base_arm, base_arm+1]
    arm = arms_list[np.random.choice(3, 1)[0]]
    for t in range(T):
        # r_t = mean_values[I_t] - mean_values[i_star]
        instantaneous_regret.append(loss_vector[t, arm])

    return sum(instantaneous_regret)


# ########################### Data generation ###########################
# Experiment parameter
time_horizon    = 1000
runs            = 20

# Generating loss vectors
delta = 0.1
# np.random.seed(100)
loss_vector_1_8     = np.random.binomial(1, 0.5, size=(time_horizon, 8))
loss_vector_9       = np.random.binomial(1, 0.5-delta, size=(time_horizon, 1))
loss_vector_10_T1   = np.random.binomial(1, 0.5+delta, size=(int(time_horizon/2), 1))
loss_vector_10_T2   = np.random.binomial(1, 0.5-2*delta, size=(int(time_horizon/2), 1))

# Joining losses for arm 10
loss_vector_10      = np.vstack((loss_vector_10_T1, loss_vector_10_T2))

# Combining losses for all arms
loss_vector_1_9     = np.hstack((loss_vector_1_8, loss_vector_9))
loss_vectors        = np.hstack((loss_vector_1_9, loss_vector_10))

# print loss_vectors

# Different c_value
c_ini = 1
c_gap = 1
c_values = [float(format(c_ini + i*c_gap, '.2f')) for i in range(5)]
# print c_values

# ### Runnging algorithm ####
c_value = 0.1
optimal_arm = 9
alg_parameters = [loss_vectors, optimal_arm, c_value]

# Regret
algos_regret = []
for c in range(len(c_values)):
    alg_parameters[2] = c_values[c]
    case_regret = []
    for _ in range(runs):     
        iter_regret = online_dummy_algo(alg_parameters)
        case_regret.append(iter_regret)
    
    # For saving regret data, uncomment below 2 lines
    # file_to_save =  "totalRegret_" + str(c+1) +".csv"
    # np.savetxt(file_to_save, np.array(case_regret), '%5.4f', delimiter=",")

    algos_regret.append(case_regret)

# print(algos_regret)

# ########################### Plotting details ##########################
xlabel = r'$\eta$'
ylabel = "Regret"
title = "Template"
file_to_save = 'template.png'
plotting_parameters = [xlabel, ylabel, title, file_to_save]

# Regret Plotting
regret_plotting(algos_regret, c_values, plotting_parameters)