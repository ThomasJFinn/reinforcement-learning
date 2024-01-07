import matplotlib.pyplot as plt

num_bandits = 10
q_mean = np.random.normal(0, 1, num_bandits)
steps = 5000
opt_init = 6

# Call the epsilon-greedy multiarmed bandit algorithm and the optimistic initial values algorithm
avg_reward_eps, opt_ratio_eps = multiarmed_bandit_eps_greedy(num_bandits, q_mean, steps, 0.01)
avg_reward_eps1, opt_ratio_eps1 = multiarmed_bandit_eps_greedy(num_bandits, q_mean, steps, 0.1)
avg_reward_optimistic, opt_ratio_optimistic = multiarmed_bandit_optimistic(num_bandits, q_mean, steps, 0, opt_init)

# Create a figure and axes for the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
ttl = "Multi-armed bandit problem with 10 choices under a $\\varepsilon$-greedy algorithm and greedy with optimistic initial values"
fig.suptitle(ttl, fontsize=16, fontweight='bold')

# Plot the average reward data
axes[0].plot(avg_reward_eps, linewidth=1.2, label=r"$\varepsilon=0.01$")
axes[0].plot(avg_reward_eps1, linewidth=1.2, color='green', label=r"$\varepsilon=0.1$")
axes[0].plot(avg_reward_optimistic, linewidth=1.2, color='red', label=r"OIV")
axes[0].set_xlabel('Steps', fontsize=12)
axes[0].set_ylabel('Average reward', fontsize=12)
axes[0].legend(loc='best', fontsize=10)

# Plot the optimal action ratio data
axes[1].plot(opt_ratio_eps, linewidth=1.2, label=r"$\varepsilon=0.01$")
axes[1].plot(opt_ratio_eps1, linewidth=1.2, color='green', label=r"$\varepsilon=0.1$")
axes[1].plot(opt_ratio_optimistic, linewidth=1.2, color='red', label=r"OIV")
axes[1].set_xlabel('Steps', fontsize=12)
axes[1].set_ylabel('Optimal action ratio', fontsize=12)
axes[1].legend(loc='best', fontsize=10)

# Adjust layout and padding
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure
plt.savefig('greedy_vs_optimistic.png')

# Show the plot
plt.show()