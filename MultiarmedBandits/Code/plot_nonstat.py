import matplotlib.pyplot as plt

num_bandits = 10
q_mean = np.random.normal(0, 1, num_bandits)
steps = 10000

# Call the multiarmed_bandit_eps_greedy function to get the data for plotting
avg_reward_nonstat_greedy, opt_ratio_nonstat_greedy = multiarmed_bandit_eps_greedy_nonstat(num_bandits, q_mean, steps, eps=0.1, s=0.05)
avg_reward_nonstat, opt_ratio_nonstat = multiarmed_bandit_mbandits(num_bandits, q_mean, steps, eps=0.1, s=0.05, alpha=0.1)

# Create a figure and axes for the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
ttl = "Non-stationary multi-armed bandit problem with 10 bandits and swap rate $s=0.05$"
fig.suptitle(ttl, fontsize=16, fontweight='bold')

# Plot the average reward data
axes[0].plot(avg_reward_nonstat, linewidth=1.2, color='blue', label=r"Weighted-average with $\varepsilon=0.1$ and $\alpha=0.1$")
axes[0].plot(avg_reward_nonstat_greedy, linewidth=1.2, color='red', label=r"Sample-average $\varepsilon$-greedy with $\varepsilon=0.1$")
axes[0].set_xlabel('Steps', fontsize=12)
axes[0].set_ylabel('Average reward', fontsize=12)
axes[0].legend(loc='best', fontsize=10)

# Plot the optimal action ratio data
axes[1].plot(opt_ratio_nonstat, linewidth=1.2, color='blue', label=r"Weighted-average with $\varepsilon=0.1$ and $\alpha=0.1$")
axes[1].plot(opt_ratio_nonstat_greedy, linewidth=1.2, color='red', label=r"Sample-average with $\varepsilon$-greedy with $\varepsilon=0.1$")
axes[1].set_xlabel('Steps', fontsize=12)
axes[1].set_ylabel('Optimal action ratio', fontsize=12)
axes[1].legend(loc='best', fontsize=10)

# Adjust layout and padding
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the plot
plt.savefig('eps_greedy_nonstat.png')

# Show the plot
plt.show()