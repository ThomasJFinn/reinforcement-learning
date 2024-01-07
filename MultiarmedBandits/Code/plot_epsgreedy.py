import matplotlib.pyplot as plt

num_bandits = 10
q_mean = np.random.normal(0, 1, num_bandits)
steps = 1000

# Call the multiarmed_bandit_eps_greedy function to get the data for plotting
avg_reward_vec, opt_ratio_vec = multiarmed_bandit_eps_greedy(num_bandits, q_mean, steps, eps=0.01)
avg_reward_vec1, opt_ratio_vec1 = multiarmed_bandit_eps_greedy(num_bandits, q_mean, steps, eps=0.1)
avg_reward_vec_greedy, opt_ratio_vec_greedy = multiarmed_bandit_eps_greedy(num_bandits, q_mean, steps, eps=0)

# Create a figure and axes for the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
ttl = "Multi-armed bandit problem with 10 choices under $\\varepsilon$-greedy algorithms"
fig.suptitle(ttl, fontsize=16, fontweight='bold')

# Plot the average reward data
axes[0].plot(avg_reward_vec_greedy, linewidth=1.2, color='green', label=r"$\varepsilon=0$")
axes[0].plot(avg_reward_vec, linewidth=1.2, label=r"$\varepsilon=0.01$")
axes[0].plot(avg_reward_vec1, linewidth=1.2, color='red', label=r"$\varepsilon=0.1$")
axes[0].set_xlabel('Steps', fontsize=12)
axes[0].set_ylabel('Average reward', fontsize=12)
axes[0].legend(loc='best', fontsize=10)

# Plot the optimal action ratio data
axes[1].plot(opt_ratio_vec_greedy, linewidth=1.2, color='green', label=r"$\varepsilon=0$")
axes[1].plot(opt_ratio_vec, linewidth=1.2, label=r"$\varepsilon=0.01$")
axes[1].plot(opt_ratio_vec1, linewidth=1.2, color='red', label=r"$\varepsilon=0.1$")
axes[1].set_xlabel('Steps', fontsize=12)
axes[1].set_ylabel('Optimal action ratio', fontsize=12)
axes[1].legend(loc='best', fontsize=10)

# Adjust layout and padding
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the plot
plt.savefig('eps_greedy.png')

# Show the plot
plt.show()