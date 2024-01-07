import matplotlib.pyplot as plt

num_bandits = 10
q_mean = np.random.normal(0, 1, num_bandits)
steps = 10000

# Call the epsilon-greedy and upper-confidence-bound action selection multiarmed bandit algorithm 
avg_reward_eps, opt_ratio_eps = multiarmed_bandit_eps_greedy(num_bandits, q_mean, steps, eps=0.1)
avg_reward_UCB, opt_ratio_UCB = multiarmed_bandit_UCB(num_bandits, q_mean, steps, weight=2)

# Create a figure and axes for the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
ttl = "Multi-armed bandit problem with 10 choices under an $\\varepsilon$-greedy algorithm \n and upper-confidence bound action selection"
fig.suptitle(ttl, fontsize=16, fontweight='bold')

# Plot the average reward data
axes[0].plot(avg_reward_eps, linewidth=1.2, label=r"Greedy with $\varepsilon=0.1$")
axes[0].plot(avg_reward_UCB, linewidth=1.2, color='green', label=r"UCB with weight $c=2$")
axes[0].set_xlabel('Steps', fontsize=12)
axes[0].set_ylabel('Average reward', fontsize=12)
axes[0].legend(loc='best', fontsize=10)

# Plot the optimal action ratio data
axes[1].plot(opt_ratio_eps, linewidth=1.2, label=r"Greedy with $\varepsilon=0.1$")
axes[1].plot(opt_ratio_UCB, linewidth=1.2, color='green', label=r"UCB with weight $c=2$")
axes[1].set_xlabel('Steps', fontsize=12)
axes[1].set_ylabel('Optimal action ratio', fontsize=12)
axes[1].legend(loc='best', fontsize=10)

# Adjust layout and padding
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plot
plt.show()