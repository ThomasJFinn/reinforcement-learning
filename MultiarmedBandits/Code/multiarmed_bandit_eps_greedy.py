import numpy as np
import random

def multiarmed_bandit_eps_greedy(num_bandits, q_mean, steps, eps):
    """
    Performs epsilon-greedy multi-armed bandit algorithm to find the optimal bandit arm

    Parameters:
        num_bandits (int): The number of bandit arms
        q_mean (array-like): The true mean value for each bandit arm
        steps (int): The number of steps to run the algorithm
        eps (float): The exploration probability (epsilon) for selecting a random action

    Returns:
        avg_reward (numpy array): The average reward at each step
        opt_ratio (numpy array): The ratio of optimal actions at each step
    """

    avg_reward = np.zeros(steps + 1)
    opt_ratio = np.zeros(steps + 1)
    opt_bandit = np.argmax(q_mean)

    # Initialize sample-averages and action count
    Q = np.zeros(num_bandits)
    action_count = np.zeros(num_bandits)

    for i in range(1, steps + 1):
        # Select next action using epsilon-greedy exploration
        p = random.uniform(0, 1)
        if p > eps:
            action = np.argmax(Q)
        else:
            action = random.randint(0, num_bandits - 1)

        # Incremental implementation for updating action values
        reward = np.random.normal(q_mean[action], 1)
        action_count[action] += 1
        Q[action] += (1 / action_count[action]) * (reward - Q[action])

        # Update average reward and optimal action ratio
        avg_reward[i] = (1 / i) * (((i - 1) * avg_reward[i - 1]) + reward)
        opt_ratio[i] = (1 / i) * ((action == opt_bandit) + (i - 1) * opt_ratio[i - 1])

    avg_reward = avg_reward[1:]  # Remove the initial element
    opt_ratio = opt_ratio[1:]  # Remove the initial element

    return avg_reward, opt_ratio