def swap_bandits(q):
    """
    Randomly swaps two entries in an input vector q

    """
    if len(q) < 2:
        # Return the vector q unchanged if it has less than 2 elements
        return q

    # Choose two distinct indices randomly
    index1, index2 = random.sample(range(len(q)), 2)

    # Swap the elements at the chosen indices
    q[index1], q[index2] = q[index2], q[index1]

    return q