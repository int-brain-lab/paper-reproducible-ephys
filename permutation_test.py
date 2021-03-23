"""
Find null distribution for arbitrary metrics, using permutation testing.

Written by Sebastian Bruijns
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def permut_test(data, metric, labels1, labels2, n_permut=1000, shuffling='labels1', plot=False):
    """
    Compute the probability of observating metric difference for datasets, via permutation testing.

    Parameters
    ----------
    data : array-like
        data set, e.g. array of dimension (n, ...)

    labels1, labels2 : array-like
        label set, one label for each datapoint, so dimension (n) each

    metric : function, array-like -> float
        Metric to use for permutation test, will be used to reduce elements of data1 and data2
        to one number

    n_permut : integer (optional)
        Number of perumtations to use for test

    plot : Boolean (optional)
        Whether or not to show a plot of the permutation distribution and a marker for the position
        of the true difference in relation to this distribution

    Returns
    -------
    p : float
        p-value of true difference in permutation null distribution

    See Also
    --------
    TODO:

    Examples
    --------
    TODO:
    """
    # Calculate metric
    observed_val = metric(data, labels1, labels2)

    # Prepare permutations
    permuted_labels1, permuted_labels2 = shuffle_labels(labels1, labels2, n_permut, shuffling)

    # Compute null dist (if this could be vectorized it would be a lot faster, but depends on metric...)
    null_dist = np.zeros(n_permut)
    for i in range(n_permut):
        null_dist[i] = metric(data, labels1[permuted_labels1[i]], labels2[permuted_labels2[i]])

    p = len(null_dist[null_dist > observed_val]) / n_permut
    if plot:
        plot_permut_test(null_dist=null_dist, observed_val=observed_val, p=p)

    return p


def shuffle_labels(labels1, labels2, n_permut, shuffling):
    """Shuffle labels according to demands."""
    if shuffling == 'labels1':
        permut_indices = np.tile(np.arange(labels1.size), (n_permut, 1))
        # TODO: use numpy random.Generator.permuted to do this in numpy version 1.20
        [np.random.shuffle(permut_indices[i]) for i in range(n_permut)]
        return permut_indices, np.tile(np.arange(labels1.size), (n_permut, 1))  # TODO: save space of second array, since it's not used?


def plot_permut_test(null_dist, observed_val, p, title=None):
    """Plot permutation test result."""
    n, _, _ = plt.hist(null_dist)

    # Plot the observed metric as red star
    plt.plot(observed_val, np.max(n) / 20, '*r', markersize=12)

    # Prettify plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title("p = {}".format(p))

    plt.show()


def example_metric(data, labels1, labels2):
    """Example metric: compute variance over means of different groups.

    (Ignoring labels2)"""
    # Find unique labels
    label_vals = np.unique(labels1)

    # Compute means of groups
    means = np.zeros(len(label_vals))
    for i, lv in enumerate(label_vals):
        means[i] = np.mean(data[labels1 == lv])

    # Return variance
    return np.var(means, ddof=1)


if __name__ == '__main__':
    rng = np.random.RandomState(2)
    data = rng.normal(0, 1, 25)
    t = time.time()
    p = permut_test(data, metric=example_metric, labels1=np.tile(np.arange(5), 5), labels2=np.ones(25, dtype=np.int), n_permut=1000, plot=True)
    print(time.time() - t)
    print(p)
