"""
Find null distribution for arbitrary metrics, using permutation testing.

Written by Sebastian Bruijns
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def permut_test(data, metric, labels1, labels2, n_permut=10000, shuffling='labels1', plot=False, return_details=False, mark_p=None):
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
        null_dist[i] = metric(data, permuted_labels1[i], permuted_labels2[i])

    p = len(null_dist[null_dist > observed_val]) / n_permut
    if plot:
        plot_permut_test(null_dist=null_dist, observed_val=observed_val, p=p, mark_p=mark_p)

    if return_details:
        return p, np.mean(null_dist), null_dist
    else:
        return p


def shuffle_labels(labels1, labels2, n_permut, shuffling):
    """Shuffle labels according to demands."""
    if shuffling == 'labels1':
        permut_indices = np.tile(np.arange(labels1.size), (n_permut, 1))
        # TODO: use numpy random.Generator.permuted to do this in numpy version 1.20
        [np.random.shuffle(permut_indices[i]) for i in range(n_permut)]
        # TODO: save space of second array, since it's not used?
        return labels1[permut_indices], np.tile(labels2, (n_permut, 1))
    if shuffling == 'labels1_based_on_2':
        # e.g. shuffle labs based on subject (e.g. if you operate over neurons and want to leave collection of neuron withon one mouse intact)
        # note: this shuffles the lab labels, based on subjects, but the subjects themselves are not shuffled

        # turn labs and subjects into numbers
        unique_labels = np.unique(labels1)
        l1_dict = dict(zip(unique_labels, np.arange(len(unique_labels))))
        l1_mapping = np.vectorize(l1_dict.get)
        labels1 = l1_mapping(labels1)

        unique_labels = np.unique(labels2)
        l2_dict = dict(zip(unique_labels, np.arange(len(unique_labels))))
        l2_mapping = np.vectorize(l2_dict.get)
        labels2 = l2_mapping(labels2)

        # get combinations of subjects and labs
        label1_values = []
        label2_values = []
        for l2 in np.unique(labels2):
            label2_values.append(l2)
            label1_values.append(labels1[labels2 == l2][0])

        permuted_labels1 = np.zeros((n_permut, len(labels1)))
        for i in range(n_permut):
            # shuffle associations between mice and labs
            np.random.shuffle(label2_values)
            mix_dict = dict(zip(label2_values, label1_values))
            lab_mapping = np.vectorize(mix_dict.get)
            permuted_labels1[i] = lab_mapping(labels2)

        return permuted_labels1, np.tile(labels2, (n_permut, 1))


def plot_permut_test(null_dist, observed_val, p, mark_p, title=None):
    """Plot permutation test result."""
    n, _, _ = plt.hist(null_dist)

    # Plot the observed metric as red star
    plt.plot(observed_val, np.max(n) / 20, '*r', markersize=12)
    plt.axvline(np.mean(null_dist), color='k', label="Expectation")
    if mark_p is not None:
        sorted = np.sort(null_dist)
        critical_point = sorted[int((1 - mark_p) * len(null_dist))]
        print("p value of critical point is {}".format(len(null_dist[null_dist > critical_point]) / len(null_dist)))
        plt.axvline(critical_point, color='r', label="Significance")

    plt.legend()
    # Prettify plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title("p = {}".format(p))

    plt.savefig("temp")
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


def permut_dist(data, labs, mice):
    lab_means = []
    for lab in np.unique(labs):
        lab_means.append(np.mean(data[labs == lab]))
    lab_means = np.array(lab_means)
    return np.sum(np.abs(lab_means - np.mean(lab_means)))


def distribution_dist(data, labs, mice):
    # Don't just consider means, but take entire distribution into account
    # we compare the overall dist with the individual labs
    sorted_points, sorted_counts = np.unique(data, return_counts=True)
    n = data.shape[0]
    dist_sum = 0
    for lab in np.unique(labs):
        lab_total = np.sum(labs == lab)
        lab_count = 0
        overall_count = 0
        prev_point = 0
        lab_points = data[labs == lab]
        for i, (p, c) in enumerate(zip(sorted_points, sorted_counts)):
            dist_sum += (p - prev_point) * np.abs(overall_count / n - lab_count / lab_total)
            prev_point = p
            lab_count += np.sum(lab_points == p)
            overall_count += c
    return dist_sum


def distribution_dist_approx(data, labs, mice, n=400):
    dist_sum = 0
    low = data.min()
    high = data.max()
    p1_array = np.zeros(n)
    for p in data:
        p1_array[min(int((p - low) / (high - low) * n), n-1):] += 1
    p1_array = p1_array / p1_array[-1]
    for lab in np.unique(labs):
        dist_sum += helper(n, p1_array, data[labs == lab], low, high)
    return dist_sum


def helper(n, p1_array, points2, low, high):
    p2_array = np.zeros(n)
    for p in points2:
        p2_array[min(int((p - low) / (high - low) * n), n-1):] += 1
    return np.sum(np.abs(p1_array - p2_array / p2_array[-1])) / n * (high - low)


if __name__ == '__main__':
    rng = np.random.RandomState(4)
    data = rng.normal(0, 1, 25)
    t = time.time()
    p = permut_test(data, metric=example_metric, labels1=np.tile(np.arange(5), 5), labels2=np.ones(25, dtype=int),
                    n_permut=1000, plot=False)
    print(time.time() - t)
    print(p)

    data = rng.normal(0, 1, 15)
    t = time.time()
    p = permut_test(data, metric=distribution_dist_approx, labels2=np.array(["0", "0", "0", "1", "2", "2", "3", "3", "3", "3", "4", "4", "5", "5", "5"]), labels1=np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1, 1]),
                    shuffling='labels1_based_on_2', n_permut=100000, plot=False)
    print(time.time() - t)
    print(p)

    t = time.time()
    p = permut_test(data, metric=distribution_dist, labels2=np.array(["0", "0", "0", "1", "2", "2", "3", "3", "3", "3", "4", "4", "5", "5", "5"]), labels1=np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1, 1]),
                    shuffling='labels1_based_on_2', n_permut=100000, plot=False)
    print(time.time() - t)
    print(p)
