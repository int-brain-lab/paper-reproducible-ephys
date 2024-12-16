"""
Find null distribution for arbitrary metrics, using permutation testing.

Written by Sebastian Bruijns
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from reproducible_ephys_functions import LAB_MAP

lab_number_map, institution_map, lab_colors = LAB_MAP()

def permut_test(data, metric, labels1, labels2, n_permut=10000, shuffling='labels1', plot=False, return_details=False, mark_p=None, n_cores=1, title=None):
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
    observed_val = metric(data, labels1, labels2, print_it=False, plot_it=plot)

    # Prepare permutations
    permuted_labels1, permuted_labels2 = shuffle_labels(labels1, labels2, n_permut, shuffling, n_cores=n_cores)


    # Compute null dist (if this could be vectorized it would be a lot faster, but depends on metric...)
    null_dist = np.zeros(n_permut)
    if n_cores == 1:
        for i in range(n_permut):
            null_dist[i] = metric(data, permuted_labels1[i], permuted_labels2[i], print_it=False, plot_it=plot)
    else:
        size = n_permut // n_cores
        arg_list = [(metric, data, permuted_labels1[i*size:(i+1)*size], permuted_labels2[i*size:(i+1)*size]) for i in range(n_cores)]
        pool = mp.Pool(n_cores)
        part_null_dist = pool.starmap(metric_helper, arg_list)
        pool.close()
        null_dist = np.concatenate(part_null_dist, axis=0)
    p = len(null_dist[null_dist > observed_val]) / n_permut
    if plot:
        plot_permut_test(null_dist=null_dist, observed_val=observed_val, p=p, mark_p=mark_p)

    if return_details:
        return p, np.mean(null_dist), null_dist
    else:
        return p


def metric_helper(metric, data, permuted_labels1, permuted_labels2):
    n = len(permuted_labels1)
    part_null_dist = np.zeros(n)
    for i in range(n):
        part_null_dist[i] = metric(data, permuted_labels1[i], permuted_labels2[i], print_it=False, plot_it=False)
    return part_null_dist


def shuffle_labels(labels1, labels2, n_permut, shuffling, n_cores=1):
    """Shuffle labels according to demands."""
    if shuffling == 'labels1':
        permut_indices = np.tile(np.arange(labels1.size), (n_permut, 1))
        # TODO: use numpy random.Generator.permuted to do this in numpy version 1.20
        [np.random.shuffle(permut_indices[i]) for i in range(n_permut)]
        # TODO: save space of second array, since it's not used?
        return labels1[permut_indices], np.tile(labels2, (n_permut, 1))
    if shuffling == 'all':
        permut_indices = np.tile(np.arange(labels1.size), (n_permut, 1))
        [np.random.shuffle(permut_indices[i]) for i in range(n_permut)]
        permut_indices_2 = np.tile(np.arange(labels1.size), (n_permut, 1))
        [np.random.shuffle(permut_indices_2[i]) for i in range(n_permut)]
        # TODO: save space of second array, since it's not used?
        return labels1[permut_indices], labels2[permut_indices_2]
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
        if n_cores == 1:
            for i in range(n_permut):
                # shuffle associations between mice and labs
                np.random.shuffle(label2_values)
                mix_dict = dict(zip(label2_values, label1_values))
                lab_mapping = np.vectorize(mix_dict.get)
                permuted_labels1[i] = lab_mapping(labels2)
        else:
            size = n_permut // n_cores
            start_seed = np.random.randint(100000)
            arg_list = [(size, label1_values, label2_values, labels1, labels2, start_seed + i) for i in range(n_cores)]
            pool = mp.Pool(n_cores)
            part_permuted_labels1 = pool.starmap(shuffle_helper, arg_list)
            pool.close()
            permuted_labels1 = np.concatenate(part_permuted_labels1, axis=0)

        return permuted_labels1, np.tile(labels2, (n_permut, 1))


def shuffle_helper(size, label1_values, label2_values, labels1, labels2, seed):
    # shuffle associations between mice and labs
    np.random.seed(seed)
    part_permuted_labels1 = np.zeros((size, len(labels1)))
    for i in range(size):
        np.random.shuffle(label2_values)
        mix_dict = dict(zip(label2_values, label1_values))
        lab_mapping = np.vectorize(mix_dict.get)
        part_permuted_labels1[i] = lab_mapping(labels2)
    return part_permuted_labels1


def plot_permut_test(null_dist, observed_val, p, mark_p, title=None):
    """Plot permutation test result."""
    plt.figure(figsize=(16 * 0.75, 9 * 0.75))
    n, _, _ = plt.hist(null_dist, bins=25)

    # Plot the observed metric as red star
    plt.plot(observed_val, np.max(n) / 20, '*r', markersize=12, label="Observed distance")
    plt.axvline(np.mean(null_dist), color='k', label="Expectation")
    if mark_p is not None:
        sorted = np.sort(null_dist)
        critical_point = sorted[int((1 - mark_p) * len(null_dist))]
        print("p value of critical point is {}".format(len(null_dist[null_dist > critical_point]) / len(null_dist)))
        plt.axvline(critical_point, color='r', label="Significance")

    plt.xlabel("Firing rate modulation", size=22)
    plt.ylabel("Permuted occurences", size=22)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)

    plt.legend(frameon=False, fontsize=17)
    # Prettify plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title("p = {}".format(np.round(p, 3)), size=22)

    plt.tight_layout()
    plt.savefig("null dist")
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


def permut_dist(data, labs, mice, print_it, plot_it):
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


def distribution_dist_approx_max(data, labs, mice, n=400, print_it=False, plot_it=False):
    low = data.min()
    high = data.max()

    if print_it:
        print()
    if plot_it:
        plt.figure(figsize=(16 * 0.75, 9 * 0.75))

    cdf_dists = []
    dist_inds = []
    for lab in np.unique(labs):
        p1_array = np.zeros(n)
        p1_array = np.bincount(np.clip(((data[labs == lab] - low) / (high - low) * n).astype(int), a_min=None, a_max=n-1), minlength=n)
        p1_array = np.cumsum(p1_array)
        p1_array = p1_array / p1_array[-1]

        if plot_it:
            plt.plot(np.linspace(low, high, n), p1_array, label=lab, color=lab_colors[lab])
            # if lab == 'UCLA':
            #     for mouse in np.unique(mice[labs == lab]):
            #         temp = np.zeros(n)
            #         temp = np.bincount(np.clip(((data[np.logical_and(labs == lab, mice == mouse)] - low) / (high - low) * n).astype(int), a_min=None, a_max=n-1), minlength=n)
            #         temp = np.cumsum(temp)
            #         temp = temp / temp[-1]
            #         # print(data[np.logical_and(labs == lab, mice == mouse)].shape)
            #         plt.plot(np.linspace(low, high, n), temp, label=mouse, ls='--')

        temp, temp_ind = helper(n, p1_array, data[labs != lab], low, high)
        cdf_dists.append(temp)
        dist_inds.append(temp_ind)

    if print_it:
        max_diff_ind = np.argmax(cdf_dists)
        print(cdf_dists, cdf_dists[max_diff_ind], max_diff_ind)

    if plot_it:
        max_diff_ind = np.argmax(cdf_dists)
        total_array = np.zeros(n)
        total_array = np.bincount(np.clip(((data - low) / (high - low) * n).astype(int), a_min=None, a_max=n-1), minlength=n)
        total_array = np.cumsum(total_array)
        total_array = total_array / total_array[-1]
        plt.axvline(np.linspace(low, high, n)[dist_inds[max_diff_ind]], c='red')
        plt.plot(np.linspace(low, high, n), total_array, label='Overall', color='k', lw=3)
        plt.xlabel("Firing rate modulation", size=22)
        plt.ylabel("Cumulative probability", size=22)
        plt.legend(frameon=False, fontsize=17)
        # plt.xlim(-5, 10)
        plt.ylim(0, 1.01)
        plt.gca().tick_params(axis='both', which='major', labelsize=14)
        # plt.title("Distance = {}".format(np.round(cdf_dists[max_diff_ind], 2)), size=22)
        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.tight_layout()
        plt.savefig("CA1 CDFs plus no title")
        plt.show()

    return max(cdf_dists)


def distribution_dist_approx(data, labs, mice, n=400, print_it=False, plot_it=False, title=False):
    dist_sum = 0
    low = data.min()
    high = data.max()

    for lab in np.unique(labs):
        p1_array = np.zeros(n)
        p1_array = np.bincount(np.clip(((data[labs == lab] - low) / (high - low) * n).astype(int), a_min=None, a_max=n-1), minlength=n)
        p1_array = np.cumsum(p1_array)
        p1_array = p1_array / p1_array[-1]

        if plot_it:
            plt.plot(np.linspace(low, high, n), p1_array, label=lab)

        temp = helper(n, p1_array, data[labs != lab], low, high)
        if print_it:
            print(temp)
        dist_sum += temp

    if print_it:
        print(dist_sum)

    if plot_it:
        total_array = np.zeros(n)
        total_array = np.bincount(np.clip(((data - low) / (high - low) * n).astype(int), a_min=None, a_max=n-1), minlength=n)
        total_array = np.cumsum(total_array)
        total_array = total_array / total_array[-1]
        plt.plot(np.linspace(low, high, n), total_array, label='Overall')
        plt.legend()
        plt.xlim(-5, 10)
        plt.title(title)
        plt.show()

    return dist_sum


def helper(n, p1_array, points2, low, high):
    p2_array = np.zeros(n)

    p2_array = np.bincount(np.clip(((points2 - low) / (high - low) * n).astype(int), a_min=None, a_max=n-1), minlength=n)
    p2_array = np.cumsum(p2_array)

    diffs = np.abs(p1_array - p2_array / p2_array[-1])
    max_ind = np.argmax(diffs)
    return diffs[max_ind], max_ind


def power_test(n_simul, dist, labels1, labels2, diff_labels1, metric=distribution_dist_approx, shuffling='labels1_based_on_2'):
    ps = np.zeros(n_simul)
    for i in range(n_simul):
        if i % 20 == 0:
            print(i)
        data = np.random.normal(size=labels2.shape)
        data[labels1 == diff_labels1] += dist
        p = permut_test(data, metric=metric, labels1=labels1, labels2=labels2,
                        shuffling=shuffling, n_permut=5000, plot=False)
        ps[i] = p
    # plt.hist(ps)
    # plt.show()
    return ps


if __name__ == '__main__':
    import pickle
    labels1, labels2 = pickle.load(open("temp", 'rb'))

    # mean comparison
    # new_labels1, new_labels2 = [], []
    # for i in range(len(labels2)):
    #     if labels2[i] not in new_labels2:
    #         new_labels2.append(labels2[i])
    #         new_labels1.append(labels1[i])
    # labels1, labels2 = np.array(new_labels1), np.array(new_labels2)
    # metric, shuffling = permut_dist, 'labels1'

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
                    shuffling='labels1_based_on_2', n_permut=10000, plot=False)
    print(time.time() - t)
    print(p)

    t = time.time()
    p = permut_test(data, metric=distribution_dist, labels2=np.array(["0", "0", "0", "1", "2", "2", "3", "3", "3", "3", "4", "4", "5", "5", "5"]), labels1=np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1, 1]),
                    shuffling='labels1_based_on_2', n_permut=100000, plot=False)
    print(time.time() - t)
    print(p)
