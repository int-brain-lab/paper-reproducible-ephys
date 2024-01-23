from one.api import ONE
from reproducible_ephys_functions import get_insertions, compute_metrics


def run_repro_ephys_metrics(one):
    insertions = get_insertions(level=0, one=one, freeze='release_2023_12')
    metrics = compute_metrics(insertions, one)


if __name__ == '__main__':
    one = ONE()
    run_repro_ephys_metrics(one)
