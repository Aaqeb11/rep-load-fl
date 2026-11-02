import numpy as np

from bcfl.data import dirichlet_split


def make_labels(samples_per_class=200):
    return np.repeat(np.arange(10), samples_per_class)


def test_every_sample_is_given_to_exactly_one_node():
    labels = make_labels()
    split = dirichlet_split(labels, num_nodes=8, alpha=0.5, seed=0)
    all_indices = sorted(i for node in split for i in node)
    assert all_indices == list(range(len(labels)))


def test_every_node_gets_the_minimum():
    split = dirichlet_split(make_labels(), num_nodes=8, alpha=0.5, seed=0, min_samples_per_node=30)
    assert min(len(node) for node in split) >= 30


def test_same_seed_gives_same_split():
    labels = make_labels()
    assert dirichlet_split(labels, 8, 0.5, seed=3) == dirichlet_split(labels, 8, 0.5, seed=3)


def test_small_alpha_is_more_uneven_than_large_alpha():
    labels = make_labels()

    def average_classes_per_node(alpha):
        split = dirichlet_split(labels, num_nodes=10, alpha=alpha, seed=0)
        # Count classes that make up at least 5% of a node's data.
        counts = []
        for node in split:
            per_class = np.bincount(labels[node], minlength=10)
            counts.append(np.sum(per_class >= 0.05 * len(node)))
        return np.mean(counts)

    assert average_classes_per_node(0.1) < average_classes_per_node(100)
