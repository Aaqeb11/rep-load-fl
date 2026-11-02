"""Datasets and the non-IID split across fog nodes.

Each fog node gets its own slice of the training data. To make the data
uneven across nodes (non-IID), we use a Dirichlet split:

- For every class (cat, car, ...), we draw how that class is shared out
  among the nodes.
- A small alpha (e.g. 0.1) means each class goes mostly to a few nodes,
  so each node sees only a few classes. Very uneven.
- A large alpha (e.g. 100) means each class is shared almost equally,
  so every node sees every class. Close to IID.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

NUM_CLASSES = 10

# Each simulated node runs in its own process. These caches make sure each
# process loads the dataset and computes the split only once.
_dataset_cache = {}
_split_cache = {}


# ---------------------------------------------------------------------------
# Loading the dataset
# ---------------------------------------------------------------------------

def load_dataset(name, data_dir, download=False):
    """Return (train_set, test_set) for the chosen dataset."""
    cache_key = (name, data_dir)
    if cache_key in _dataset_cache:
        return _dataset_cache[cache_key]

    if name == "cifar10":
        train_set, test_set = _load_cifar10(data_dir, download)
    elif name == "synthetic":
        train_set, test_set = _make_synthetic()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    _dataset_cache[cache_key] = (train_set, test_set)
    return train_set, test_set


def _load_cifar10(data_dir, download):
    """CIFAR-10: 50,000 training and 10,000 test images, 32x32 colour, 10 classes."""
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        # Standard CIFAR-10 mean and standard deviation for each colour channel.
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = datasets.CIFAR10(data_dir, train=True, download=download, transform=transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=download, transform=transform)
    return train_set, test_set


def _make_synthetic(num_train=2000, num_test=500):
    """Fake CIFAR-shaped data for quick tests (no download needed).

    Every class has its own fixed "pattern" image. Each sample is that pattern
    plus random noise, so a model can actually learn to tell classes apart.
    """
    generator = torch.Generator().manual_seed(1234)
    class_patterns = torch.randn(NUM_CLASSES, 3, 32, 32, generator=generator)

    def make_split(size):
        labels = torch.randint(0, NUM_CLASSES, (size,), generator=generator)
        noise = torch.randn(size, 3, 32, 32, generator=generator) * 1.5
        images = class_patterns[labels] + noise
        return TensorDataset(images, labels)

    return make_split(num_train), make_split(num_test)


def get_labels(dataset):
    """Return the label of every sample as a numpy array."""
    if hasattr(dataset, "targets"):          # torchvision datasets
        return np.array(dataset.targets)
    return dataset.tensors[1].numpy()        # TensorDataset (synthetic)


# ---------------------------------------------------------------------------
# Dirichlet split
# ---------------------------------------------------------------------------

def dirichlet_split(labels, num_nodes, alpha, seed, min_samples_per_node=10):
    """Split sample indices across nodes using a Dirichlet distribution.

    Returns a list with one entry per node; each entry is a list of sample indices.
    If some node ends up with too few samples, we simply draw again.
    """
    rng = np.random.default_rng(seed)

    for _attempt in range(1000):
        node_indices = [[] for _ in range(num_nodes)]

        for class_id in range(NUM_CLASSES):
            # All samples of this class, in random order.
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)

            # What share of this class each node gets (shares add up to 1).
            shares = rng.dirichlet([alpha] * num_nodes)

            # Turn shares into cut points, then cut the class into pieces.
            cut_points = (np.cumsum(shares) * len(class_indices)).astype(int)[:-1]
            pieces = np.split(class_indices, cut_points)

            for node_id, piece in enumerate(pieces):
                node_indices[node_id].extend(piece.tolist())

        smallest_node = min(len(indices) for indices in node_indices)
        if smallest_node >= min_samples_per_node:
            return node_indices

    raise RuntimeError(
        "Could not find a split where every node has enough samples. "
        "Try a larger alpha or a smaller min_samples_per_node."
    )


def get_node_split(config):
    """Return the split for this config (computed once per process)."""
    data_cfg = config["data"]
    num_nodes = config["federation"]["num_nodes"]
    cache_key = (data_cfg["dataset"], num_nodes, data_cfg["alpha"], config["seed"])

    if cache_key not in _split_cache:
        train_set, _ = load_dataset(data_cfg["dataset"], data_cfg["data_dir"])
        _split_cache[cache_key] = dirichlet_split(
            labels=get_labels(train_set),
            num_nodes=num_nodes,
            alpha=data_cfg["alpha"],
            seed=config["seed"],
            min_samples_per_node=data_cfg["min_samples_per_node"],
        )
    return _split_cache[cache_key]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def get_node_loader(config, partition_id):
    """Training data loader for one fog node."""
    train_set, _ = load_dataset(config["data"]["dataset"], config["data"]["data_dir"])
    node_indices = get_node_split(config)[partition_id]
    return DataLoader(
        Subset(train_set, node_indices),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )


def get_test_loader(config, batch_size=256):
    """Test data loader, used by the coordinator to measure global accuracy."""
    _, test_set = load_dataset(config["data"]["dataset"], config["data"]["data_dir"])
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)


def describe_split(config):
    """Count how many samples of each class every node has (for reports and checks)."""
    train_set, _ = load_dataset(config["data"]["dataset"], config["data"]["data_dir"])
    labels = get_labels(train_set)
    counts = []
    for indices in get_node_split(config):
        counts.append(np.bincount(labels[indices], minlength=NUM_CLASSES).tolist())
    return counts
