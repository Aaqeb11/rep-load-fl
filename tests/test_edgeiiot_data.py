import json

import numpy as np

from bcfl.data import get_dataset_info, get_node_loader, get_test_loader, load_dataset


def write_tiny_prepared_dataset(folder, rows=200, features=6, classes=3):
    rng = np.random.default_rng(0)
    for split, n in [("train", rows), ("test", rows // 4)]:
        np.save(folder / f"X_{split}.npy", rng.normal(size=(n, features)).astype(np.float32))
        np.save(folder / f"y_{split}.npy", rng.integers(0, classes, size=n).astype(np.int64))
    meta = {"class_names": [f"c{i}" for i in range(classes)],
            "feature_names": [f"f{i}" for i in range(features)]}
    (folder / "meta.json").write_text(json.dumps(meta))


def make_config(folder):
    return {
        "data": {"dataset": "edgeiiot", "data_dir": str(folder), "alpha": 0.5,
                 "min_samples_per_node": 5},
        "federation": {"num_nodes": 4},
        "training": {"batch_size": 16},
        "seed": 0,
    }


def test_dataset_info_reads_meta(tmp_path):
    write_tiny_prepared_dataset(tmp_path)
    info = get_dataset_info(make_config(tmp_path))
    assert info["num_features"] == 6
    assert info["num_classes"] == 3


def test_training_data_is_memory_mapped(tmp_path):
    write_tiny_prepared_dataset(tmp_path)
    train_set, _ = load_dataset("edgeiiot", str(tmp_path))
    assert isinstance(train_set.X, np.memmap)


def test_nodes_together_hold_every_training_row(tmp_path):
    write_tiny_prepared_dataset(tmp_path)
    config = make_config(tmp_path)
    total = sum(len(get_node_loader(config, node).dataset) for node in range(4))
    assert total == 200

    images, labels = next(iter(get_node_loader(config, 0)))
    assert images.dtype.is_floating_point and images.shape[1] == 6


def test_test_loader_covers_the_whole_test_set(tmp_path):
    write_tiny_prepared_dataset(tmp_path)
    loader = get_test_loader(make_config(tmp_path))
    assert len(loader.dataset) == 50
