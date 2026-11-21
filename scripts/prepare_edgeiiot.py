"""Prepare Edge-IIoTset once: clean, encode, split, scale, and save as .npy files.

Run this ONCE before any training. Every simulated fog node then opens the small
.npy files (memory-mapped) instead of each reading the 1.2 GB CSV.

Usage:
    python scripts/prepare_edgeiiot.py --csv data/DNN-EdgeIIoT-dataset.csv --out data/edgeiiot

Output folder:
    X_train.npy, y_train.npy, X_test.npy, y_test.npy   the data, as numbers
    meta.json                                          class names, feature names, counts
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Columns that let the model "cheat" instead of learning what an attack looks like.
# This list comes from the dataset authors' README:
#   timestamps, IP addresses, port numbers and raw packet content.
IDENTIFIER_COLUMNS = [
    "frame.time", "icmp.transmit_timestamp",
    "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "tcp.srcport", "tcp.dstport", "udp.port",
    "http.file_data", "http.request.full_uri", "http.request.uri.query",
    "tcp.options", "tcp.payload", "mqtt.msg",
]

# Extra leaks the README does not drop.
# udp.stream is a counter that goes up by one for every new UDP conversation in the
# capture. Like a timestamp, it tells the model WHEN in the recording a row happened,
# and each attack was recorded in its own session.
EXTRA_LEAK_COLUMNS = ["udp.stream"]

# Text columns. Each one becomes several 0/1 columns, one per possible value.
TEXT_COLUMNS = [
    "http.request.method", "http.referer", "http.request.version",
    "dns.qry.name.len", "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic",
]

LABEL_COLUMN = "Attack_type"        # what we predict: Normal or one of 14 attacks
UNUSED_COLUMNS = ["Attack_label"]   # just "attack or not"; we use Attack_type instead


def load_and_clean(csv_path):
    """Read the CSV and remove leaky columns, missing values, duplicates and useless columns."""
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"loaded:                       {df.shape[0]:>9,} rows, {df.shape[1]} columns")

    df = df.drop(columns=IDENTIFIER_COLUMNS + EXTRA_LEAK_COLUMNS + UNUSED_COLUMNS)
    print(f"after dropping leaky columns: {df.shape[0]:>9,} rows, {df.shape[1]} columns")

    df = df.dropna()
    print(f"after dropping missing values:{df.shape[0]:>9,} rows")

    # Duplicates: the same row could otherwise land in both train and test,
    # and the model would be "tested" on something it has already seen.
    df = df.drop_duplicates()
    print(f"after dropping duplicates:    {df.shape[0]:>9,} rows")

    # A column with the same value in every row tells the model nothing.
    constant_columns = []
    for column in df.columns:
        if column != LABEL_COLUMN and df[column].nunique() == 1:
            constant_columns.append(column)
    df = df.drop(columns=constant_columns)
    print(f"dropped constant columns:     {constant_columns}")

    return df


def encode_text_columns(df):
    """Turn each text column into 0/1 columns, e.g. http.request.method -> method_GET, method_POST ...

    We do NOT use GET=1, POST=2, PUT=3: the model would think PUT is "more" than GET.
    """
    for column in TEXT_COLUMNS:
        # Some cells are "0.0" (text) and some 0.0 (number); make them all text first.
        df[column] = df[column].astype(str)
    return pd.get_dummies(df, columns=TEXT_COLUMNS)


def split_labels_and_features(df):
    """Return (X, y, class_names, feature_names) as plain numpy arrays."""
    # A FIXED, sorted list, so each class gets the same number on every machine and run.
    class_names = sorted(df[LABEL_COLUMN].unique())
    class_to_number = {name: number for number, name in enumerate(class_names)}

    y = df[LABEL_COLUMN].map(class_to_number).to_numpy(dtype=np.int64)

    features = df.drop(columns=[LABEL_COLUMN])
    # Safety check: every input column must be numeric by now.
    # (is_numeric_dtype also accepts the True/False columns made by get_dummies.)
    leftover_text = []
    for column in features.columns:
        if not pd.api.types.is_numeric_dtype(features[column]):
            leftover_text.append(column)
    if leftover_text:
        raise ValueError(f"These columns are still text and need encoding: {leftover_text}")

    X = features.to_numpy(dtype=np.float32)
    feature_names = list(features.columns)
    return X, y, class_names, feature_names


def print_class_counts(name, y, class_names):
    counts = np.bincount(y, minlength=len(class_names))
    print(f"\n{name}: {len(y):,} rows")
    for class_number, class_name in enumerate(class_names):
        print(f"  {class_number:>2}  {class_name:<22} {counts[class_number]:>9,}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Edge-IIoTset for training.")
    parser.add_argument("--csv", required=True, help="Path to DNN-EdgeIIoT-dataset.csv")
    parser.add_argument("--out", required=True, help="Folder to write the .npy files into")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = load_and_clean(args.csv)
    df = encode_text_columns(df)
    X, y, class_names, feature_names = split_labels_and_features(df)
    print(f"\nfeatures: {X.shape[1]}   classes: {len(class_names)}")

    # STRATIFIED split: every class keeps the same share in train and test,
    # so even the rarest attack has enough rows in the test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_fraction, stratify=y, random_state=args.seed,
    )

    # Scale every feature to mean 0 and standard deviation 1.
    # The scaler learns the mean and spread from the TRAINING data only:
    # the test set stands for future, unseen traffic, so it must not influence training.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "X_train.npy"), X_train)
    np.save(os.path.join(args.out, "y_train.npy"), y_train)
    np.save(os.path.join(args.out, "X_test.npy"), X_test)
    np.save(os.path.join(args.out, "y_test.npy"), y_test)

    meta = {
        "source_csv": os.path.basename(args.csv),
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "class_names": class_names,
        "feature_names": feature_names,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print_class_counts("train", y_train, class_names)
    print_class_counts("test", y_test, class_names)
    size_mb = (X_train.nbytes + X_test.nbytes) / 1e6
    print(f"\nsaved to {args.out}  (features take {size_mb:,.0f} MB as float32)")


if __name__ == "__main__":
    main()
