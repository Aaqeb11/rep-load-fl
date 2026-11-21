"""The models every fog node trains, plus plain train and test loops.

- SimpleCNN: for 32x32 colour images (CIFAR-10, Step 1)
- MLP:       a small fully connected network for tables of numbers (Edge-IIoTset)
"""

import numpy as np
import torch
from torch import nn


class SimpleCNN(nn.Module):
    """A small convolutional network for 32x32 colour images (10 classes).

    Small on purpose: 20-30 copies have to train on one laptop.
    It has no batch-norm layers, so averaging models stays simple.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 32x32 -> 32x32
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


class MLP(nn.Module):
    """A small fully connected network for tabular data such as network traffic.

    Each packet is a row of numbers (90 features for Edge-IIoTset), not an image,
    so there are no convolution layers: just Linear layers with ReLU in between.
    """

    def __init__(self, num_features, num_classes, hidden_sizes=(128, 64)):
        super().__init__()
        layers = []
        size_in = num_features
        for size_out in hidden_sizes:
            layers.append(nn.Linear(size_in, size_out))
            layers.append(nn.ReLU())
            size_in = size_out
        layers.append(nn.Linear(size_in, num_classes))   # one score per class
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)


def build_model(config):
    """Create the model named in the config ("cnn" or "mlp").

    Every node and the coordinator call this, so they all build the SAME
    structure, which is what lets FedAvg average the weights layer by layer.
    """
    # Imported here to avoid a circular import (data.py does not need model.py).
    from bcfl.data import get_dataset_info

    name = config.get("model", "cnn")
    info = get_dataset_info(config)
    if name == "cnn":
        return SimpleCNN(num_classes=info["num_classes"])
    if name == "mlp":
        return MLP(num_features=info["num_features"], num_classes=info["num_classes"])
    raise ValueError(f"Unknown model '{name}'. Options: ['cnn', 'mlp']")


def get_device(name):
    """Pick the device, falling back to CPU if the requested one is missing."""
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(model, loader, epochs, learning_rate, device):
    """Train the model on one node's data. Returns the average training loss."""
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_batches = 0
    for _epoch in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


def test(model, loader, device):
    """Measure the model on a dataset.

    Returns a dictionary:
        accuracy   share of rows predicted correctly
        loss       average cross-entropy loss
        macro_f1   F1 per class, then the plain average over classes
        confusion  table[true class][predicted class] = number of rows
    """
    model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += loss_fn(outputs, labels).item()
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(outputs.argmax(dim=1).cpu().numpy())

    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    num_classes = outputs.shape[1]

    confusion = confusion_matrix(labels, predictions, num_classes)
    return {
        "accuracy": float((labels == predictions).mean()),
        "loss": total_loss / len(labels),
        "macro_f1": macro_f1(confusion),
        "confusion": confusion,
    }


def confusion_matrix(labels, predictions, num_classes):
    """Count rows by (true class, predicted class)."""
    table = np.zeros((num_classes, num_classes), dtype=np.int64)
    # For every row, add 1 to table[its true class, its predicted class].
    # (np.add.at does this for all 360,000 test rows at once, instead of a slow loop.)
    np.add.at(table, (labels, predictions), 1)
    return table


def macro_f1(confusion):
    """F1 for every class, then the plain average: every class counts equally.

    For one class:
        precision = correct predictions of it / all predictions of it
        recall    = correct predictions of it / all real rows of it
        F1        = 2 * precision * recall / (precision + recall)
    A class that is never predicted (or never present) gets F1 = 0.
    """
    f1_scores = []
    for class_id in range(confusion.shape[0]):
        correct = confusion[class_id, class_id]
        predicted_as_this = confusion[:, class_id].sum()
        actually_this = confusion[class_id, :].sum()

        precision = correct / predicted_as_this if predicted_as_this > 0 else 0.0
        recall = correct / actually_this if actually_this > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return float(np.mean(f1_scores))
