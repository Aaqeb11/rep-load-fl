"""The model every fog node trains, plus plain train and test loops."""

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
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


def test(model, loader, device):
    """Measure accuracy and loss on a dataset. Returns (accuracy, loss)."""
    model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return correct / total, total_loss / total
