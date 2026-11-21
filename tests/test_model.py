import numpy as np
import torch

from bcfl.model import MLP, confusion_matrix, macro_f1


def test_lazy_model_has_high_accuracy_but_low_macro_f1():
    # 8 Normal rows (class 0) and 2 attack rows (class 1); the model always says Normal.
    labels = np.array([0] * 8 + [1] * 2)
    predictions = np.zeros(10, dtype=np.int64)
    confusion = confusion_matrix(labels, predictions, num_classes=2)

    accuracy = (labels == predictions).mean()
    assert accuracy == 0.8
    # Normal: precision 0.8, recall 1.0 -> F1 = 0.889; attack: F1 = 0 -> average 0.444
    assert abs(macro_f1(confusion) - (2 * 0.8 / 1.8) / 2) < 1e-9


def test_perfect_predictions_give_macro_f1_of_one():
    labels = np.array([0, 1, 2, 2, 1])
    confusion = confusion_matrix(labels, labels, num_classes=3)
    assert macro_f1(confusion) == 1.0


def test_confusion_matrix_counts_true_against_predicted():
    confusion = confusion_matrix(np.array([0, 0, 1]), np.array([0, 1, 1]), num_classes=2)
    assert confusion.tolist() == [[1, 1], [0, 1]]


def test_mlp_gives_one_score_per_class():
    model = MLP(num_features=90, num_classes=15)
    scores = model(torch.zeros(4, 90))
    assert scores.shape == (4, 15)
