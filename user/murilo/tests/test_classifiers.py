"""Test scikit-learn classifiers for linear probing."""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score, f1_score


class TestClassifiers:
    """Test linear probe classifiers."""

    def test_logistic_regression(self):
        """Test basic LogisticRegression."""
        X_train = np.random.randn(100, 1024)
        y_train = np.random.randint(0, 2, size=(100,))

        clf = LogisticRegression(max_iter=100)
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_train)
        assert predictions.shape == (100,)

    def test_one_vs_rest_multilabel(self):
        """Test OneVsRestClassifier for multi-label classification."""
        # Dummy data: 100 samples, 1024 features, 50 classes
        X_train = np.random.randn(100, 1024)
        y_train = np.random.randint(0, 2, size=(100, 50))  # Multi-hot labels

        X_test = np.random.randn(20, 1024)
        y_test = np.random.randint(0, 2, size=(20, 50))

        # Train classifier
        clf = OneVsRestClassifier(
            LogisticRegression(C=1.0, max_iter=100, random_state=42)
        )
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)
        assert y_pred.shape == (20, 50)

        # Get probabilities
        y_pred_proba = clf.predict_proba(X_test)
        assert y_pred_proba.shape == (20, 50)

    def test_multilabel_metrics(self):
        """Test multi-label classification metrics."""
        # Dummy predictions
        y_true = np.random.randint(0, 2, size=(100, 10))
        y_pred_proba = np.random.rand(100, 10)
        y_pred_binary = (y_pred_proba > 0.5).astype(int)

        # Compute F1 scores
        f1_micro = f1_score(y_true, y_pred_binary, average='micro')
        f1_macro = f1_score(y_true, y_pred_binary, average='macro')

        assert 0 <= f1_micro <= 1
        assert 0 <= f1_macro <= 1

        # Compute per-class Average Precision
        aps = []
        for i in range(y_true.shape[1]):
            if y_true[:, i].sum() > 0:  # Skip classes with no positive samples
                ap = average_precision_score(y_true[:, i], y_pred_proba[:, i])
                aps.append(ap)

        cmap = np.mean(aps)
        assert 0 <= cmap <= 1
        print(f"cmAP: {cmap:.3f}, F1 micro: {f1_micro:.3f}, F1 macro: {f1_macro:.3f}")
