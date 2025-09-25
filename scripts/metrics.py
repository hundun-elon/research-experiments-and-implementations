import argparse
import json
import numpy as np
from typing import List, Dict, Any, Optional

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ----------------------------
# Metrics
# ----------------------------
def quadratic_weighted_kappa(y_true, y_pred, min_rating=None, max_rating=None):
    """
    Quadratic Weighted Kappa (QWK) to measure inter-rater reliability.
    """
    assert len(y_true) == len(y_pred)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(np.rint(y_pred), dtype=int)  # round predictions

    if min_rating is None:
        min_rating = min(y_true.min(), y_pred.min())
    if max_rating is None:
        max_rating = max(y_true.max(), y_pred.max())

    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = np.zeros((num_ratings, num_ratings), dtype=float)

    for a, b in zip(y_true, y_pred):
        conf_mat[a - min_rating, b - min_rating] += 1

    # expected matrix
    hist_true = np.bincount(y_true - min_rating, minlength=num_ratings)
    hist_pred = np.bincount(y_pred - min_rating, minlength=num_ratings)
    expected = np.outer(hist_true, hist_pred) / len(y_true)

    # weight matrix
    W = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            W[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)

    kappa = 1 - (np.sum(W * conf_mat) / np.sum(W * expected))
    return kappa


def expected_calibration_error(y_true: List[int], y_prob: List[float], n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if np.any(bin_mask):
            acc = np.mean(y_true[bin_mask])
            conf = np.mean(y_prob[bin_mask])
            ece += (np.sum(bin_mask) / len(y_true)) * abs(acc - conf)
    return ece


def brier_score(y_true: List[int], y_prob: List[float]) -> float:
    """
    Brier score for probabilistic predictions.
    """
    y_true = np.array(y_true, dtype=float)
    y_prob = np.array(y_prob, dtype=float)
    return np.mean((y_prob - y_true) ** 2)


def accuracy_within_tolerance(y_true, y_pred, tolerance=1):
    """
    % of predictions within ±tolerance of human score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred) <= tolerance)


# ----------------------------
# Aggregator
# ----------------------------
def evaluate_predictions(data: List[Dict[str, Any]], max_score: Optional[int] = None) -> Dict[str, float]:
    """
    Compute all metrics from a list of predictions in standard format:
    [
      {"id": "1", "human_score": 5, "model_score": 4, "confidence": 0.7},
      ...
    ]
    """
    y_true = np.array([d["human_score"] for d in data])
    y_pred = np.array([d["model_score"] for d in data])
    confidences = [d.get("confidence") for d in data if d.get("confidence") is not None]

    results = {}

    # Basic metrics
    results["MAE"] = float(mean_absolute_error(y_true, y_pred))
    results["RMSE"] = float(mean_squared_error(y_true, y_pred, squared=False))
    results["QWK"] = float(quadratic_weighted_kappa(y_true, y_pred))
    results["Accuracy_±1"] = float(accuracy_within_tolerance(y_true, y_pred, tolerance=1))
    results["Accuracy_±2"] = float(accuracy_within_tolerance(y_true, y_pred, tolerance=2))

    # Rank correlation
    rho, _ = spearmanr(y_true, y_pred)
    results["Spearman_rho"] = float(rho)

    # Calibration
    if len(confidences) == len(y_true):
        results["ECE"] = float(expected_calibration_error(y_true, confidences))
        results["Brier"] = float(brier_score(y_true, confidences))

    return results


# ----------------------------
# Helpers
# ----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL into list of dicts."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def log_to_wandb(results: Dict[str, float]):
    """Optionally log metrics to W&B if available."""
    if WANDB_AVAILABLE:
        wandb.log(results)
    else:
        print("[WARN] wandb not installed; skipping logging.")


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to predictions JSONL file")
    args = parser.parse_args()

    if args.input:
        data = load_jsonl(args.input)
        metrics = evaluate_predictions(data)
        print(json.dumps(metrics, indent=2))
        log_to_wandb(metrics)
    else:
        # Fallback demo
        sample = [
            {"id": "1", "human_score": 5, "model_score": 4, "confidence": 0.7},
            {"id": "2", "human_score": 3, "model_score": 3, "confidence": 0.6},
            {"id": "3", "human_score": 4, "model_score": 5, "confidence": 0.8},
        ]
        metrics = evaluate_predictions(sample)
        print(json.dumps(metrics, indent=2))
