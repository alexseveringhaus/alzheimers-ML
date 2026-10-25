"""Unified entrypoint for all Alzheimer's classification models.

Usage:
    python main.py                  # run all models
    python main.py --model logreg   # run one model
    python main.py --model nn
    python main.py --model longitudinal
    python main.py --model cnn

Results are saved to results/ and checkpoints/ at the project root.
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
CKPT_DIR    = Path(__file__).parent / "checkpoints"


def run_logreg() -> dict:
    from LogisticRegression.model import model
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION + RANDOM FOREST")
    print("=" * 60)
    return model()


def run_nn() -> dict:
    from NeuralNet.neural_network import model
    print("\n" + "=" * 60)
    print("FEEDFORWARD NEURAL NETWORK")
    print("=" * 60)
    return model()


def run_longitudinal() -> dict:
    from LongitudinalTauBC.model import model
    print("\n" + "=" * 60)
    print("LONGITUDINAL TAU — LOGISTIC REGRESSION")
    print("=" * 60)
    return model()


def run_cnn() -> dict:
    from importlib import import_module
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "2D-CNN"))
    cnn = import_module("cnn")
    print("\n" + "=" * 60)
    print("2D CONVOLUTIONAL NEURAL NETWORK")
    print("=" * 60)
    return cnn.main()


MODEL_RUNNERS = {
    "logreg":       run_logreg,
    "nn":           run_nn,
    "longitudinal": run_longitudinal,
    "cnn":          run_cnn,
}


def iter_metrics(name: str, node: dict):
    """Yield (model_name, metrics) for every metrics leaf, at any nesting depth.

    Results are either flat ({"NeuralNetwork": {metrics}}) or nested
    ({"combined": {"LogisticRegression": {metrics}}}). A leaf has test_accuracy.
    """
    if "test_accuracy" in node:
        yield name, node
        return
    for child_name, child in node.items():
        if isinstance(child, dict):
            yield from iter_metrics(child_name, child)


def print_summary(all_results: dict) -> None:
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'Feature Set':<20} {'CV AUC':>12} {'Test AUC':>10} {'Accuracy':>10}")
    print("-" * 84)

    for model_name, metrics in iter_metrics("", all_results):
        cv_mean = metrics.get("cv_auc_mean")
        cv_std  = metrics.get("cv_auc_std")
        cv   = f"{cv_mean:.4f} ± {cv_std:.4f}" if cv_mean is not None else "N/A"
        tauc_val = metrics.get("test_auc")
        tauc = f"{tauc_val:.4f}" if tauc_val is not None else "N/A"
        acc  = f"{metrics.get('test_accuracy', 0):.4f}"
        feat = metrics.get('feature_set', '')
        print(f"{model_name:<30} {feat:<20} {cv:>12} {tauc:>10} {acc:>10}")


def save_results(all_results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Alzheimer's classification models.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_RUNNERS.keys()),
        default=None,
        help="Which model to run (default: all)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    to_run = [args.model] if args.model else list(MODEL_RUNNERS.keys())

    all_results: dict = {}
    metrics_path = RESULTS_DIR / "metrics.json"
    if args.model and metrics_path.exists():
        # Single-model run: keep the other models' results, replace only this model's entries
        all_results = json.loads(metrics_path.read_text())

    for name in to_run:
        result = MODEL_RUNNERS[name]()
        all_results.update(result)

    save_results(all_results)
    if len(to_run) > 1:
        print_summary(all_results)


if __name__ == "__main__":
    main()
