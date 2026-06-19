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
    cnn.main()
    return {}  # CNN prints its own accuracy; no structured metrics returned yet


MODEL_RUNNERS = {
    "logreg":       run_logreg,
    "nn":           run_nn,
    "longitudinal": run_longitudinal,
    "cnn":          run_cnn,
}


def print_summary(all_results: dict) -> None:
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'Feature Set':<20} {'CV AUC':>12} {'Test AUC':>10} {'Accuracy':>10}")
    print("-" * 84)

    for section, section_results in all_results.items():
        if not section_results:
            continue
        for model_name, metrics in section_results.items():
            if not isinstance(metrics, dict):
                continue
            cv   = f"{metrics.get('cv_auc_mean', 0):.4f} ± {metrics.get('cv_auc_std', 0):.4f}"
            tauc = f"{metrics.get('test_auc', 0):.4f}"
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
    for name in to_run:
        result = MODEL_RUNNERS[name]()
        all_results.update(result)

    save_results(all_results)
    if len(to_run) > 1:
        print_summary(all_results)


if __name__ == "__main__":
    main()
