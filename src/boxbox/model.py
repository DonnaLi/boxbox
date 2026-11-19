"""Train a next-lap-time model and export it to ONNX.

Usage:
    python -m boxbox.model --races 2024:Bahrain 2024:Spain 2024:Monza
    python -m boxbox.model --synthetic          # offline dev
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from .data import load_race, synthetic_race
from .features import FEATURE_NAMES, N_FEATURES, build_training_frame

MODEL_DIR = Path("models")


def collect(races: list[str], synthetic: bool):
    frames = []
    if synthetic:
        for seed in range(6):
            frames.append(build_training_frame(*synthetic_race(seed=seed)))
    else:
        for spec in races:
            year, event = spec.split(":", 1)
            frames.append(build_training_frame(*load_race(int(year), event)))
    return frames


def train(frames, holdout_last: bool = True):
    """Hold out the *last race* rather than a random split, so the eval
    reflects predicting a race the model has never seen."""
    train_frames = frames[:-1] if holdout_last and len(frames) > 1 else frames
    test_x, test_y = frames[-1]
    X = np.concatenate([f[0] for f in train_frames])
    y = np.concatenate([f[1] for f in train_frames])

    model = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0)
    model.fit(X, y)

    # Baseline: "next lap = last lap". Any model has to beat this to be worth serving.
    baseline_mae = mean_absolute_error(test_y, test_x[:, FEATURE_NAMES.index("last_lap_s")])
    model_mae = mean_absolute_error(test_y, model.predict(test_x))
    return model, {"baseline_mae_s": round(float(baseline_mae), 4),
                   "model_mae_s": round(float(model_mae), 4),
                   "n_train": int(len(y)), "n_test": int(len(test_y))}


def export_onnx(model, path: Path) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    onx = convert_sklearn(model, initial_types=[("x", FloatTensorType([None, N_FEATURES]))],
                          target_opset=17)
    path.write_bytes(onx.SerializeToString())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--races", nargs="*", default=["2024:Bahrain", "2024:Spain", "2024:Monza"],
                    help="YEAR:EVENT pairs; the last one is the held-out test race")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    model, metrics = train(collect(args.races, args.synthetic))
    export_onnx(model, MODEL_DIR / "lap_time.onnx")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
