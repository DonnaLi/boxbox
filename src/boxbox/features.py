"""Feature engineering: turn raw lap rows into model inputs.

Kept free of FastF1 imports so it can be unit-tested and used by the
server without pulling in the data stack.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

FEATURE_NAMES = [
    "lap_number",
    "race_progress",      # lap_number / total_laps, a rough fuel-load proxy
    "tyre_life",
    "stint",
    "position",
    "last_lap_s",
    "rolling3_lap_s",
    "lap_delta_s",        # last lap minus rolling mean: is the car fading?
    *[f"compound_{c.lower()}" for c in COMPOUNDS],
]
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class LapState:
    """One car's state at the end of a lap: what the server receives."""
    driver: str
    lap_number: int
    total_laps: int
    tyre_life: float
    stint: int
    position: int
    compound: str
    last_lap_s: float
    rolling3_lap_s: float

    def to_vector(self) -> np.ndarray:
        onehot = [1.0 if self.compound.upper() == c else 0.0 for c in COMPOUNDS]
        return np.array(
            [
                self.lap_number,
                self.lap_number / max(self.total_laps, 1),
                self.tyre_life,
                self.stint,
                self.position,
                self.last_lap_s,
                self.rolling3_lap_s,
                self.last_lap_s - self.rolling3_lap_s,
                *onehot,
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def build_training_frame(laps: pd.DataFrame, total_laps: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) where y is the *next* lap time in seconds.

    Expects FastF1-style columns: Driver, LapNumber, LapTime (seconds float),
    TyreLife, Stint, Position, Compound, TrackStatus, IsPitLap (bool).
    Laps that are pit laps or not under green flag are dropped as targets,
    because they're not what a strategy model should learn "pace" from.
    """
    rows_x, rows_y = [], []
    for _, d in laps.sort_values(["Driver", "LapNumber"]).groupby("Driver"):
        d = d.reset_index(drop=True)
        times = d["LapTime"].to_numpy(dtype=float)
        for i in range(1, len(d) - 1):
            nxt = d.iloc[i + 1]
            if nxt["IsPitLap"] or str(nxt["TrackStatus"]) != "1" or np.isnan(times[i + 1]):
                continue
            if np.isnan(times[i]):
                continue
            window = times[max(0, i - 2): i + 1]
            window = window[~np.isnan(window)]
            cur = d.iloc[i]
            state = LapState(
                driver=str(cur["Driver"]),
                lap_number=int(cur["LapNumber"]),
                total_laps=total_laps,
                tyre_life=float(cur["TyreLife"]),
                stint=int(cur["Stint"]),
                position=int(cur["Position"]),
                compound=str(cur["Compound"]),
                last_lap_s=float(times[i]),
                rolling3_lap_s=float(window.mean()),
            )
            rows_x.append(state.to_vector())
            rows_y.append(times[i + 1])
    if not rows_x:
        return np.empty((0, N_FEATURES), np.float32), np.empty((0,), np.float32)
    return np.stack(rows_x), np.array(rows_y, dtype=np.float32)
