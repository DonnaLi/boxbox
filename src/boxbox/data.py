"""Load race laps from FastF1 (or generate synthetic laps for offline dev)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(".fastf1-cache")

LAP_COLUMNS = [
    "Driver", "LapNumber", "LapTime", "TyreLife", "Stint",
    "Position", "Compound", "TrackStatus", "IsPitLap",
]


def load_race(year: int, event: str) -> tuple[pd.DataFrame, int]:
    """Return (laps, total_laps) for a race, normalised to LAP_COLUMNS.

    Data is cached locally: please don't hammer the community-run APIs.
    """
    import fastf1  # imported lazily so tests don't need network

    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    session = fastf1.get_session(year, event, "R")
    session.load(telemetry=False, weather=False, messages=False)
    laps = session.laps.copy()
    laps["LapTime"] = laps["LapTime"].dt.total_seconds()
    laps["IsPitLap"] = laps["PitInTime"].notna() | laps["PitOutTime"].notna()
    laps["TrackStatus"] = laps["TrackStatus"].astype(str)
    laps = laps[LAP_COLUMNS].dropna(subset=["Position", "TyreLife", "Stint"])
    return laps, int(laps["LapNumber"].max())


def synthetic_race(n_drivers: int = 20, total_laps: int = 57, seed: int = 0) -> tuple[pd.DataFrame, int]:
    """Plausible fake race: tyre degradation, fuel burn-off, one or two stops.

    Used for tests and for developing offline. Never report benchmark
    *accuracy* numbers from this, only latency/throughput.
    """
    rng = np.random.default_rng(seed)
    deg = {"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.04}
    rows = []
    for k in range(n_drivers):
        base = 92.0 + k * 0.08 + rng.normal(0, 0.2)
        stops = sorted(rng.choice(range(15, total_laps - 10), size=rng.integers(1, 3), replace=False))
        plan = ["SOFT", "MEDIUM", "HARD"] if len(stops) == 2 else [rng.choice(["SOFT", "MEDIUM"]), "HARD"]
        stint, tyre_life = 1, 1
        for lap in range(1, total_laps + 1):
            compound = plan[stint - 1]
            is_pit = lap in stops
            t = base + deg[compound] * tyre_life - 0.03 * lap + rng.normal(0, 0.25)
            if is_pit:
                t += 21.0
            rows.append(dict(
                Driver=f"D{k:02d}", LapNumber=lap, LapTime=t, TyreLife=tyre_life,
                Stint=stint, Position=k + 1, Compound=compound,
                TrackStatus="1", IsPitLap=is_pit,
            ))
            tyre_life += 1
            if is_pit:
                stint, tyre_life = stint + 1, 1
    return pd.DataFrame(rows, columns=LAP_COLUMNS), total_laps
