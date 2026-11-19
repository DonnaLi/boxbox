"""Replay a race against the server at N x real time and record latency.

    python -m boxbox.bench --speed 1 10 50 --synthetic
    python -m boxbox.bench --speed 1 10 50 --race 2024:Monza

At the end of every lap, all cars send a request at once (that burst is
the realistic worst case). Results go to results/bench.csv.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import time
from pathlib import Path

import httpx
import numpy as np

from .data import load_race, synthetic_race

LAP_SECONDS = 90.0  # one real lap ~ 90s; replay waits LAP_SECONDS / speed between laps


def lap_requests(laps, total_laps):
    """Yield, for each lap number, the list of request payloads for all cars."""
    for lap_no, group in laps.sort_values("LapNumber").groupby("LapNumber"):
        if lap_no < 3:
            continue
        payloads = []
        for _, r in group.iterrows():
            prev = laps[(laps.Driver == r.Driver) & (laps.LapNumber.between(lap_no - 2, lap_no))]
            times = prev["LapTime"].dropna().to_numpy(float)
            if np.isnan(r.LapTime) or len(times) == 0:
                continue
            payloads.append(dict(driver=str(r.Driver), lap_number=int(lap_no), total_laps=total_laps,
                                 tyre_life=float(r.TyreLife), stint=int(r.Stint),
                                 position=int(r.Position), compound=str(r.Compound),
                                 last_lap_s=float(r.LapTime), rolling3_lap_s=float(times.mean())))
        yield payloads


async def run(url: str, laps, total_laps: int, speed: float, max_laps: int | None):
    latencies = []
    async with httpx.AsyncClient(base_url=url, timeout=10) as client:
        async def one(p):
            t0 = time.perf_counter()
            r = await client.post("/predict", json=p)
            r.raise_for_status()
            latencies.append((time.perf_counter() - t0) * 1000)

        for i, payloads in enumerate(lap_requests(laps, total_laps)):
            if max_laps and i >= max_laps:
                break
            start = time.perf_counter()
            await asyncio.gather(*(one(p) for p in payloads))
            await asyncio.sleep(max(0.0, LAP_SECONDS / speed - (time.perf_counter() - start)))
    return np.array(latencies)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--speed", type=float, nargs="+", default=[1, 10, 50])
    ap.add_argument("--race", default="2024:Monza")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-laps", type=int, default=None, help="cap laps per run (quick tests)")
    ap.add_argument("--label", default="default", help="tag for this config, e.g. batch20_wait5")
    args = ap.parse_args()

    if args.synthetic:
        laps, total = synthetic_race(seed=99)
    else:
        year, event = args.race.split(":", 1)
        laps, total = load_race(int(year), event)

    out = Path("results/bench.csv")
    out.parent.mkdir(exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "speed", "n", "p50_ms", "p95_ms", "p99_ms", "max_ms"])
        for s in args.speed:
            lat = asyncio.run(run(args.url, laps, total, s, args.max_laps))
            row = [args.label, s, len(lat), *(round(float(np.percentile(lat, q)), 2) for q in (50, 95, 99)),
                   round(float(lat.max()), 2)]
            w.writerow(row)
            print(dict(zip(["label", "speed", "n", "p50_ms", "p95_ms", "p99_ms", "max_ms"], row)))


if __name__ == "__main__":
    main()
