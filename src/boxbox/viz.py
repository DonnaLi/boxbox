"""Animated gap chart of a race, lap by lap, with pit stops marked.

    python -m boxbox.viz                         # Monaco 2022, LEC/SAI/PER/VER, laps 1-30
    python -m boxbox.viz --race 2024:Monza --drivers LEC NOR --laps 1 53
    python -m boxbox.viz --synthetic             # offline test

Writes docs/<name>.gif. Every annotation comes from the timing data
(pit laps and tyre compounds), nothing is hand-placed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from .data import CACHE_DIR, synthetic_race  # noqa: E402

# Neutral palette (deliberately not team colours / branding)
PALETTE = ["#E4572E", "#F3A712", "#4C6EF5", "#2EC4B6", "#A0A0A0", "#9B5DE5"]
BG, FG, GRID = "#0F1115", "#E8E8E8", "#2A2D35"


def load_real(year: int, event: str) -> pd.DataFrame:
    """Columns: Driver, LapNumber, TimeS (session time at lap end), Compound, PitIn."""
    import fastf1

    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    s = fastf1.get_session(year, event, "R")
    s.load(telemetry=False, weather=False, messages=False)
    laps = s.laps
    return pd.DataFrame({
        "Driver": laps["Driver"],
        "LapNumber": laps["LapNumber"].astype(int),
        "TimeS": laps["Time"].dt.total_seconds(),
        "Compound": laps["Compound"].fillna("UNKNOWN").astype(str),
        "PitIn": laps["PitInTime"].notna(),
    }).dropna(subset=["TimeS"])


def load_synthetic() -> pd.DataFrame:
    laps, _ = synthetic_race(n_drivers=6, seed=3)
    laps = laps.sort_values(["Driver", "LapNumber"])
    laps["TimeS"] = laps.groupby("Driver")["LapTime"].cumsum()
    return laps.rename(columns={"IsPitLap": "PitIn"})[["Driver", "LapNumber", "TimeS", "Compound", "PitIn"]]


def gaps_to_leader(df: pd.DataFrame) -> pd.DataFrame:
    """Gap (s) behind whoever completed each lap first. Wide table: index=lap, cols=driver."""
    df = df.copy()
    df["Gap"] = df["TimeS"] - df.groupby("LapNumber")["TimeS"].transform("min")
    return df.pivot_table(index="LapNumber", columns="Driver", values="Gap")


def pit_events(df: pd.DataFrame, drivers: list[str]) -> list[tuple[int, str, str]]:
    """(lap, driver, new_compound) for each stop, read from the next lap's compound."""
    out = []
    for d in drivers:
        dd = df[df.Driver == d].sort_values("LapNumber").reset_index(drop=True)
        for i in range(len(dd) - 1):
            if dd.loc[i, "PitIn"]:
                out.append((int(dd.loc[i, "LapNumber"]), d, dd.loc[i + 1, "Compound"]))
    return out


def render(df: pd.DataFrame, drivers: list[str], lap_range: tuple[int, int],
           title: str, out: Path, fps: int = 3) -> Path:
    df = df[df.Driver.isin(drivers) & df.LapNumber.between(*lap_range)]
    # gaps relative to the leader among *all* cars on that lap would need the full field;
    # we compare within the selected drivers, which is what the story is about.
    gaps = gaps_to_leader(df).reindex(columns=drivers)
    events = pit_events(df, drivers)
    laps = list(gaps.index)
    ymax = float(min(gaps.max().max(), 40)) + 2

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    fig.patch.set_facecolor(BG)

    def draw(k: int):
        ax.clear()
        ax.set_facecolor(BG)
        cur = laps[: k + 1]
        for i, d in enumerate(drivers):
            c = PALETTE[i % len(PALETTE)]
            y = gaps.loc[cur, d]
            ax.plot(cur, y, color=c, lw=2.2)
            if pd.notna(y.iloc[-1]):
                ax.scatter([cur[-1]], [y.iloc[-1]], color=c, s=30, zorder=3)
                ax.annotate(d, (cur[-1], y.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                            color=c, fontsize=9, va="center")
        for lap, d, comp in events:
            if lap <= cur[-1] and d in gaps.columns and pd.notna(gaps.loc[lap, d]):
                c = PALETTE[drivers.index(d) % len(PALETTE)]
                ax.scatter([lap], [gaps.loc[lap, d]], marker="v", s=60, color=c,
                           edgecolor=FG, linewidth=0.6, zorder=4)
                ax.annotate(f"{d} pits → {comp.lower()}", (lap, gaps.loc[lap, d]),
                            xytext=(0, -14), textcoords="offset points", ha="center",
                            color=FG, fontsize=7, alpha=0.85)
        ax.set_xlim(lap_range[0], lap_range[1] + 2)
        ax.set_ylim(ymax, -2)  # leader at the top
        ax.set_xlabel("Lap", color=FG)
        ax.set_ylabel("Gap to leader (s)", color=FG)
        ax.set_title(f"{title}  ·  lap {cur[-1]}", color=FG, loc="left", fontsize=11)
        ax.tick_params(colors=FG)
        ax.grid(color=GRID, lw=0.6)
        for s in ax.spines.values():
            s.set_color(GRID)
        fig.text(0.99, 0.01, "data: FastF1 · boxbox", color="#777", fontsize=7, ha="right")

    anim = FuncAnimation(fig, draw, frames=len(laps), interval=1000 // fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", default="2022:Monaco")
    ap.add_argument("--drivers", nargs="+", default=["LEC", "SAI", "PER", "VER"])
    ap.add_argument("--laps", nargs=2, type=int, default=[1, 30])
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        df, drivers = load_synthetic(), ["D00", "D01", "D02", "D03"]
        title, name = "Synthetic race (test only)", "synthetic_test"
    else:
        year, event = args.race.split(":", 1)
        df, drivers = load_real(int(year), event), args.drivers
        title, name = f"{event} {year}", f"{event.lower().replace(' ', '_')}{year}"
    out = Path(args.out or f"docs/{name}.gif")
    print(render(df, drivers, tuple(args.laps), title, out, args.fps))


if __name__ == "__main__":
    main()
