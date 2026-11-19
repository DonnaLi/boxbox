"""Hairline-style race replay GIFs, built from real timing + car position data.

    python -m boxbox.viz                       # Monaco 2022 track replay, laps 15-23
    python -m boxbox.viz --view gaps           # gap-to-leader chart instead
    python -m boxbox.viz --race 2024:Monza --drivers LEC NOR PIA --laps 20 30
    python -m boxbox.viz --synthetic           # offline test (fake oval)

Writes docs/<event><year>_<view>.gif. Car positions are the real X/Y
from F1 live timing (via FastF1); every caption (pit stops, tyres,
running order) is computed from the data, nothing is hand-placed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .data import CACHE_DIR, synthetic_race  # noqa: E402

# ── hairline style ────────────────────────────────────────────────
PAPER, INK, MUTED, RULE = "#FAFAF7", "#111111", "#6B6B66", "#D9D9D4"
ACCENT = "#D9480F"                                  # alerts / the story car
DRIVER_INKS = [INK, "#2F5BEA", ACCENT, "#7A7A75", "#0F8A6B", "#8B5CF6"]
COMPOUND_LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}
HAIR = 0.8  # line width in points: thin, crisp

plt.rcParams.update({
    "font.family": "monospace",
    "font.monospace": ["JetBrains Mono", "SF Mono", "Menlo", "DejaVu Sans Mono"],
    "font.size": 8, "font.weight": "normal",
    "axes.linewidth": HAIR, "axes.edgecolor": RULE,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.width": HAIR, "ytick.major.width": HAIR,
})


@dataclass
class Race:
    title: str
    laps: pd.DataFrame          # Driver, LapNumber, StartS, TimeS, Compound, PitInS
    total_laps: int
    pos: dict = field(default_factory=dict)       # driver -> (t, x, y)
    outline: tuple | None = None                  # (x, y) of one clean lap
    source: str = "data: F1 live timing via FastF1 · unofficial"


# ── loading ───────────────────────────────────────────────────────
def _rotate(x, y, deg):
    a = np.deg2rad(deg)
    return x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)


def load_real(year: int, event: str, drivers: list[str], with_pos: bool) -> Race:
    import fastf1

    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    s = fastf1.get_session(year, event, "R")
    s.load(telemetry=with_pos, weather=False, messages=False)
    L = s.laps
    laps = pd.DataFrame({
        "Driver": L["Driver"], "LapNumber": L["LapNumber"].astype(int),
        "StartS": L["LapStartTime"].dt.total_seconds(), "TimeS": L["Time"].dt.total_seconds(),
        "Compound": L["Compound"].fillna("UNKNOWN").astype(str),
        "PitInS": L["PitInTime"].dt.total_seconds(),
    }).dropna(subset=["TimeS"])
    race = Race(f"{event} {year}", laps, int(laps.LapNumber.max()))
    if not with_pos:
        return race

    try:
        rot = float(s.get_circuit_info().rotation)
    except Exception:
        rot = 0.0
    for d in drivers:
        num = str(s.get_driver(d)["DriverNumber"])
        p = s.pos_data[num]
        p = p[p["Status"] == "OnTrack"] if "Status" in p else p
        x, y = _rotate(p["X"].to_numpy(float), p["Y"].to_numpy(float), rot)
        race.pos[d] = (p["SessionTime"].dt.total_seconds().to_numpy(), x, y)
    fl = L.pick_fastest().get_pos_data()
    race.outline = _rotate(fl["X"].to_numpy(float), fl["Y"].to_numpy(float), rot)
    return race


def load_synthetic(drivers: list[str]) -> Race:
    laps, total = synthetic_race(n_drivers=len(drivers), seed=3)
    laps["Driver"] = laps["Driver"].map({f"D{i:02d}": d for i, d in enumerate(drivers)})
    laps = laps.sort_values(["Driver", "LapNumber"])
    laps["TimeS"] = laps.groupby("Driver")["LapTime"].cumsum()
    laps["StartS"] = laps["TimeS"] - laps["LapTime"]
    laps["PitInS"] = np.where(laps["IsPitLap"], laps["TimeS"] - 3, np.nan)
    race = Race("Synthetic oval (test only)", laps, total,
                source="synthetic test data · not a real race")
    th = np.linspace(0, 2 * np.pi, 400)
    race.outline = (np.cos(th) * 1000, np.sin(th) * 450)
    for d in drivers:
        dd = laps[laps.Driver == d]
        t = np.arange(0, dd.TimeS.max(), 0.5)
        lapf = np.interp(t, np.r_[0, dd.TimeS.to_numpy()], np.arange(len(dd) + 1))
        race.pos[d] = (t, np.cos(lapf * 2 * np.pi) * 1000, np.sin(lapf * 2 * np.pi) * 450)
    return race


# ── state at time t (all derived from the data) ───────────────────
def order_at(race: Race, drivers: list[str], t: float):
    rows = []
    for d in drivers:
        dd = race.laps[race.laps.Driver == d].sort_values("LapNumber")
        done = dd[dd.TimeS <= t]
        n = len(done)
        last = done.TimeS.iloc[-1] if n else 0.0
        cur = dd[dd.LapNumber == n + 1]
        comp = (cur.Compound.iloc[0] if len(cur) else (dd.Compound.iloc[-1] if len(dd) else "?"))
        age = int((dd.Compound.iloc[:n + 1][::-1] == comp).cumprod().sum()) if len(dd) else 0
        rows.append((d, n, last, comp, age))
    rows.sort(key=lambda r: (-r[1], r[2]))
    lap_now = min(rows[0][1] + 1, race.total_laps) if rows else 0
    return rows, lap_now


def pit_events(race: Race, drivers: list[str]):
    out = []
    for d in drivers:
        dd = race.laps[race.laps.Driver == d].sort_values("LapNumber").reset_index(drop=True)
        for i in range(len(dd) - 1):
            if pd.notna(dd.loc[i, "PitInS"]):
                out.append((float(dd.loc[i, "PitInS"]), int(dd.loc[i, "LapNumber"]), d,
                            dd.loc[i + 1, "Compound"]))
    return sorted(out)


def _ink(drivers, d):
    return DRIVER_INKS[drivers.index(d) % len(DRIVER_INKS)]


def _frame(fig, source="data: F1 live timing via FastF1 · unofficial"):
    fig.patch.set_facecolor(PAPER)
    fig.add_artist(Line2D([0.03, 0.97], [0.90, 0.90], lw=HAIR, color=INK))
    fig.add_artist(Line2D([0.03, 0.97], [0.07, 0.07], lw=HAIR, color=RULE))
    fig.text(0.03, 0.035, "boxbox · race replay", color=MUTED, fontsize=7)
    fig.text(0.97, 0.035, source, color=MUTED,
             fontsize=7, ha="right")


# ── view 1: track replay ──────────────────────────────────────────
def render_track(race: Race, drivers, lap_range, out: Path, fps=20, step_s=2.0) -> Path:
    L = race.laps[race.laps.Driver.isin(drivers)]
    t0 = L[L.LapNumber == lap_range[0]].StartS.min()
    t1 = L[L.LapNumber == lap_range[1]].TimeS.max()
    if np.isnan(t0):
        t0 = L.TimeS.min()
    times = np.arange(t0, t1, step_s)
    events = [e for e in pit_events(race, drivers) if t0 <= e[0] <= t1]

    fig = plt.figure(figsize=(9, 5.4), dpi=100)
    _frame(fig, race.source)
    title = fig.text(0.03, 0.925, "", color=INK, fontsize=10)
    ax = fig.add_axes([0.03, 0.10, 0.60, 0.77])
    ax.set_aspect("equal")
    ax.axis("off")
    ox, oy = race.outline
    ax.plot(ox, oy, color=INK, lw=HAIR * 1.6, solid_capstyle="round")
    ax.plot(ox, oy, color=PAPER, lw=HAIR * 0.5)          # inner hairline: double-rule track
    pad = 0.06 * (np.ptp(ox) + np.ptp(oy))
    ax.set_xlim(ox.min() - pad, ox.max() + pad)
    ax.set_ylim(oy.min() - pad, oy.max() + pad)

    dots = {d: ax.scatter([], [], s=34, color=_ink(drivers, d), edgecolor=PAPER,
                          linewidth=1.2, zorder=5) for d in drivers}
    tags = {d: ax.text(0, 0, d, fontsize=7, color=_ink(drivers, d), zorder=6) for d in drivers}

    # right panel: running order + event feed
    px = 0.67
    fig.add_artist(Line2D([px - 0.015, px - 0.015], [0.10, 0.87], lw=HAIR, color=RULE))
    fig.text(px, 0.84, "RUNNING ORDER", color=MUTED, fontsize=7)
    row_y = [0.79 - i * 0.055 for i in range(len(drivers))]
    rows_txt = [fig.text(px, y, "", fontsize=9, color=INK) for y in row_y]
    for y in row_y:
        fig.add_artist(Line2D([px, 0.97], [y - 0.018] * 2, lw=HAIR * 0.6, color=RULE))
    ey = row_y[-1] - 0.09
    fig.text(px, ey, "EVENTS", color=MUTED, fontsize=7)
    feed = [fig.text(px, ey - 0.05 - i * 0.045, "", fontsize=8, color=INK) for i in range(4)]

    def draw(k):
        t = times[k]
        rows, lap_now = order_at(race, drivers, t)
        title.set_text(f"{race.title}   ·   lap {lap_now}/{race.total_laps}")
        for d in drivers:
            ts, xs, ys = race.pos[d]
            if t < ts[0] or t > ts[-1]:
                continue
            x, y = np.interp(t, ts, xs), np.interp(t, ts, ys)
            dots[d].set_offsets([[x, y]])
            tags[d].set_position((x + pad * 0.25, y + pad * 0.25))
        for i, (d, _, _, comp, age) in enumerate(rows):
            rows_txt[i].set_text(f"P{i + 1}  {d:<4}  {COMPOUND_LETTER.get(comp, '?')}{age:>3}")
            rows_txt[i].set_color(_ink(drivers, d))
        recent = [e for e in events if e[0] <= t][-4:][::-1]
        for i, line in enumerate(feed):
            if i < len(recent):
                _, lap, d, comp = recent[i]
                line.set_text(f"L{lap:<3} {d} box → {comp.lower()}")
                line.set_color(ACCENT if i == 0 and t - recent[0][0] < 20 else MUTED)
            else:
                line.set_text("")
        return []

    anim = FuncAnimation(fig, draw, frames=len(times), interval=1000 // fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


# ── view 2: gap chart ─────────────────────────────────────────────
def render_gaps(race: Race, drivers, lap_range, out: Path, fps=3) -> Path:
    df = race.laps[race.laps.Driver.isin(drivers) & race.laps.LapNumber.between(*lap_range)].copy()
    df["Gap"] = df["TimeS"] - df.groupby("LapNumber")["TimeS"].transform("min")
    gaps = df.pivot_table(index="LapNumber", columns="Driver", values="Gap").reindex(columns=drivers)
    events = [(lap, d, c) for _, lap, d, c in pit_events(race, drivers) if lap in gaps.index]
    laps = list(gaps.index)
    ymax = float(min(np.nanmax(gaps.to_numpy()), 40)) + 2

    fig = plt.figure(figsize=(9, 5), dpi=100)
    _frame(fig, race.source)
    title = fig.text(0.03, 0.925, "", color=INK, fontsize=10)
    ax = fig.add_axes([0.08, 0.16, 0.86, 0.70])

    def draw(k):
        ax.clear()
        ax.set_facecolor(PAPER)
        cur = laps[: k + 1]
        title.set_text(f"{race.title}   ·   gap to leader   ·   lap {cur[-1]}")
        for d in drivers:
            c, y = _ink(drivers, d), gaps.loc[cur, d]
            ax.plot(cur, y, color=c, lw=HAIR * 1.4)
            if pd.notna(y.iloc[-1]):
                ax.scatter([cur[-1]], [y.iloc[-1]], s=16, color=c, zorder=3)
                ax.annotate(d, (cur[-1], y.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                            color=c, fontsize=8, va="center")
        for lap, d, comp in events:
            if lap <= cur[-1] and pd.notna(gaps.loc[lap, d]):
                ax.axvline(lap, color=RULE, lw=HAIR, zorder=0)
                ax.annotate(f"{d} box → {comp.lower()}", (lap, 0), xytext=(3, 4),
                            textcoords="offset points", color=MUTED, fontsize=7, rotation=90,
                            va="top")
        ax.set_xlim(lap_range[0], lap_range[1] + 2)
        ax.set_ylim(ymax, -1)
        ax.set_xlabel("lap", color=MUTED)
        ax.set_ylabel("gap to leader (s)", color=MUTED)
        ax.grid(axis="y", color=RULE, lw=HAIR * 0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    anim = FuncAnimation(fig, draw, frames=len(laps), interval=1000 // fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", default="2022:Monaco")
    ap.add_argument("--drivers", nargs="+", default=["LEC", "SAI", "PER", "VER"])
    ap.add_argument("--laps", nargs=2, type=int, default=None,
                    help="lap range; default 15-23 for track, 1-30 for gaps")
    ap.add_argument("--view", choices=["track", "gaps"], default="track")
    ap.add_argument("--step", type=float, default=2.0, help="race seconds per frame (track view)")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    lap_range = tuple(a.laps or ((15, 23) if a.view == "track" else (1, 30)))
    if a.synthetic:
        race, name = load_synthetic(a.drivers), "synthetic_test"
    else:
        year, event = a.race.split(":", 1)
        race = load_real(int(year), event, a.drivers, with_pos=a.view == "track")
        name = f"{event.lower().replace(' ', '_')}{year}"
    out = Path(a.out or f"docs/{name}_{a.view}.gif")
    if a.view == "track":
        print(render_track(race, a.drivers, lap_range, out, fps=a.fps or 20, step_s=a.step))
    else:
        print(render_gaps(race, a.drivers, lap_range, out, fps=a.fps or 3))


if __name__ == "__main__":
    main()
