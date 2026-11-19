# boxbox 🏁

**Real-time F1 race-strategy inference.** boxbox predicts every car's next lap time as a race unfolds, flags when a driver's pace is dropping off, and serves those predictions fast enough to matter. It's tuned under realistic race load: all 20 cars reporting at once, every lap.

> "Box, box" is the radio call telling a driver to pit this lap. Strategy calls in F1 are made in seconds, on incomplete information. This project is about the engineering that makes a model useful in that window.

<p align="center">
  <img src="https://media.giphy.com/media/ehTz6odmTnLhSifWml/giphy.gif" alt="Charles Leclerc, visibly frustrated" width="420">
  <br>
  <sub>How every driver feels about a late strategy call. GIF: <a href="https://giphy.com/gifs/f1-angry-fuming-azerbaijan-grand-prix-ehTz6odmTnLhSifWml">Formula 1 on GIPHY</a></sub>
</p>

---

## Why I built this

**Monaco 2022.** Charles Leclerc started his home race on pole and was leading comfortably. Then two strategy calls took the win away from him.

First, on a drying track, Red Bull brought Sergio Pérez in for intermediates. Ferrari underestimated how fast he'd be on them, and he gained enough in roughly one lap to jump Leclerc. That was a **lap-time prediction error**.

Three laps later, Ferrari double-stacked Leclerc behind his teammate for slicks. They realised the gap was too tight and radioed "stay out," but by then he was already committed to the pit lane. He waited in the box, lost more places, and finished fourth. That was a **decision that arrived too late**.

<!-- After running `python -m boxbox.viz`, uncomment the line below:
![Monaco 2022 replay: real car positions, laps 15–23](docs/monaco2022_track.gif)
-->

Those two failures are this project. In F1, a strategy call has a deadline. A driver might be 15 seconds from pit entry when the pit wall has to decide. A prediction that arrives after the car has passed pit entry isn't slightly worse. It's worthless.

That's what interested me. Most ML projects stop at "the model is accurate on a test set." In practice, the hard part is often **inference**: getting a prediction out reliably, under load, inside a latency budget, at a cost that makes sense. I wanted a project where that part couldn't be skipped. A live race has it built in. Requests come in synchronized bursts (every car crosses the line within about 90 seconds), the deadline is real, and the data keeps changing: tyres wear, fuel burns off, safety cars reset everything.

So boxbox is deliberately an **inference-first** project. The model is intentionally simple: gradient-boosted trees, exported to ONNX. The questions I'm actually trying to answer are engineering ones:

- How much does micro-batching buy when 20 requests land at once, and what does it cost in tail latency?
- Where is the time going: model compute, serialization, the network, or Python?
- How does p99 latency hold up when I replay a race at 10× or 50× real time?
- How badly does a model trained on older seasons break on the very different 2026 cars, and how would I detect that in production?

The end goal is a replay of Monaco 2022 where boxbox flags Pérez's undercut threat before Ferrari reacted.

<!-- ✏️ Donna: optional. Add a sentence in your own words about watching that race, or what F1 means to you. -->

## How it works

```
 FastF1 race data ──► replay engine ──► 20 concurrent requests / lap
                       (1×, 10×, 50×)          │
                                               ▼
                                     FastAPI server
                                               │
                                     micro-batcher  (≤ max_batch, ≤ max_wait_ms)
                                               │
                                     ONNX Runtime  ──► next-lap prediction + pace-drop signal
```

| Module | What it does |
|---|---|
| `data.py` | Loads race laps through FastF1 (cached locally), or generates a synthetic race for offline development |
| `features.py` | Builds model inputs: tyre age, compound, stint, race progress, recent pace and pace trend |
| `model.py` | Trains on past races, **holds out an entire race** for evaluation, compares against a "next lap = last lap" baseline, exports to ONNX |
| `batcher.py` | Async dynamic micro-batcher: the core inference-engineering piece |
| `server.py` | FastAPI `/predict` and `/health` endpoints (`/health` reports the average batch size actually achieved) |
| `bench.py` | Replays a race against the server at N× speed and logs p50/p95/p99 latency to `results/bench.csv` |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

# train (real data: downloads and caches the races on first run)
python -m boxbox.model --races 2024:Bahrain 2024:Spain 2024:Monza
# or fully offline
python -m boxbox.model --synthetic

# serve
uvicorn boxbox.server:app --port 8000

# benchmark (second terminal)
python -m boxbox.bench --race 2024:Monza --speed 1 10 50 --label batch20_wait5
```

To make the Monaco 2022 replay GIFs. Car dots follow the real X/Y positions from F1 live timing; the running order, tyres and pit stops are all computed from the data:

```bash
python -m boxbox.viz                      # track replay → docs/monaco2022_track.gif
python -m boxbox.viz --view gaps          # gap chart    → docs/monaco2022_gaps.gif
python -m boxbox.viz --race 2024:Monza --drivers LEC NOR PIA --laps 20 30
```

To compare batching configurations, restart the server with different settings and benchmark again:

```bash
BOXBOX_MAX_BATCH=1  uvicorn boxbox.server:app   # no batching
BOXBOX_MAX_BATCH=20 BOXBOX_MAX_WAIT_MS=2 uvicorn boxbox.server:app
```

## Results

_Coming soon: latency vs. replay speed for each batching configuration, and model MAE against the baseline on a held-out race._

## Roadmap

See [`docs/milestones.md`](docs/milestones.md).

## Data & credits

Timing data comes from the community-built [FastF1](https://github.com/theOehrly/Fast-F1) library. This is an unofficial, non-commercial project, not affiliated with Formula 1 or any team. Data is cached locally to avoid putting load on the free community services.
