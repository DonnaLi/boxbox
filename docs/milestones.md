# Milestones

Each milestone ends with something measurable that can go in the README.

## Week 1: Real data + honest baseline
- [ ] Train on 3–5 races from 2024 with `python -m boxbox.model`
- [ ] Record model MAE vs. the "next lap = last lap" baseline on a held-out race
- [ ] Plot predicted vs. actual lap times for a few drivers

## Week 2: Serving + first benchmark
- [ ] Run the server and benchmark at 1×, 10×, 50×
- [ ] Compare `MAX_BATCH=1` against `MAX_BATCH=20` at a few `MAX_WAIT_MS` values
- [ ] First chart: p99 latency vs. replay speed, one line per config

## Week 3: Find the bottleneck
- [ ] Profile one request end to end (py-spy or cProfile): model, JSON, network, or Python?
- [ ] Try one optimisation, e.g. multiple uvicorn workers, orjson, or a WebSocket that sends a whole lap in one message
- [ ] Measure the before/after

## Week 4: Model optimisation
- [ ] Compare sklearn predict vs. ONNX Runtime latency at batch sizes 1, 8, 20
- [ ] Try a smaller model (fewer trees) and plot the accuracy vs. latency trade-off

## Week 5: Distribution shift (2026 regulations)
- [ ] Evaluate the 2024-trained model on 2026 races
- [ ] Add a simple drift monitor (e.g. rolling residuals) to `/health`
- [ ] Retrain on 2026 data and compare

## Week 6: Polish + the Monaco 2022 replay
- [ ] Replay `2022:Monaco` and check whether boxbox would have flagged Pérez's undercut (lap 16–18) before Ferrari reacted. Note that it's a wet-to-dry race, so the model has to handle intermediates and changing track status
- [ ] Live dashboard replaying a race with predictions overlaid
- [ ] Write-up for donnali.xyz: what I measured, what surprised me, what I'd do next
