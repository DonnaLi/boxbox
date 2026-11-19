import asyncio

import numpy as np

from boxbox.batcher import MicroBatcher
from boxbox.data import synthetic_race
from boxbox.features import N_FEATURES, LapState, build_training_frame
from boxbox.model import train


def test_lapstate_vector_shape_and_onehot():
    s = LapState("VER", 10, 57, 8, 1, 1, "MEDIUM", 93.1, 93.0)
    v = s.to_vector()
    assert v.shape == (N_FEATURES,)
    assert v[-4] == 1.0 and v[-5:].sum() == 1.0  # MEDIUM is 2nd of 5 compounds


def test_training_frame_skips_pit_laps():
    laps, total = synthetic_race(n_drivers=3, seed=1)
    X, y = build_training_frame(laps, total)
    assert len(X) == len(y) > 0
    assert y.max() < 110  # no 21s-penalty pit laps leaked into targets


def test_model_beats_last_lap_baseline():
    frames = [build_training_frame(*synthetic_race(seed=s)) for s in range(4)]
    _, m = train(frames)
    assert m["model_mae_s"] < m["baseline_mae_s"]


def test_batcher_groups_concurrent_requests():
    async def go():
        b = MicroBatcher(lambda x: x.sum(axis=1), max_batch=20, max_wait_ms=20)
        b.start()
        xs = [np.full(3, i, np.float32) for i in range(10)]
        out = await asyncio.gather(*(b.submit(x) for x in xs))
        await b.stop()
        return out, b.batch_sizes
    out, sizes = asyncio.run(go())
    assert out == [3.0 * i for i in range(10)]   # results go back to the right caller
    assert max(sizes) > 1                          # and they were actually batched
