"""Dynamic micro-batcher.

Requests arrive one car at a time. Running the model once per request
wastes most of each call on overhead, so we hold requests for up to
`max_wait_ms` (or until `max_batch` arrive) and run them together.

The trade-off this exposes, and the thing the benchmark measures:
bigger batches = more throughput, but every request pays up to
`max_wait_ms` of extra latency.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable

import numpy as np


class MicroBatcher:
    def __init__(self, predict_fn: Callable[[np.ndarray], np.ndarray],
                 max_batch: int = 20, max_wait_ms: float = 5.0):
        self.predict_fn = predict_fn
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.queue: asyncio.Queue = asyncio.Queue()
        self.batch_sizes: list[int] = []
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def submit(self, x: np.ndarray) -> float:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((x, fut))
        return await fut

    async def _loop(self) -> None:
        while True:
            x, fut = await self.queue.get()
            items = [(x, fut)]
            deadline = time.perf_counter() + self.max_wait
            while len(items) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            batch = np.stack([i[0] for i in items]).astype(np.float32)
            try:
                preds = self.predict_fn(batch)
                for (_, f), p in zip(items, preds):
                    if not f.done():
                        f.set_result(float(p))
            except Exception as e:  # never leave a caller hanging
                for _, f in items:
                    if not f.done():
                        f.set_exception(e)
            self.batch_sizes.append(len(items))
