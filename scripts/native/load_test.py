"""Repeat the 10/50/100 visitor baseline; no model calls are made."""

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx


async def run(args):
    results = []
    limits = httpx.Limits(max_connections=220, max_keepalive_connections=220)
    async with httpx.AsyncClient(base_url=args.url, timeout=20, limits=limits, trust_env=not args.direct) as client:
        (await client.post("/api/login", json={"password": os.environ["LLM_EVAL_ADMIN_PASSWORD"]})).raise_for_status()
        jobs = (await client.get("/api/jobs")).json()
        job = next(row for row in jobs if row.get("mode") == "benchmark" and row["status"] == "COMPLETED")
        for users in (10, 50, 100):
            samples = []
            started = time.perf_counter()
            deadline = started + args.seconds

            async def visitor(deadline=deadline, samples=samples):
                etag = None
                while time.perf_counter() < deadline:
                    cycle = time.perf_counter()
                    for route in ("/api/jobs", "/api/jobs/" + job["id"]):
                        stamp = time.perf_counter()
                        headers = {"If-None-Match": etag} if args.conditional and etag and route != "/api/jobs" else {}
                        try:
                            response = await client.get(route, headers=headers)
                            if route != "/api/jobs":
                                etag = response.headers.get("etag", etag)
                            samples.append((time.perf_counter() - stamp, response.status_code, len(response.content)))
                        except httpx.HTTPError as exc:
                            samples.append((time.perf_counter() - stamp, type(exc).__name__, 0))
                    await asyncio.sleep(max(0, 2 - (time.perf_counter() - cycle)))

            await asyncio.gather(*(visitor() for _ in range(users)))
            elapsed = time.perf_counter() - started
            ordered = sorted(duration for duration, _, _ in samples)
            result = {
                "users": users,
                "duration_seconds": elapsed,
                "requests": len(samples),
                "rps": len(samples) / elapsed,
                "p50_ms": statistics.median(ordered) * 1000,
                "p95_ms": ordered[int(0.95 * (len(ordered) - 1))] * 1000,
                "max_ms": max(ordered) * 1000,
                "status_counts": dict(Counter(str(status) for _, status, _ in samples)),
                "bytes": sum(size for _, _, size in samples),
                "conditional": args.conditional,
                "direct": args.direct,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--direct", action="store_true", help="Bypass client environment proxies")
    asyncio.run(run(parser.parse_args()))
