"""Twenty independent anonymous browser identities, real model calls, ownership checks."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx


async def run(url, output):
    clients = [httpx.AsyncClient(base_url=url, timeout=60) for _ in range(20)]
    tasks = ["gsm", "gpqa", "ifeval", "humanevalplus", "mbppplus"]
    started = time.monotonic()
    try:

        async def submit(index, client):
            (await client.post("/api/session")).raise_for_status()
            response = await client.post(
                "/api/jobs",
                json={
                    "mode": "benchmark",
                    "task": tasks[index % len(tasks)],
                    "limit": 1,
                    "workers": 1,
                    "max_output_tokens": 16384,
                    "api_key": os.environ["DEEPSEEK_API_KEY"],
                },
            )
            response.raise_for_status()
            return response.json()["id"]

        ids = await asyncio.gather(*(submit(index, client) for index, client in enumerate(clients)))
        assert len({client.cookies.get("llm_eval_guest") for client in clients}) == 20
        denied = 0
        for index, client in enumerate(clients):
            rows = (await client.get("/api/jobs")).json()
            assert [row["id"] for row in rows] == [ids[index]]
            other = ids[(index + 1) % len(ids)]
            for suffix in ("", "/report", "/artifacts"):
                assert (await client.get(f"/api/jobs/{other}{suffix}")).status_code == 404
                denied += 1
            assert (await client.post(f"/api/jobs/{other}/cancel")).status_code == 404
            denied += 1

        async def wait(client, identifier):
            for _ in range(300):
                response = await client.get(f"/api/jobs/{identifier}")
                response.raise_for_status()
                row = response.json()
                if row["status"] not in {"QUEUED", "RUNNING", "CANCELLING"}:
                    return row
                await asyncio.sleep(2)
            raise TimeoutError("Anonymous job exceeded ten minutes")

        rows = await asyncio.gather(
            *(wait(client, identifier) for client, identifier in zip(clients, ids, strict=True))
        )
        result = {
            "users": 20,
            "denied_cross_owner_operations": denied,
            "wall_seconds": time.monotonic() - started,
            "jobs": rows,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "users": 20,
                    "denied_cross_owner_operations": denied,
                    "statuses": [row["status"] for row in rows],
                    "attempts": sum(row.get("summary", {}).get("api_attempts", 0) for row in rows),
                }
            )
        )
        assert all(row["status"] == "COMPLETED" for row in rows)
    finally:
        await asyncio.gather(*(client.aclose() for client in clients))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/multiuser-live.json"))
    args = parser.parse_args()
    asyncio.run(run(args.url, args.output))
