"""Check independent anonymous HTTPS sessions from an external runner."""

import argparse
import asyncio
import json
import ssl
import statistics
import time
from pathlib import Path

import httpx


async def main(args):
    context = ssl.create_default_context()
    results = []
    for count in (10, 50, 100):
        samples = []
        cookies = []

        async def visitor(samples=samples, cookies=cookies):
            async with httpx.AsyncClient(base_url=args.url, verify=context, timeout=20, trust_env=False) as client:
                started = time.perf_counter()
                response = await client.post("/api/session")
                response.raise_for_status()
                assert "Secure" in response.headers["set-cookie"]
                cookies.append(client.cookies.get("llm_eval_guest"))
                for route in ("/api/jobs", "/api/config", "/api/health"):
                    response = await client.get(route)
                    response.raise_for_status()
                    row = response.json()
                    if route == "/api/jobs":
                        assert row == []
                    elif route == "/api/config":
                        assert row["role"] == "guest" and not row["server_key_available"]
                    else:
                        assert row["runtime"] == "cpp"
                samples.append(time.perf_counter() - started)

        await asyncio.gather(*(visitor() for _ in range(count)))
        assert len(set(cookies)) == count
        ordered = sorted(samples)
        results.append(
            {
                "users": count,
                "requests": count * 4,
                "errors": 0,
                "four_request_session_p50_ms": statistics.median(ordered) * 1000,
                "four_request_session_p95_ms": ordered[int(0.95 * (len(ordered) - 1))] * 1000,
            }
        )
        print(json.dumps(results[-1]), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if not arguments.url.startswith("https://"):
        parser.error("The public service URL must use HTTPS")
    asyncio.run(main(arguments))
