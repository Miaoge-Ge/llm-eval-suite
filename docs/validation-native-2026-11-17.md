# Native service validation — 2026-11-17

The production web service now uses C++20 for HTTP, anonymous sessions, scheduling and model HTTP/SSE requests. Official Python graders and Docker code execution are preserved. Benchmark and API testing are available; model identity verification remains planned.

## Verified behavior

- **20 independent anonymous users on the deployed HTTPS site:** all 20 real model jobs completed, with exactly 20 requests across GSM8K, GPQA, IFEval, HumanEval+ and MBPP+. All 80 attempts to view, download or cancel a different user's job returned 404.
- **Capacity:** observed peaks were 20 running jobs, 16 model requests and four scorers. Limits are 20 jobs, 40 model requests and four scorers, with two running/queued jobs per owner. Additional admitted tasks wait. Shared resources can affect waiting time; user data and actions are isolated.
- **Memory:** the production service cgroup peaked at approximately 529 MiB during this validation. Docker container memory belongs to separate cgroups; this is not a whole-host peak. The host has four CPUs and 3,723 MiB RAM. Each code container is separately bounded.
- **Full scoring:** all 40 reference answers passed through the native HTTP path and real Docker, including complete Plus tests. Offline scoring preserved results without model calls. All 219 LiveCodeBench private asset hashes matched; its first problem passed public and private tests on the server.
- **Failures:** simulated rate limits and incomplete streams still produced exactly 30 attempts for API30. Reasoning-only length truncations were not retried, missing TPS stayed missing, benchmark retries retained both attempts, and scorer interruption recovery retained recorded usage.
- **Isolation/recovery:** checked same-IP guest separation, ownership after restart, tampered cookies, cross-origin requests, guest access to default keys, credential-free scorer environments, cancellation, interrupted jobs and global request/scorer limits.
- **Hard termination:** a request-start journal retains in-flight attempts across SIGKILL/restart. Interrupted delivery is marked unconfirmed with missing usage; API reports retain the attempt instead of silently counting zero. This boundary has a dedicated real-process restart test.

Guest identity is a random signed browser cookie, not an IP address. Different profiles behind one IP are separate; tabs in one profile share a workspace. Clearing the cookie loses access to previous anonymous tasks. See [native architecture](native-service.md).

## Server-side performance comparison

Both versions ran sequentially on the same server using the same saved Quick40 report. Each visitor requested the job list and report detail every two seconds for 20 seconds. No model calls ran during these measurements. These are loopback HTTP measurements, including the load generator; the C++ run did not use conditional/304 requests.

| Visitors | Requests per version | Python P95 | C++ P95 | Errors in either version |
| --- | ---: | ---: | ---: | ---: |
| 10 | 200 | 108.96 ms | 11.61 ms | 0 |
| 50 | 1,000 | 459.22 ms | 55.01 ms | 0 |
| 100 | 2,000 | 827.95 ms | 120.15 ms | 0 |

At 100 visitors, measured P95 fell about 85.5%. The change includes cached snapshots, compact JSON and omission of full summaries from list responses: decoded response bytes were 35,018,000 for Python and 24,663,000 for C++. This is an implementation comparison, not a controlled claim that C++ alone is 6.9 times faster. It also does not imply model generation or official grading is 6.9 times faster.

An initial native load run exposed idle HTTP connections occupying the blocking backend's fixed thread pool. The fix closes internal backend response connections while Caddy retains browser HTTPS connections. A regression test holds 100 connections open concurrently and verifies successful responses. A separate accounting race could briefly report five active scorers while only four processes existed; its counter lifetime was corrected and concurrent-job checks added.

## Public network measurements

The original Python HTTPS baseline from the Windows client recorded P95 of 182 / 892 / 3,493 ms for 10 / 50 / 100 visitors. A later pre-fix native run recorded 52 connection errors at 100 visitors. After the backend connection fix, the full-response run had zero errors, but its P95 was 4,140 / 2,225 / 4,443 ms. The client network also showed intermittent SSH/TLS establishment failures; these WAN runs are not a controlled language comparison.

With conditional ETags, the 100-visitor run returned 613 HTTP 200 and 413 HTTP 304 responses with zero errors, P95 3,312 ms and 2,831,392 decoded bytes. Its load was 1,026 requests, compared with 1,032 requests and 12,726,108 bytes in the full-response run. The 10-visitor conditional run still had a 12.35-second P95 startup outlier. Caching reduces repeated transfer; it does not fix network instability.

An independent GitHub runner also tested the actual HTTPS deployment with fresh anonymous sessions, without model credentials:

| Simultaneous visitors | HTTPS requests | Errors | P95 for a complete four-request session |
| --- | ---: | ---: | ---: |
| 10 | 40 | 0 | 1.93 s |
| 50 | 200 | 0 | 3.67 s |
| 100 | 400 | 0 | 3.82 s |

Each session performs cookie creation, job list, configuration and health requests; histories must be empty, identities distinct, cookies Secure and the server default key unavailable. These session totals are not per-request P95 and should not be compared directly with the polling table.

The host supports the tested 100 visiting sessions and 20 simultaneous light evaluation jobs. It does not guarantee 100 full code benchmarks executing at once without a queue. For sustained heavier use, measure the intended dataset mix and upstream provider limits before increasing scorer or request concurrency. Public bandwidth and network latency still matter.

## Real model results and corrections

Latest [native live CI](https://github.com/Miaoge-Ge/llm-eval-suite/actions/runs/44032803620), using `deepseek-flash`, 10 workers, thinking enabled and a 16,384-token budget:

- API30: **30/30**, 30 attempts, completed in 25.57 seconds.
- Quick40: **39/40**, 40 attempts, completed in 99.80 seconds. `gpqa/gpqa_diamond/30` answered A instead of B; that model error remains.
- Public anonymous HTTPS checks: all three visitor tiers passed.

The earlier server-native run measured API30 30/30 in 27.33 seconds (first meaningful content P50 0.688 s, first body P50 5.695 s, estimated TPS P50 304.3). Quick40 initially scored 38/40 because `#### $145 per cup` was not parsed. Another CI response used `#### 160 minutes (or 2 hours 40 minutes)`. Both clear final-answer formats now have regression coverage. Original reports remain intact; offline regrading produces 39/40 with **zero new API calls**. Arithmetic expressions and ambiguous alternatives remain rejected.

## Regression evidence and limits

Latest [CI](https://github.com/Miaoge-Ge/llm-eval-suite/actions/runs/44032802791) passed. The preceding identical runtime checks reported Ubuntu 338 passed/23 skipped, Windows 339 passed/22 skipped, real Docker integration 12 passed, and native HTTP/Docker tests 10 passed/1 skipped. Native tests require the Linux binary; LiveCodeBench private assets are absent from CI and were tested on the server. The server full regression reported 357 passed/1 platform skip, followed by successful connection-pressure and shared-capacity checks.

Windows Docker Desktop initially timed out on control operations; these failures were preserved. After separating the intentional two-second timeout test from resource/completion tests with a ten-second startup allowance, the isolated Docker suite passed all 11 tests. The full reference Quick40 run and Linux Docker checks also passed. Ruff, formatting, mypy and JavaScript syntax checks passed.

Browser visual inspection was not performed because access to the browser preview was denied. HTTP behavior, TLS, cookies, ownership, real evaluations and downloads were checked independently. This report does not claim visual acceptance or certainty about every possible workload.

Raw local evidence is kept under `results/native-server-validation/`, `results/native-github/`, `results/load-baseline/` and `results/load-native/` (ignored by Git). GitHub live artifacts include prompts, raw generations, timing records, reports and public-session results.
