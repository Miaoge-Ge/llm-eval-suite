# DeepSeek Flash validation — 2026-11-17

Source commit: `8ca3aae`. [Regression CI](https://github.com/Miaoge-Ge/llm-eval-suite/actions/runs/44009861565) and [real API evaluation](https://github.com/Miaoge-Ge/llm-eval-suite/actions/runs/44009881928) completed successfully.

## Settings

OpenAI-compatible endpoint `https://api.deepseek.com/v1`, requested model `deepseek-flash`, thinking enabled, output budget 16,384 tokens including reasoning, 10 workers. Executed on GitHub-hosted Ubuntu 24.04. These are observations from one run, not a stable ranking or identity proof.

## API service quality

| Metric | Observed |
| --- | --- |
| Planned requests / actual attempts | 30 / 30 |
| First-attempt success | 30 / 30 (100%) |
| Automatic retries | 0 |
| Truncation / stream interruptions | 0 / 0 |
| First meaningful content P50 / P95 | 1.04 s / 1.46 s |
| First body P50 / P95 | 6.47 s / 10.01 s |
| End-to-end P50 / P95 | 6.81 s / 11.50 s |
| Estimated visible output TPS P50 | 292.84 |
| Wall-clock time | 27.73 s |

TPS uses provider completion tokens minus reported reasoning tokens and the observed body streaming window. It is an estimate, not a server-side token emission trace. Each of short answer, long answer and long context had 10 successful requests. SLO thresholds were not configured.

## Quick40

| Dimension | Passed / planned |
| --- | --- |
| Code | 10 / 10 |
| Math | 10 / 10 |
| Knowledge | 10 / 10 |
| Instruction following | 9 / 10 |
| Total | 39 / 40 (97.5%) |

All 40 cases were graded, using 40 attempts, without API errors or output truncation. Code cases ran their complete Plus tests in real Docker. Wall-clock time was 74.14 s.

The failed case was `ifeval/1082`: the prompt explicitly required two paragraphs separated by `***`. The model returned two paragraphs without that divider. The original IFEval checker correctly rejected it; no grading relaxation was applied.

## Corrections confirmed

The [first real run](https://github.com/Miaoge-Ge/llm-eval-suite/actions/runs/44003815590) remains preserved. Its smaller budgets caused reasoning-only truncation. Three GSM answers also used valid units/currency after the explicit final-answer marker and were incorrectly rejected by the old extractor.

Offline regrading the original answers with the corrected extractor made no API calls: 37 passed, one instruction failure, and two missing body generations. That regrade remains incomplete; it does not fabricate answers for the two missing cases. Increasing the live output budget produced body answers in the new run, while keeping the original prompts and requested answer lengths.

## Regression evidence

- Local Windows suite with real Docker enabled: **347 passed**.
- GitHub Ubuntu: **334 passed, 13 skipped**; Windows: **335 passed, 12 skipped**. Skips cover platform-specific and opt-in Docker tests.
- Separate GitHub Docker job: **12 passed**, including full Quick40 reference-answer grading, resume and offline regrading.
- Injected 429 and interrupted streams at 10 workers: exactly **30 requests for 30 cases**, with failures retained and original concurrency preserved by offline scoring.
- Web tests cover authenticated reports/downloads, cross-origin mutation rejection, default-key endpoint restriction, real runner integration, queue cancellation and restart recovery.

Raw generations, per-chunk timings and full reports are downloadable from the live workflow artifact (14-day retention). Local copies should be kept when longer retention is required.
