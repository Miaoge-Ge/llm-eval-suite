from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

from llm_eval.core.artifacts import RunLock, load_manifest, load_snapshot, write_atomic
from llm_eval.core.reporting import RunSummary, render_markdown_report
from llm_eval.core.runner import EvaluationRunner
from llm_eval.core.settings import config_from_snapshot, load_framework_config
from llm_eval.core.status import RunStatus
from llm_eval.tasks import DEFAULT_TASK_NAME, TASK_REGISTRY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-eval", description="LLM benchmark performance and API service quality")
    subparsers = parser.add_subparsers(dest="command")
    for command, task, help_text in (
        ("api-test", "api_experience", "30 streaming requests, configurable concurrency, no retries"),
        ("quick-test", "quick40", "Fixed 40 original-language benchmark cases"),
    ):
        sub = subparsers.add_parser(command, aliases=["api"] if command == "api-test" else [], help=help_text)
        sub.add_argument("--config", default="configs/model.yaml", help="Model config YAML")
        sub.set_defaults(task=task, limit=None, resume=None)
        sub.add_argument("--workers", type=int, default=None)
    for command, help_text in (
        ("run", "Generate and grade"),
        ("generate", "Save model answers without executing grading"),
        ("resume", "Continue a compatible run without repeating finished cases"),
    ):
        sub = subparsers.add_parser(command, aliases=["benchmark"] if command == "run" else [], help=help_text)
        sub.add_argument("--config", default="configs/model.yaml", help="Model config YAML")
        sub.add_argument("--task", default=None, help="Task name (inferred for resume)")
        sub.add_argument("--workers", type=int, default=None)
        sub.add_argument("--limit", type=int, default=None)
        if command == "resume":
            sub.add_argument("--run-dir", type=Path, required=True)
        else:
            sub.add_argument("--resume", type=Path, default=None, help="Existing run to continue")
            sub.add_argument("--list-tasks", action="store_true")
    grade = subparsers.add_parser("grade", help="Regrade saved answers offline; no API key required")
    grade.add_argument("--run-dir", type=Path, required=True)
    grade.add_argument("--workers", type=int, default=None)
    grade.add_argument("--execution-backend", choices=("docker", "local"), default="docker")
    grade.add_argument("--execution-timeout", type=int, default=None)
    grade.add_argument("--execution-image", default=None, help="Override the saved Docker image for regrading")
    grade.add_argument("--output-dir", type=Path, default=None)
    report = subparsers.add_parser("report", help="Rebuild a report offline from final checkpoints")
    report.add_argument("--run-dir", type=Path, required=True)
    return parser


def rebuild_report(root: Path) -> Path:
    root = root.resolve()
    with RunLock(root):
        manifest, _, archive = load_snapshot(root)
        summary_path = root / "summary.json"
        if not summary_path.exists():
            raise ValueError("No final summary exists; resume the interrupted run or grade its saved answers first")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(RunSummary)}
        summary = RunSummary(**{key: value for key, value in payload.items() if key in allowed})
        config = config_from_snapshot(manifest["config"], root)
        destination = root / "report.md"
        write_atomic(destination, render_markdown_report(summary, config, list(archive.checkpoints().values())))
        return destination


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = ["run", *argv]
    args = build_parser().parse_args(argv)
    if getattr(args, "list_tasks", False):
        print("Available tasks:")
        for name in sorted(TASK_REGISTRY):
            print(f"  - {name}")
        return 0

    try:
        if args.command == "report":
            print(f"Report: {rebuild_report(args.run_dir)}")
            return 0
        if args.command == "grade":
            root = args.run_dir.resolve()
            manifest = load_manifest(root)
            config = config_from_snapshot(manifest["config"], root)
            if args.execution_timeout is not None and args.execution_timeout <= 0:
                raise ValueError("--execution-timeout must be positive")
            config = replace(
                config,
                run=replace(
                    config.run,
                    workers=max(1, args.workers) if args.workers is not None else config.run.workers,
                    execution=replace(
                        config.run.execution,
                        backend=args.execution_backend,
                        image=args.execution_image or config.run.execution.image,
                    ),
                    execution_timeout_seconds=args.execution_timeout or config.run.execution_timeout_seconds,
                    output_dir=args.output_dir.resolve() if args.output_dir else config.run.output_dir,
                ),
            )
            runner = EvaluationRunner(config, operation="grade", source_dir=root)
        else:
            resume_dir = args.run_dir if args.command == "resume" else args.resume
            resume_manifest = load_manifest(resume_dir) if resume_dir else None
            config = load_framework_config(
                model_config_path=args.config,
                task=args.task or (resume_manifest["identity"]["task"] if resume_manifest else DEFAULT_TASK_NAME),
                workers=args.workers,
                limit=args.limit
                if args.limit is not None
                else (resume_manifest["config"]["run"]["limit"] if resume_manifest else None),
            )
            operation = (
                resume_manifest["operation"]
                if resume_manifest
                else ("run" if args.command in {"api-test", "api", "quick-test", "benchmark"} else args.command)
            )
            runner = EvaluationRunner(config, operation=operation, resume_dir=resume_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        summary, writer = runner.run()
    except KeyboardInterrupt:
        print("Evaluation interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(_render_summary(summary, writer.paths.report_md))
    if summary.run_status == RunStatus.INTERRUPTED:
        return 130
    return 0 if summary.run_status == RunStatus.COMPLETED else 2


def _render_summary(summary: RunSummary, report_path: Path) -> str:
    if summary.task == "api_experience":
        metrics = summary.metrics.get("overall", {})
        return (
            f"API experience: {summary.run_status}\n"
            f"Attempted: {metrics.get('attempted', 0)}/30; "
            f"successful streams: {metrics.get('successful', 0)}; truncated: {metrics.get('truncated', 0)}\n"
            f"First-attempt success rate: {metrics.get('first_attempt_success_rate')}\n"
            f"Report: {report_path}"
        )
    lines = [
        "=" * 60,
        "LLM Evaluation Summary",
        "=" * 60,
        f"Operation: {summary.operation}",
        f"Task: {summary.task}",
        f"Model: {summary.model}",
        f"Run status: {summary.run_status}",
        f"Stop reason: {summary.error or 'none'}",
        f"Cases: {summary.completed_cases}/{summary.total_cases}",
        f"Cancelled: {summary.cancelled_cases}",
        f"Grading coverage: {summary.coverage_rate:.2%}",
        f"Pass rate: {summary.pass_rate:.2%}",
        f"Wall clock: {summary.wall_clock_human}",
        f"Avg case: {summary.average_case_seconds:.2f}s",
        f"Throughput: {summary.throughput_tokens_per_second:.1f} tokens/s",
        f"Statuses: {summary.status_counts}",
        f"Tokens: {summary.token_usage}",
        f"API attempts this session: {summary.new_api_attempts}",
        f"Tokens this session: {summary.new_token_usage}",
        f"Report: {report_path}",
        "=" * 60,
    ]
    if summary.test_coverage:
        scope = summary.test_coverage
        lines.insert(
            -1,
            f"Test scope: {scope['scope']}; public tests: {scope['public_tests']}; "
            f"private tests: {scope['private_tests']}",
        )
        if scope["public_only_problems"]:
            lines.insert(-1, "Public-only cases present: this is not a full-test benchmark score")
    return "\n".join(lines)
