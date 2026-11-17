"""Canonical per-case status strings shared by tasks, runner, and reporting."""

from __future__ import annotations


class Status:
    PASSED = "PASSED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    API_ERROR = "API_ERROR"
    API_ERROR_FATAL = "API_ERROR_FATAL"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    OUTPUT_LIMIT = "OUTPUT_LIMIT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    CANCELLED = "CANCELLED"
    GENERATED = "GENERATED"
    MISSING_GENERATION = "MISSING_GENERATION"
    STORAGE_ERROR = "STORAGE_ERROR"


# Statuses that count toward the pass-rate denominator; API and internal
# errors are reported separately instead of being graded as failures.
GRADED_STATUSES = (Status.PASSED, Status.FAILED, Status.TIMEOUT, Status.OUTPUT_LIMIT, Status.RESOURCE_LIMIT)
ERROR_STATUSES = (
    Status.API_ERROR,
    Status.API_ERROR_FATAL,
    Status.INTERNAL_ERROR,
    Status.EXECUTION_ERROR,
    Status.MISSING_GENERATION,
    Status.STORAGE_ERROR,
)


class RunStatus:
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    INTERRUPTED = "INTERRUPTED"
