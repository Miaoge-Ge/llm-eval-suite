"""Raw client-observed stream events shared by evaluation modes."""

from typing import Any


class StreamTimeline:
    def __init__(self, started: float) -> None:
        self.started = started
        self.events: list[dict[str, Any]] = []

    def observe(self, now: float, content: str = "", reasoning: str = "") -> None:
        if content or reasoning:
            self.events.append({"at_seconds": max(0.0, now - self.started), "content": content, "reasoning": reasoning})

    def finish(self, now: float) -> dict[str, Any]:
        return {"duration_seconds": max(0.0, now - self.started), "events": self.events}
