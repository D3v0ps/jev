"""Running totals for a series of requests: tokens, dollars, and latency.

Every claim these patterns make about speed or cost should come from a ledger a
caller can print, not from a number written into a README.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of `values`. Raises on an empty list."""
    if not values:
        raise ValueError("percentile of no samples")
    if not 0 <= fraction <= 1:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


@dataclass
class Ledger:
    """What a run spent. `unpriced` counts replies from a model with no price on file."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    unpriced: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, reply: object) -> None:
        """Add one Reply. Typed loosely to keep jevkit.answers free of a back-import."""
        self.calls += 1
        self.input_tokens += reply.input_tokens  # type: ignore[attr-defined]
        self.output_tokens += reply.response.usage.output_tokens  # type: ignore[attr-defined]
        self.latencies_ms.append(reply.latency_ms)  # type: ignore[attr-defined]
        spend = reply.usd  # type: ignore[attr-defined]
        if spend is None:
            self.unpriced += 1
        else:
            self.usd += spend

    @property
    def p50_ms(self) -> float:
        return percentile(self.latencies_ms, 0.50)

    @property
    def p95_ms(self) -> float:
        return percentile(self.latencies_ms, 0.95)

    @property
    def mean_ms(self) -> float:
        if not self.latencies_ms:
            raise ValueError("no samples")
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def sequential_rate(self) -> float:
        """Decisions per second one caller sustains at the median latency, without concurrency."""
        return 1000.0 / self.p50_ms if self.p50_ms else float("inf")

    def summary(self) -> str:
        """One line for a demo or a log."""
        if not self.calls:
            return "no requests"
        money = f"${self.usd:.6f}" if not self.unpriced else f"${self.usd:.6f} (+{self.unpriced} unpriced)"
        return (
            f"{self.calls} requests · {self.input_tokens} input tokens · {money} · "
            f"p50 {self.p50_ms:.0f} ms · p95 {self.p95_ms:.0f} ms · {self.sequential_rate:.1f}/s sequential"
        )
