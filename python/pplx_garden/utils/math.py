import math
from dataclasses import dataclass


def round_up(n: int, m: int) -> int:
    return (n + m - 1) // m * m


def ceil_div(n: int, m: int) -> int:
    return (n + m - 1) // m


def floor_div(n: int, m: int) -> int:
    return n // m


def stddev(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0

    n = len(xs)
    m = 0.0
    s = 0.0
    for k, x in enumerate(xs):
        old_m = m
        m = m + (x - m) / (k + 1)
        s = s + (x - m) * (x - old_m)
    variance = s / (n - 1)
    return math.sqrt(variance)


def mean(xs: list[float]) -> float:
    if len(xs) == 0:
        return 0.0
    return sum(xs) / len(xs)


def mean_and_stddev(xs: list[float]) -> tuple[float, float]:
    return mean(xs), stddev(xs)


@dataclass
class Statistics:
    mean: float
    stddev: float
    min: float
    p01: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float
    max: float

    @classmethod
    def create(cls, xs: list[float]) -> "Statistics":
        if len(xs) == 0:
            return cls(
                mean=0.0,
                stddev=0.0,
                min=0.0,
                p01=0.0,
                p25=0.0,
                p50=0.0,
                p75=0.0,
                p95=0.0,
                p99=0.0,
                max=0.0,
            )

        n = len(xs)
        sorted_xs = sorted(xs)
        mean, stddev = mean_and_stddev(xs)

        def percentile(p: float) -> float:
            index = int(n * p)
            if n * p == index or index + 1 >= n:
                return sorted_xs[index]
            return (sorted_xs[index] + sorted_xs[index + 1]) / 2

        return cls(
            mean=mean,
            stddev=stddev,
            min=sorted_xs[0],
            p01=percentile(0.01),
            p25=percentile(0.25),
            p50=percentile(0.5),
            p75=percentile(0.75),
            p95=percentile(0.95),
            p99=percentile(0.99),
            max=sorted_xs[-1],
        )

    def __str__(self) -> str:
        fields = [
            ("min", self.min),
            ("p01", self.p01),
            ("p25", self.p25),
            ("p50", self.p50),
            ("p75", self.p75),
            ("p95", self.p95),
            ("p99", self.p99),
            ("max", self.max),
        ]

        desc = f"mean={self.mean:.1f}±{self.stddev:.1f} μs"
        for name, value in fields:
            desc += f", {name}={value:.1f} μs"
        return desc
