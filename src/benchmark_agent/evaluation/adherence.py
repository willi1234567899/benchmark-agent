from __future__ import annotations

from collections import Counter

from benchmark_agent.models import AdherenceResult, ProxyWindowStats

API_INFLUX = "influx"


def evaluate_adherence(proxy_stats: ProxyWindowStats | None, *, api: str) -> AdherenceResult | None:
    if proxy_stats is None:
        return None
    samples = proxy_stats.requests
    if api == API_INFLUX:
        return AdherenceResult(
            api=api, requests=len(samples), accepted=len(samples), rejected=0, legacy=0, ok=True
        )
    accepted = [
        sample
        for sample in samples
        if sample.api == api and sample.rejection is None and sample.operation
    ]
    rejected = [sample for sample in samples if sample.api == api and sample.rejection is not None]
    legacy = [sample for sample in samples if sample.api == "legacy"]
    unknown = [sample for sample in samples if sample.api == "unknown"]
    reasons = Counter(str(sample.rejection) for sample in [*rejected, *legacy, *unknown])
    return AdherenceResult(
        api=api,
        requests=len(samples),
        accepted=len(accepted),
        rejected=len(rejected) + len(unknown),
        legacy=len(legacy),
        rejections=[f"{code} x{count}" for code, count in sorted(reasons.items())],
        ok=bool(accepted) and not rejected and not legacy and not unknown,
    )
