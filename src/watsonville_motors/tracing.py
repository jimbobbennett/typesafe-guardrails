"""Arize AX tracing setup.

`arize.otel.register()` builds a tracer provider pointed at Arize, and
`OpenAIAgentsInstrumentor` auto-instruments the Agents SDK so agent runs, LLM calls and
tool calls all become spans without us touching them.

The guardrail calls are *not* covered by any instrumentor -- there is no OpenInference
instrumentor for TypeSafe -- so `guardrail.py` emits those spans by hand via `get_tracer`.
"""

from __future__ import annotations

import os

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.trace import Tracer

#: Deliberately a literal, not an env lookup: a globally-set ARIZE_PROJECT_NAME should not
#: silently redirect these traces into whatever project the shell happens to point at.
PROJECT_NAME = "watsonville-motors-guardrails"

_tracer: Tracer | None = None


def setup_tracing() -> Tracer:
    """Register the Arize exporter and instrument the Agents SDK. Idempotent."""
    global _tracer
    if _tracer is None:
        from arize.otel import register

        provider = register(
            space_id=os.environ["ARIZE_SPACE_ID"],
            api_key=os.environ["ARIZE_API_KEY"],
            project_name=PROJECT_NAME,
            verbose=False,
        )
        OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)
        _tracer = provider.get_tracer("watsonville_motors.guardrail")
    return _tracer


def get_tracer() -> Tracer:
    """The tracer used for hand-rolled guardrail spans."""
    return _tracer or setup_tracing()


def flush() -> None:
    """Force-flush spans, so a decision shows up in Arize while you are still looking."""
    from opentelemetry import trace

    force_flush = getattr(trace.get_tracer_provider(), "force_flush", None)
    if force_flush is not None:
        force_flush(timeout_millis=10_000)
