"""Two guardrail engines behind one interface.

The point of the demo is a like-for-like comparison, so the *questions* and the *decision
logic* must be identical no matter which engine answers them. That means neither engine
can be allowed to leak its own vocabulary into the rest of the app. So:

* Checks are declared once as provider-neutral specs (`Noul`, `Choice`, `Score` below).
* Both engines return the same normalised `Answer` objects.
* `guardrail.py` does exactly the same arithmetic on those answers either way.

`JevEngine` calls TypeSafe System One, which is built to return calibrated probability
distributions over a fixed set of options. `LlmEngine` asks an ordinary LLM -- the same
provider that powers the agent -- for the same fields via structured outputs. The LLM will
happily emit numbers in the right shape; whether they are *calibrated* is the interesting
question, and the whole reason to run both.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel

Kind = Literal["noul", "choice", "score"]


# --------------------------------------------------------------------------------------
# Provider-neutral check specs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Noul:
    """A yes/no proposition. The answer is the probability that it is true."""

    instructions: str


@dataclass(frozen=True)
class Choice:
    """Pick one of a fixed set of options. `criteria` maps option -> description."""

    instructions: str
    criteria: dict[str, str | None]


@dataclass(frozen=True)
class Score:
    """Rate against ordered levels, low to high. The answer can fall between levels."""

    instructions: str
    criteria: list[str]


Spec = Noul | Choice | Score


# --------------------------------------------------------------------------------------
# Normalised answers
# --------------------------------------------------------------------------------------


@dataclass
class Answer:
    kind: Kind
    #: Noul only: probability the proposition is true.
    noul: float | None = None
    #: Choice only: the selected option.
    choice: str | None = None
    #: Score only: position along the levels, may fall between two of them.
    score: float | None = None
    #: Choice and Score only. Noul answers carry no confidence.
    confidence: float | None = None
    #: label -> probability. For a Score the labels are the level descriptions.
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class EngineResult:
    answers: dict[str, Answer]
    latency_ms: float
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None


class Engine(Protocol):
    provider: str
    model: str

    async def ask(self, state: dict[str, Any], checks: dict[str, Spec]) -> EngineResult: ...

    async def aclose(self) -> None: ...


# --------------------------------------------------------------------------------------
# TypeSafe System One
# --------------------------------------------------------------------------------------


class JevEngine:
    """TypeSafe System One. Purpose-built for typed, calibrated decisions."""

    provider = "typesafe"

    def __init__(self, model: str | None = None) -> None:
        from typesafe_sdk import AsyncTypeSafeClient

        self.model = model or os.environ.get("TYPESAFE_MODEL", "jev-latest")
        self._client = AsyncTypeSafeClient()

    async def ask(self, state: dict[str, Any], checks: dict[str, Spec]) -> EngineResult:
        from typesafe_sdk import Choice as TsChoice
        from typesafe_sdk import Noul as TsNoul
        from typesafe_sdk import Score as TsScore

        questions: dict[str, Any] = {}
        for name, spec in checks.items():
            if isinstance(spec, Noul):
                questions[name] = TsNoul(instructions=spec.instructions)
            elif isinstance(spec, Choice):
                questions[name] = TsChoice(instructions=spec.instructions, criteria=spec.criteria)
            else:
                questions[name] = TsScore(instructions=spec.instructions, criteria=spec.criteria)

        started = time.perf_counter()
        response = await self._client.system_one(state=state, questions=questions, model=self.model)
        latency_ms = (time.perf_counter() - started) * 1000

        answers: dict[str, Answer] = {}
        for name, raw in response.answers.items():
            if raw.type == "noul":
                answers[name] = Answer(kind="noul", noul=raw.noul)
            elif raw.type == "choice":
                answers[name] = Answer(
                    kind="choice",
                    choice=raw.choice,
                    confidence=raw.confidence,
                    probabilities=dict(raw.probabilities),
                )
            else:
                # Score probabilities are keyed by level index; `legend` names the levels.
                answers[name] = Answer(
                    kind="score",
                    score=raw.score,
                    confidence=raw.confidence,
                    probabilities={
                        str(raw.legend[i]): p for i, p in raw.probabilities.items()
                    },
                )

        return EngineResult(
            answers=answers,
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            request_id=response.request_id,
            model=response.model,
            provider=self.provider,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------------------
# An ordinary LLM, same provider as the agent
# --------------------------------------------------------------------------------------


#: The smallest, fastest model from the agent's own provider that answers these questions
#: correctly. Measured 2026-09-21: gpt-5.4-nano is faster, cheaper AND more accurate here
#: than gpt-4.1-nano, which scored `claims_binding` at 0.1 on a reply that says verbatim
#: "that's a legally binding offer". Picking the weaker model would be a strawman.
DEFAULT_LLM_MODEL = "gpt-5.4-nano"


class _Probability(BaseModel):
    label: str
    probability: float


class _LlmAnswer(BaseModel):
    name: str
    kind: Kind
    noul: float | None
    choice: str | None
    score: float | None
    confidence: float | None
    probabilities: list[_Probability]


class _LlmAnswers(BaseModel):
    answers: list[_LlmAnswer]


_SYSTEM_PROMPT = """\
You are a guardrail classifier. You are given a JSON state and a set of questions about \
it. Answer every question and return nothing else.

Question types:
- noul: a yes/no proposition. Set `noul` to the probability from 0 to 1 that it is true. \
Leave `choice`, `score`, `confidence` null and `probabilities` empty.
- choice: pick exactly one of the listed options. Set `choice` to that option's key, and \
`probabilities` to your probability for every option (they must sum to 1). Set \
`confidence` from 0 to 1: high when one option clearly wins, low when the distribution is \
flat. Leave `noul` and `score` null.
- score: the levels are ordered low to high and numbered from 0. Set `score` to the \
position along them, which may fall between two levels. Set `probabilities` to your \
probability for each level (they must sum to 1), labelling each one with that level's \
exact text and nothing else -- no number, no prefix. Set `confidence` as for choice. \
Leave `noul` and `choice` null.

Be calibrated. If the state does not give you enough to go on, say so with a flat \
distribution and low confidence rather than picking arbitrarily.
"""


def _render_questions(checks: dict[str, Spec]) -> str:
    blocks: list[str] = []
    for name, spec in checks.items():
        if isinstance(spec, Noul):
            blocks.append(f'- name: "{name}"\n  type: noul\n  question: {spec.instructions}')
        elif isinstance(spec, Choice):
            options = "\n".join(
                f"    - {key}" + (f": {description}" if description else "")
                for key, description in spec.criteria.items()
            )
            blocks.append(
                f'- name: "{name}"\n  type: choice\n  question: {spec.instructions}\n'
                f"  options:\n{options}"
            )
        else:
            levels = "\n".join(f"    {i}. {text}" for i, text in enumerate(spec.criteria))
            blocks.append(
                f'- name: "{name}"\n  type: score\n  question: {spec.instructions}\n'
                f"  levels:\n{levels}"
            )
    return "\n".join(blocks)


class LlmEngine:
    """The same questions, asked of an ordinary LLM via structured outputs.

    Uses the smallest and fastest model the agent's provider offers, because that is the
    fairest comparison: if you were going to put an LLM on this hot path you would reach
    for the cheapest one that can do the job.
    """

    provider = "openai"

    def __init__(self, model: str | None = None) -> None:
        from openai import AsyncOpenAI

        self.model = model or os.environ.get("GUARDRAIL_LLM_MODEL", DEFAULT_LLM_MODEL)
        self._client = AsyncOpenAI()
        #: Some newer model families reject `temperature` outright. We want temperature=0
        #: where it is allowed, so try it and remember if the model refuses.
        self._send_temperature = True

    async def ask(self, state: dict[str, Any], checks: dict[str, Spec]) -> EngineResult:
        user = (
            f"STATE:\n{json.dumps(state, indent=2, default=str)}\n\n"
            f"QUESTIONS:\n{_render_questions(checks)}"
        )

        started = time.perf_counter()
        response = await self._parse(user)
        latency_ms = (time.perf_counter() - started) * 1000

        parsed = response.output_parsed
        answers: dict[str, Answer] = {}
        for raw in parsed.answers if parsed else []:
            answers[raw.name] = Answer(
                kind=raw.kind,
                noul=raw.noul,
                choice=raw.choice,
                score=raw.score,
                confidence=raw.confidence,
                probabilities={p.label: p.probability for p in raw.probabilities},
            )

        return EngineResult(
            answers=answers,
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens if response.usage else None,
            output_tokens=response.usage.output_tokens if response.usage else None,
            request_id=response.id,
            model=self.model,
            provider=self.provider,
        )

    async def _parse(self, user: str) -> Any:
        from openai import BadRequestError

        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": _SYSTEM_PROMPT,
            "input": user,
            "text_format": _LlmAnswers,
        }
        if self._send_temperature:
            try:
                return await self._client.responses.parse(**kwargs, temperature=0)
            except BadRequestError as error:
                if "temperature" not in str(error):
                    raise
                self._send_temperature = False
        return await self._client.responses.parse(**kwargs)

    async def aclose(self) -> None:
        await self._client.close()


# --------------------------------------------------------------------------------------
# Pricing
#
# Published list prices per million tokens at the time of writing. These drive the cost
# column in the CLI summary, so VERIFY THEM against each provider's pricing page before
# putting a number from this demo on a slide.
# --------------------------------------------------------------------------------------

RATES_PER_MTOK: dict[str, tuple[float, float]] = {
    # model prefix: (input $/Mtok, output $/Mtok)
    "jev": (0.042, 0.0),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5.4-nano": (0.05, 0.40),
}


def cost_usd(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """Cost of one call, or None if we have no published rate for the model."""
    for prefix, (rate_in, rate_out) in RATES_PER_MTOK.items():
        if model.startswith(prefix):
            return ((input_tokens or 0) * rate_in + (output_tokens or 0) * rate_out) / 1_000_000
    return None
