"""Chainlit chat UI.

    uv run chainlit run app.py -w

Deliberately looks like a real dealership chat widget: an ordinary greeting, no suggested
prompts, no hint that anything is being screened. The demo is only convincing if the window
looks like the thing it is imitating, so the attack prompts live in the README instead.

The one piece of demo furniture is the engine picker in the settings panel.
"""

from __future__ import annotations

from typing import Any

import chainlit as cl
from chainlit.input_widget import Select
from dotenv import load_dotenv

from watsonville_motors.agent import build_agent, build_context
from watsonville_motors.guardrail import MODE_LABELS, Decision, GuardrailMode, Verdict
from watsonville_motors.policy import DEALERSHIP
from watsonville_motors.tracing import flush, setup_tracing

load_dotenv()

GLYPH = {Decision.ALLOW: "🟢", Decision.REVIEW: "🟡", Decision.BLOCK: "🔴"}
BOUNDARY = {
    "inbound_message": "Inbound message",
    "draft_reply": "Draft reply",
    "quote_arguments": "Quote arguments",
}
NOTICE = {
    GuardrailMode.NONE: "⚠️ **No guardrail.** Nothing stands between the model and the customer.",
    GuardrailMode.JEV: "🛡️ **TypeSafe System One.** Every boundary screened.",
    GuardrailMode.LLM: "🐢 **LLM-as-judge.** The same questions, asked of a small fast LLM.",
}


async def _set_mode(mode: GuardrailMode) -> None:
    """Swap the guardrail engine, keeping the conversation.

    The conversation surviving the swap is the point: you can jailbreak the agent with no
    guardrail, change engine, and send the same message again.
    """
    previous = cl.user_session.get("context")
    if previous is not None and previous.guardrail is not None:
        await previous.guardrail.aclose()

    context = build_context(mode)
    context.history = previous.history if previous else []
    cl.user_session.set("mode", mode)
    cl.user_session.set("context", context)
    cl.user_session.set("agent", build_agent(mode))


def _render(verdict: Verdict) -> str:
    lines = [f"**{verdict.decision.value.upper()}** — {verdict.reason}", ""]
    for name, signal in verdict.signals.items():
        if not isinstance(signal, dict):
            lines.append(f"- `{name}`: {signal}")
            continue
        headline = signal.get("choice", signal.get("score", signal.get("probability")))
        confidence = signal.get("confidence")
        suffix = f" · confidence {confidence:.2f}" if isinstance(confidence, float) else ""
        lines.append(f"- **{name}**: {headline}{suffix}")
        lines += [f"    - {label}: {p}" for label, p in (signal.get("probabilities") or {}).items()]
    cost = f" · `${verdict.cost_usd:.6f}`" if verdict.cost_usd is not None else ""
    footer = (
        f"`{verdict.model}` · `{verdict.latency_ms:.0f}ms` · "
        f"`{verdict.input_tokens} in / {verdict.output_tokens} out`{cost}"
    )
    lines += ["", footer]
    return "\n".join(lines)


@cl.on_chat_start
async def on_chat_start() -> None:
    setup_tracing()
    await cl.ChatSettings(
        [
            Select(
                id="guardrail",
                label="Guardrail engine",
                # `items` maps a readable label to the mode value; `values` would show the
                # bare enum names, which mean nothing to someone watching the demo.
                items={MODE_LABELS[m]: m.value for m in GuardrailMode},
                initial_value=GuardrailMode.JEV.value,
                description=(
                    "Which engine screens every inbound message, draft reply and quote "
                    "tool call. Switch to 'no guardrail' to reproduce the $1 Tahoe, or to "
                    "LLM-as-judge to compare speed and cost."
                ),
            )
        ]
    ).send()

    await _set_mode(GuardrailMode.JEV)
    await cl.Message(
        content=(
            f"Thanks for visiting {DEALERSHIP}! I'm here to help with our new and used "
            f"inventory, features, availability and pricing. What can I help you find today?"
        ),
        author="Sales assistant",
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    mode = GuardrailMode(settings.get("guardrail", GuardrailMode.JEV.value))
    if mode is cl.user_session.get("mode"):
        return
    await _set_mode(mode)
    await cl.Message(content=NOTICE[mode], author="Demo").send()


@cl.on_chat_end
async def on_chat_end() -> None:
    context = cl.user_session.get("context")
    if context is not None and context.guardrail is not None:
        await context.guardrail.aclose()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    from agents import (
        InputGuardrailTripwireTriggered,
        OutputGuardrailTripwireTriggered,
        Runner,
        trace,
    )

    agent = cl.user_session.get("agent")
    context = cl.user_session.get("context")
    mode = cl.user_session.get("mode") or GuardrailMode.NONE
    conversation: list[dict] = cl.user_session.get("conversation") or []

    seen = len(context.verdicts)
    reply: str | None = None
    blocked: str | None = None

    with trace(f"chat turn ({mode.value})"):
        try:
            result = await Runner.run(
                agent, input=conversation + [{"role": "user", "content": message.content}], context=context
            )
            cl.user_session.set("conversation", result.to_input_list())
            reply = str(result.final_output)
        except InputGuardrailTripwireTriggered:
            blocked = "The message never reached the agent."
        except OutputGuardrailTripwireTriggered:
            blocked = "The agent replied, but the customer never saw it."

    new = context.verdicts[seen:]
    for verdict in new:
        async with cl.Step(
            name=f"{GLYPH[verdict.decision]} {BOUNDARY.get(verdict.check, verdict.check)}"
            f" — {verdict.decision.value}",
            type="tool",
            default_open=verdict.decision is not Decision.ALLOW,
        ) as step:
            step.output = _render(verdict)

    if blocked:
        cause = next((v for v in reversed(new) if v.blocked), None)
        await cl.Message(
            content=f"🔴 **Blocked.** {blocked}\n\n> {cause.reason if cause else ''}",
            author="Guardrail",
        ).send()
    elif reply is not None:
        await cl.Message(content=reply, author="Sales assistant").send()

    context.history.append(f"customer: {message.content}")
    if reply:
        context.history.append(f"assistant: {reply}")

    flush()
