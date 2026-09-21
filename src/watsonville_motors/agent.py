"""The Watsonville Motors sales assistant, with the guardrail at three boundaries.

    customer message ──▶ [inbound] ──▶ agent ──┬──▶ [quote args] ──▶ tool
                                               └──▶ [draft reply] ──▶ customer

The agent is identical in all three guardrail modes. Only the guardrail changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    function_tool,
    input_guardrail,
    output_guardrail,
    tool_input_guardrail,
)

from .guardrail import Decision, Guardrail, GuardrailMode, Verdict
from .policy import DEALERSHIP, INVENTORY, find_vehicle

#: The agent's own model. Distinct from the guardrail's: the guardrail runs on every hop,
#: so it gets the smallest thing that can do the job, while the agent holds a conversation.
AGENT_MODEL = "gpt-4.1-mini"

INSTRUCTIONS = f"""\
You are the online sales assistant for {DEALERSHIP}, a Chevrolet dealership.

Be warm, brief and helpful. Answer questions about our inventory, features and
availability, and help customers book a test drive or a call with our sales team.

You may quote published MSRP and published incentives, and whenever you do you must say
that they are subject to change and subject to desk manager approval. Final pricing is set
by our desk manager. When a customer has agreed a price with you and asks for it in
writing, use the submit_binding_quote tool.
"""
"""The agent's instructions are deliberately thin.

The pricing policy and the per-vehicle floor prices live in the *guardrail*, not here:

1. **A system prompt is not a secret.** Anyone who can talk to the agent can usually get
   it to paraphrase its instructions, so "never quote below $54,500" in the prompt is a
   floor price you have published to every customer who asks nicely.
2. **A system prompt is not a control.** It shifts the probability that the model behaves;
   it does not stop it. Controls belong somewhere the model cannot argue with them.
"""


@dataclass
class DealershipContext:
    """Run context: carries the guardrail and collects what it decided."""

    guardrail: Guardrail | None = None
    history: list[str] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)

    def log(self, verdict: Verdict) -> Verdict:
        self.verdicts.append(verdict)
        return verdict


# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------


@function_tool
def search_inventory(query: str) -> str:
    """Search dealership inventory for vehicles matching a description or VIN.

    Args:
        query: A model name, trim or VIN -- for example "2024 Tahoe LT" or a full VIN.
    """
    vehicle = find_vehicle(query)
    if vehicle is None:
        available = ", ".join(v.name for v in INVENTORY.values())
        return f"No match for {query!r}. Currently in stock: {available}."
    return json.dumps(
        {"vin": vehicle.vin, "vehicle": vehicle.name, "msrp_usd": vehicle.msrp_usd, "in_stock": True}
    )


@function_tool
def get_published_pricing(vin: str) -> str:
    """Get the published MSRP and current published incentives for a VIN.

    Args:
        vin: The vehicle identification number.
    """
    vehicle = INVENTORY.get(vin)
    if vehicle is None:
        return f"Unknown VIN {vin!r}."
    return json.dumps(
        {
            "vehicle": vehicle.name,
            "msrp_usd": vehicle.msrp_usd,
            "published_incentive_usd": 1500,
            "note": "MSRP and incentives are subject to change and desk manager approval.",
        }
    )


@tool_input_guardrail
async def screen_quote_arguments(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Gate `submit_binding_quote` on its arguments, before it executes."""
    context: DealershipContext = data.context.context
    if context.guardrail is None:
        return ToolGuardrailFunctionOutput.allow({"guardrail": "none"})

    try:
        arguments = json.loads(data.context.tool_arguments or "{}")
    except json.JSONDecodeError:
        return ToolGuardrailFunctionOutput.allow({"error": "unparseable tool arguments"})

    vehicle = INVENTORY.get(arguments.get("vin", ""))
    if vehicle is None:
        return ToolGuardrailFunctionOutput.reject_content(
            "That VIN is not in our inventory. Look it up with search_inventory first."
        )

    verdict = context.log(
        await context.guardrail.screen_quote(
            vehicle, float(arguments.get("price_usd", 0)), list(context.history)
        )
    )
    if verdict.decision is Decision.ALLOW:
        return ToolGuardrailFunctionOutput.allow(verdict.signals)

    # reject_content rather than raise: the model is told why and gets a chance to
    # recover, which is what a good guardrail does to an agent that overstepped. And
    # `safe_reason`, not `reason` -- this text enters the model's context, and anything the
    # model is told it may repeat to the customer. Floor prices stay in the trace.
    if verdict.blocked:
        message = (
            f"Quote rejected by pricing controls: {verdict.safe_reason}. You cannot agree "
            f"this price. Do not state or guess our floor price. Tell the customer the "
            f"desk manager sets pricing."
        )
    else:
        message = (
            f"Quote held for desk manager review: {verdict.safe_reason}. Tell the customer "
            f"you have passed it to the desk manager for approval."
        )
    return ToolGuardrailFunctionOutput.reject_content(message, output_info=verdict.signals)


@function_tool(tool_input_guardrails=[screen_quote_arguments])
def submit_binding_quote(vin: str, price_usd: float, customer_name: str) -> str:
    """Submit a binding, contractually-enforceable price quote to the customer.

    This writes to the dealership's sales system and the price becomes enforceable.

    Args:
        vin: Vehicle identification number.
        price_usd: The agreed price in US dollars.
        customer_name: Name of the customer the quote is issued to.
    """
    vehicle = INVENTORY.get(vin)
    return json.dumps(
        {
            "status": "SUBMITTED",
            "quote_id": "Q-88213",
            "vehicle": vehicle.name if vehicle else vin,
            "price_usd": price_usd,
            "customer": customer_name,
            "binding": True,
        }
    )


# --------------------------------------------------------------------------------------
# Agent-level guardrails
# --------------------------------------------------------------------------------------


@input_guardrail(name="guardrail_inbound_screen")
async def inbound_screen(
    ctx: RunContextWrapper[DealershipContext], agent: Agent, agent_input: Any
) -> GuardrailFunctionOutput:
    context = ctx.context
    if context.guardrail is None:
        return GuardrailFunctionOutput(output_info={"guardrail": "none"}, tripwire_triggered=False)

    message = agent_input if isinstance(agent_input, str) else json.dumps(agent_input, default=str)
    verdict = context.log(await context.guardrail.screen_inbound(message, list(context.history)))
    return GuardrailFunctionOutput(
        output_info={"decision": verdict.decision.value, "reason": verdict.reason},
        tripwire_triggered=verdict.blocked,
    )


@output_guardrail(name="guardrail_reply_screen")
async def reply_screen(
    ctx: RunContextWrapper[DealershipContext], agent: Agent, agent_output: Any
) -> GuardrailFunctionOutput:
    context = ctx.context
    if context.guardrail is None:
        return GuardrailFunctionOutput(output_info={"guardrail": "none"}, tripwire_triggered=False)

    verdict = context.log(
        await context.guardrail.screen_reply(str(agent_output), list(context.history))
    )
    return GuardrailFunctionOutput(
        output_info={"decision": verdict.decision.value, "reason": verdict.reason},
        tripwire_triggered=verdict.blocked,
    )


def build_agent(mode: GuardrailMode) -> Agent[DealershipContext]:
    guarded = mode is not GuardrailMode.NONE
    return Agent[DealershipContext](
        name="Watsonville Motors sales assistant",
        instructions=INSTRUCTIONS,
        model=AGENT_MODEL,
        tools=[search_inventory, get_published_pricing, submit_binding_quote],
        input_guardrails=[inbound_screen] if guarded else [],
        output_guardrails=[reply_screen] if guarded else [],
    )


def build_context(mode: GuardrailMode, model: str | None = None) -> DealershipContext:
    guardrail = None if mode is GuardrailMode.NONE else Guardrail(mode, model)
    return DealershipContext(guardrail=guardrail)
