# app/agent.py
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from app.schemas import AgentAnswer, FinalAnswer, ToolCall
from app.tools import ALL_TOOLS

MODEL_ID = "claude-sonnet-5"
MAX_TOOL_CALLS = 10
RECURSION_LIMIT = 2 * MAX_TOOL_CALLS + 1
_OUTPUT_CAP = 600  # chars kept per tool output in the trail

SYSTEM_PROMPT = """You are Bluno's trade-finance compliance assistant. Answer the \
user's question about counterparties and shipments by calling the available tools \
and reasoning over what they return.

Rules:
- Decide each next action based on what you have learned so far. Call only the \
tools you need; do not call every tool.
- You have a hard budget of 10 tool calls. A well-formed answer needs far fewer.
- Ground every claim in tool output. Never invent company names, countries, \
numbers, or statuses.
- If a tool returns status NOT_FOUND for something the question depends on (e.g. a \
company that is not in Bluno's records), the question cannot be answered from the \
data — say so plainly rather than guessing.
- "Cleared to ship to a destination country" is about that country's sanctions \
clearance: CLEARED means permitted, RESTRICTED means permitted only with extra \
documentation and manual sign-off, PROHIBITED means not permitted. KYC status is a \
separate matter — mention it only when the question asks about it.
- Available credit headroom = credit limit minus outstanding exposure; 0 means \
fully utilised.
- When comparing or ranking several companies, use list_companies to get the \
candidate set, then look up each as needed.
When you have enough information, stop and give a direct, complete answer."""

SYNTH_SYSTEM = """You produce the final structured answer for a trade-finance \
question. You are given the question, the trail of tool calls and their outputs, \
and the assistant's draft answer. Return:
- answer: a direct natural-language answer grounded ONLY in the tool outputs shown.
- resolved: true if the tool outputs contain what is needed to answer; false if a \
required entity was NOT_FOUND or the data does not contain the answer.
- reasoning_summary: 1-3 sentences on how the answer was reached.
Do not invent facts beyond the tool outputs."""


def build_trail(messages) -> list[ToolCall]:
    outputs: dict[str, str] = {
        m.tool_call_id: str(m.content) for m in messages if isinstance(m, ToolMessage)
    }
    trail: list[ToolCall] = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                tid = tc.get("id", "")
                if tid not in outputs:
                    # Proposed but not yet executed — don't count it.
                    continue
                out = outputs[tid]
                if len(out) > _OUTPUT_CAP:
                    out = out[:_OUTPUT_CAP] + "…"
                trail.append(
                    ToolCall(
                        tool_name=tc["name"],
                        tool_input=dict(tc.get("args", {})),
                        tool_output=out,
                    )
                )
    return trail


def get_final_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content.strip()
        # Some providers return content as a list of blocks; join text blocks.
        if isinstance(m, AIMessage) and isinstance(m.content, list):
            text = " ".join(
                b.get("text", "") for b in m.content if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if text:
                return text
    return ""


def _render_trail(trail: list[ToolCall]) -> str:
    if not trail:
        return "(no tools were called)"
    lines = []
    for i, tc in enumerate(trail, 1):
        lines.append(f"{i}. {tc.tool_name}({tc.tool_input}) -> {tc.tool_output}")
    return "\n".join(lines)


class AgentRunner:
    def __init__(self, model, agent):
        self._model = model
        self._agent = agent

    def answer(self, question_id: str, question: str) -> AgentAnswer:
        last_state = None
        budget_exceeded = False
        try:
            for state in self._agent.stream(
                {"messages": [HumanMessage(content=question)]},
                config={"recursion_limit": RECURSION_LIMIT},
                stream_mode="values",
            ):
                last_state = state
                # Hard stop: once the trail reaches the budget, do not consume
                # any further tool-calling rounds.
                if len(build_trail(state["messages"])) >= MAX_TOOL_CALLS:
                    budget_exceeded = True
                    break
        except GraphRecursionError:
            budget_exceeded = True

        messages = last_state["messages"] if last_state else []
        trail = build_trail(messages)[:MAX_TOOL_CALLS]
        final_text = get_final_text(messages)

        final = self._synthesize(question, trail, final_text)
        return AgentAnswer(
            question_id=question_id,
            answer=final.answer,
            resolved=final.resolved,
            tool_calls=trail,
            tool_call_count=len(trail),
            budget_exceeded=budget_exceeded,
            reasoning_summary=final.reasoning_summary,
        )

    def _synthesize(self, question: str, trail: list[ToolCall], final_text: str) -> FinalAnswer:
        structured = self._model.with_structured_output(FinalAnswer)
        human = (
            f"Question: {question}\n\n"
            f"Tool call trail:\n{_render_trail(trail)}\n\n"
            f"Assistant draft answer:\n{final_text or '(none)'}\n\n"
            "Produce the final structured answer."
        )
        messages = [SystemMessage(content=SYNTH_SYSTEM), HumanMessage(content=human)]
        for _ in range(2):
            try:
                return structured.invoke(messages)
            except Exception:
                continue
        # Structured synthesis failed twice (e.g. the model omitted a required
        # field) — degrade to the draft answer rather than raising a 500.
        return FinalAnswer(
            answer=final_text or "The agent could not produce a final answer for this question.",
            resolved=bool(final_text),
            reasoning_summary="Structured synthesis failed after retries; falling back to the assistant's draft answer.",
        )


def build_agent() -> AgentRunner:
    model = ChatAnthropic(model=MODEL_ID, max_tokens=4096, timeout=60)
    agent = create_react_agent(model, ALL_TOOLS, prompt=SYSTEM_PROMPT)
    return AgentRunner(model=model, agent=agent)
