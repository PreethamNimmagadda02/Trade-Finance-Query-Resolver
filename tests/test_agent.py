# tests/test_agent.py
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from app.agent import build_trail, get_final_text, AgentRunner
from app.schemas import FinalAnswer


def _ai_with_tool_calls(calls):
    return AIMessage(content="", tool_calls=calls)


def test_build_trail_orders_and_pairs():
    messages = [
        HumanMessage(content="q"),
        _ai_with_tool_calls([
            {"name": "find_company", "args": {"name": "ACME Exports"}, "id": "t1"},
        ]),
        ToolMessage(content='{"status": "found"}', tool_call_id="t1"),
        _ai_with_tool_calls([
            {"name": "get_credit_headroom", "args": {"name": "ACME Exports"}, "id": "t2"},
        ]),
        ToolMessage(content='{"available_headroom_usd": 320000}', tool_call_id="t2"),
        AIMessage(content="Final answer."),
    ]
    trail = build_trail(messages)
    assert [t.tool_name for t in trail] == ["find_company", "get_credit_headroom"]
    assert trail[0].tool_input == {"name": "ACME Exports"}
    assert "320000" in trail[1].tool_output


def test_get_final_text():
    messages = [AIMessage(content="hello"), AIMessage(content="the answer")]
    assert get_final_text(messages) == "the answer"


class _FakeAgent:
    """Yields streamed states; the final state's messages drive the trail."""

    def __init__(self, states):
        self._states = states

    def stream(self, _inputs, config=None, stream_mode=None):
        yield from self._states


class _FakeStructured:
    def __init__(self, result):
        self._result = result

    def invoke(self, _messages):
        return self._result


class _FakeModel:
    def __init__(self, final):
        self._final = final

    def with_structured_output(self, _schema):
        return _FakeStructured(self._final)


def test_runner_assembles_answer_within_budget():
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "find_company", "args": {"name": "ACME Exports"}, "id": "t1"}]),
        ToolMessage(content='{"status": "found"}', tool_call_id="t1"),
        AIMessage(content="ACME is cleared."),
    ]
    agent = _FakeAgent([{"messages": messages}])
    model = _FakeModel(FinalAnswer(answer="Yes, cleared.", resolved=True, reasoning_summary="Germany is CLEARED."))
    runner = AgentRunner(model=model, agent=agent)
    ans = runner.answer("Q-01", "Is ACME cleared?")
    assert ans.question_id == "Q-01"
    assert ans.answer == "Yes, cleared."
    assert ans.resolved is True
    assert ans.tool_call_count == 1
    assert ans.budget_exceeded is False
    assert ans.tool_calls[0].tool_name == "find_company"


def test_runner_salvages_on_recursion_error():
    from langgraph.errors import GraphRecursionError

    good_state = {
        "messages": [
            HumanMessage(content="q"),
            AIMessage(content="", tool_calls=[{"name": "find_company", "args": {"name": "X"}, "id": "t1"}]),
            ToolMessage(content='{"status": "NOT_FOUND"}', tool_call_id="t1"),
        ]
    }

    class _RaisingAgent:
        def stream(self, _inputs, config=None, stream_mode=None):
            yield good_state
            raise GraphRecursionError("limit")

    model = _FakeModel(FinalAnswer(answer="Cannot resolve.", resolved=False, reasoning_summary="Not in data."))
    runner = AgentRunner(model=model, agent=_RaisingAgent())
    ans = runner.answer("Q-07", "Can Zenith Global ship to Germany?")
    assert ans.budget_exceeded is True
    assert ans.resolved is False
    assert ans.tool_call_count == 1  # trail salvaged from the last good state


def test_runner_stops_at_budget_midstream():
    def _sequential(n):
        msgs = [HumanMessage(content="q")]
        for i in range(n):
            tid = f"s{i}"
            msgs.append(AIMessage(content="", tool_calls=[{"name": "find_company", "args": {"name": f"C{i}"}, "id": tid}]))
            msgs.append(ToolMessage(content='{"status": "found"}', tool_call_id=tid))
        return msgs

    state_a = {"messages": _sequential(8)}  # under budget

    # 8 sequential + a 3-call parallel batch, all executed -> 11 executed (crosses budget)
    msgs_b = _sequential(8)
    batch_ids = ["b0", "b1", "b2"]
    msgs_b.append(AIMessage(content="", tool_calls=[
        {"name": "find_company", "args": {"name": f"B{i}"}, "id": bid} for i, bid in enumerate(batch_ids)
    ]))
    for bid in batch_ids:
        msgs_b.append(ToolMessage(content='{"status": "found"}', tool_call_id=bid))
    state_b = {"messages": msgs_b}

    state_c = {"messages": _sequential(13)}  # must never be consumed

    class _CountingAgent:
        def __init__(self, states):
            self._states = states
            self.consumed = 0

        def stream(self, _inputs, config=None, stream_mode=None):
            for s in self._states:
                self.consumed += 1
                yield s

    agent = _CountingAgent([state_a, state_b, state_c])
    model = _FakeModel(FinalAnswer(answer="done", resolved=True, reasoning_summary="budget hit"))
    runner = AgentRunner(model=model, agent=agent)
    ans = runner.answer("Q-XX", "many")

    assert agent.consumed == 2  # stopped mid-stream at the budget-crossing state; state_c never pulled
    assert ans.budget_exceeded is True
    assert ans.tool_call_count == 10  # capped — never surfaces more than the budget
    assert len(ans.tool_calls) == 10
