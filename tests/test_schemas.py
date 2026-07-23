# tests/test_schemas.py
from app.schemas import ToolCall, AgentAnswer, FinalAnswer, AskRequest, AskBatchRequest


def test_agent_answer_roundtrip():
    ans = AgentAnswer(
        question_id="Q-01",
        answer="Yes.",
        resolved=True,
        tool_calls=[ToolCall(tool_name="find_company", tool_input={"name": "ACME Exports"}, tool_output="found")],
        tool_call_count=1,
        budget_exceeded=False,
        reasoning_summary="Looked up the company.",
    )
    d = ans.model_dump()
    assert d["question_id"] == "Q-01"
    assert d["tool_calls"][0]["tool_name"] == "find_company"
    assert d["tool_call_count"] == 1
    assert d["budget_exceeded"] is False


def test_final_answer_fields():
    fa = FinalAnswer(answer="No.", resolved=False, reasoning_summary="Not in data.")
    assert fa.resolved is False


def test_ask_request_defaults_question_id():
    r = AskRequest(question="Can ACME ship?")
    assert r.question_id == ""


def test_ask_batch_request():
    b = AskBatchRequest(questions=[AskRequest(question_id="Q-01", question="x")])
    assert b.questions[0].question_id == "Q-01"
