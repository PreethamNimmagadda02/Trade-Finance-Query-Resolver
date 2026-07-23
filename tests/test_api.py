from fastapi.testclient import TestClient

from app.main import app, get_runner
from app.schemas import AgentAnswer, ToolCall


class _StubRunner:
    def answer(self, question_id, question):
        return AgentAnswer(
            question_id=question_id,
            answer=f"stub answer for {question}",
            resolved=True,
            tool_calls=[ToolCall(tool_name="find_company", tool_input={"name": "ACME Exports"}, tool_output="found")],
            tool_call_count=1,
            budget_exceeded=False,
            reasoning_summary="stub",
        )


app.dependency_overrides[get_runner] = lambda: _StubRunner()
client = TestClient(app)


def test_ask():
    resp = client.post("/ask", json={"question_id": "Q-01", "question": "Is ACME cleared?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["question_id"] == "Q-01"
    assert body["tool_call_count"] == 1
    assert body["budget_exceeded"] is False


def test_ask_batch():
    resp = client.post(
        "/ask_batch",
        json={"questions": [
            {"question_id": "Q-01", "question": "a"},
            {"question_id": "Q-02", "question": "b"},
        ]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert body[1]["question_id"] == "Q-02"
