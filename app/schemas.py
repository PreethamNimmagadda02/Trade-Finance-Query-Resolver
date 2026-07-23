# app/schemas.py
from pydantic import BaseModel


class ToolCall(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: str  # what the tool returned (summarised is fine)


class AgentAnswer(BaseModel):
    question_id: str
    answer: str
    resolved: bool  # False when the question cannot be answered from the data
    tool_calls: list[ToolCall]
    tool_call_count: int
    budget_exceeded: bool
    reasoning_summary: str  # 1-3 sentences on how the agent reached its answer


class FinalAnswer(BaseModel):
    """Structured output the agent emits after reasoning."""

    answer: str
    resolved: bool
    reasoning_summary: str


class AskRequest(BaseModel):
    question_id: str = ""
    question: str


class AskBatchRequest(BaseModel):
    questions: list[AskRequest]
