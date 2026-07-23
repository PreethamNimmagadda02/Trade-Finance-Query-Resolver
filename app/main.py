from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.responses import RedirectResponse

from app.agent import AgentRunner, build_agent
from app.data import init_db
from app.schemas import AgentAnswer, AskBatchRequest, AskRequest

load_dotenv()
init_db()  # load the workbook once at startup

app = FastAPI(title="Bluno Trade Finance Query Resolver")


@lru_cache(maxsize=1)
def get_runner() -> AgentRunner:
    """Build the ReAct agent once and reuse it. Lazy so importing the app
    without an API key (e.g. in tests that override this) does not fail."""
    return build_agent()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.post("/ask", response_model=AgentAnswer)
def ask(req: AskRequest, runner: AgentRunner = Depends(get_runner)) -> AgentAnswer:
    return runner.answer(req.question_id, req.question)


@app.post("/ask_batch", response_model=list[AgentAnswer])
def ask_batch(req: AskBatchRequest, runner: AgentRunner = Depends(get_runner)) -> list[AgentAnswer]:
    return [runner.answer(q.question_id, q.question) for q in req.questions]
