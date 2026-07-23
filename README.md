# Bluno Trade Finance Query Resolver

A ReAct agent, exposed via FastAPI, that answers natural-language trade-finance
questions about counterparties and shipments. It reasons over tools defined on the
provided workbook, stays within a 10-tool-call budget, and returns a structured
answer with the full tool-call trail.

## Quick start

```bash
uv sync
cp .env.example .env    # then put your ANTHROPIC_API_KEY in .env
uv run uvicorn app.main:app
```

The service listens on http://127.0.0.1:8000 (interactive docs at `/docs`).

### Ask one question

```bash
curl -s http://127.0.0.1:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question_id": "Q-01", "question": "Is ACME Exports cleared to ship to their destination country?"}'
```

### Ask the batch (how the 11 test questions are submitted)

```bash
curl -s http://127.0.0.1:8000/ask_batch \
  -H 'content-type: application/json' \
  -d '{"questions": [{"question_id": "Q-01", "question": "Is ACME Exports cleared to ship to their destination country?"}]}'
```

## 1. Agent design

LangGraph's prebuilt `create_react_agent` drives the reason→act→observe loop on
Anthropic `claude-sonnet-5` (via `langchain-anthropic`). A thin wrapper
(`app/agent.py`) streams the graph, enforces the tool budget, walks the message
trail into `ToolCall`s, and runs one final structured-output call to produce the
`{answer, resolved, reasoning_summary}` fields. The agent chooses its own actions
per question — nothing is hard-coded per question.

## 2. Tools

Boundaries follow the data's distinct concerns rather than one-tool-per-sheet:

- `find_company(name)` — resolve a counterparty; returns id, destination country,
  KYC status, or `NOT_FOUND`.
- `check_country_clearance(country)` — CLEARED / RESTRICTED / PROHIBITED + note.
- `get_credit_headroom(name)` — credit limit, exposure, and derived headroom.
- `get_company_shipments(name)` — a company's shipments (for highest-value joins).
- `list_companies(destination_country?, kyc_status?)` — filtered candidate set for
  multi-company comparisons.

Each entity tool resolves the company name internally and returns a structured
`NOT_FOUND` for unknown names, which is what lets the agent answer honestly when an
entity is absent.

## 3. Budget, multi-entity, and anti-fabrication

- **Budget:** the agent is capped at 10 tool calls. The LangGraph `recursion_limit`
  bounds the loop; if the agent tries to exceed it we stop, return the best answer
  so far, and set `budget_exceeded=True`. `tool_call_count` reports the actual
  number of calls. Well-scoped tools mean the hardest questions use ~3–4 calls.
- **Multi-entity:** `list_companies` gives the agent the candidate set so it can
  compare/rank several companies without brute-forcing.
- **Anti-fabrication:** tools return `NOT_FOUND` for unknown entities; the system
  prompt forbids answering beyond tool output; the synthesis step sets
  `resolved=False` when a required entity is missing (e.g. a company not in the
  records).

## 4. Structured answer and tool-call trail

After the ReAct loop, the wrapper collects every `AIMessage` tool call paired with
its `ToolMessage` output into an ordered `tool_calls` list, counts them, and marks
`budget_exceeded`. A single `with_structured_output(FinalAnswer)` call over that
trail produces `answer`, `resolved`, and `reasoning_summary`. This call is not a
tool call and is not counted against the budget.

## 5. Assumptions

1. "Cleared to ship to destination country" = sanctions clearance (Sanctions
   sheet), independent of KYC; KYC is reported only when the question asks.
2. Available credit headroom = credit limit − outstanding exposure; 0 = fully
   utilised.
3. RESTRICTED = conditionally permitted (extra documentation + manual sign-off),
   distinct from CLEARED and PROHIBITED.
4. Company resolution is case-insensitive exact match on `company_name`; an unknown
   name is `NOT_FOUND` and yields `resolved=False`.
5. In Shipments, `company_id` owns the shipment and `counterparty_name` is the
   destination party whose KYC is checked (e.g. Q-06).
6. All monetary values are USD.
7. `tool_call_count` counts tool executions in the trail; the final structured
   answer call is not counted.

## Testing

```bash
uv run pytest          # deterministic data/tool/agent/API tests (no API key needed)
```
