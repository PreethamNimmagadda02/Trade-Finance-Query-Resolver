from __future__ import annotations

import json

from langchain_core.tools import tool

from app.data import get_db


@tool
def find_company(name: str) -> str:
    """Look up a counterparty by name. Use this first when a question names a
    company, to confirm it exists in Bluno's records and to get its destination
    country and KYC status. Returns status NOT_FOUND if the company is not in the
    data — in that case the question cannot be answered and must not be guessed."""
    c = get_db().resolve_company(name)
    if c is None:
        return json.dumps({"status": "NOT_FOUND", "name": name})
    return json.dumps({"status": "found", **c})


@tool
def check_country_clearance(country: str) -> str:
    """Check whether a destination country is cleared for shipment. Returns a
    clearance of CLEARED (permitted), RESTRICTED (permitted only with extra
    documentation and manual compliance sign-off), or PROHIBITED (no shipments
    permitted), plus an explanatory note. Use this to answer whether a company can
    ship to its destination."""
    s = get_db().get_clearance(country)
    if s is None:
        return json.dumps({"status": "NOT_FOUND", "country": country})
    return json.dumps({"status": "found", **s})


@tool
def get_credit_headroom(name: str) -> str:
    """Get a company's available credit headroom (credit_limit minus outstanding
    exposure, in USD) along with the underlying limit and exposure. Use this for
    any question about available credit or how much more a company can borrow/ship.
    Headroom of 0 means credit is fully utilised. Returns NOT_FOUND for unknown
    companies."""
    c = get_db().get_credit(name)
    if c is None:
        return json.dumps({"status": "NOT_FOUND", "name": name})
    return json.dumps({"status": "found", **c})


@tool
def get_company_shipments(name: str) -> str:
    """List all shipments belonging to a company, each with its destination
    counterparty name, value in USD, and status. Use this to find a company's
    highest-value shipment or to reason about its shipment activity. Returns
    NOT_FOUND for unknown companies."""
    c = get_db().get_shipments(name)
    if c is None:
        return json.dumps({"status": "NOT_FOUND", "name": name})
    return json.dumps({"status": "found", **c})


@tool
def list_companies(destination_country: str | None = None, kyc_status: str | None = None) -> str:
    """List companies, optionally filtered by destination_country and/or
    kyc_status (e.g. VERIFIED, PENDING). Use this for questions that compare or
    rank multiple companies (e.g. 'which VERIFIED company in Germany has the most
    headroom') so you can get the candidate set without guessing which companies
    exist."""
    rows = get_db().filter_companies(destination_country=destination_country, kyc_status=kyc_status)
    return json.dumps({"status": "found", "companies": rows})


ALL_TOOLS = [
    find_company,
    check_country_clearance,
    get_credit_headroom,
    get_company_shipments,
    list_companies,
]
