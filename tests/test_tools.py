import json

from app.data import init_db, DEFAULT_DATASET_PATH
from app import tools

init_db(DEFAULT_DATASET_PATH)


def _call(tool, **kwargs):
    return json.loads(tool.invoke(kwargs))


def test_find_company_found():
    out = _call(tools.find_company, name="ACME Exports")
    assert out["status"] == "found"
    assert out["destination_country"] == "Germany"


def test_find_company_not_found():
    out = _call(tools.find_company, name="Zenith Global")
    assert out["status"] == "NOT_FOUND"


def test_check_country_clearance_restricted():
    out = _call(tools.check_country_clearance, country="Vietnam")
    assert out["status"] == "found"
    assert out["clearance"] == "RESTRICTED"


def test_get_credit_headroom():
    out = _call(tools.get_credit_headroom, name="ACME Exports")
    assert out["available_headroom_usd"] == 320000


def test_get_credit_headroom_not_found():
    out = _call(tools.get_credit_headroom, name="Zenith Global")
    assert out["status"] == "NOT_FOUND"


def test_get_company_shipments():
    out = _call(tools.get_company_shipments, name="Meridian Logistics")
    values = [s["value_usd"] for s in out["shipments"]]
    assert max(values) == 460000


def test_list_companies_filter():
    out = _call(tools.list_companies, destination_country="Germany", kyc_status="VERIFIED")
    names = {c["company_name"] for c in out["companies"]}
    assert names == {"ACME Exports", "Nordic Timber"}


def test_all_tools_exported():
    assert len(tools.ALL_TOOLS) == 5
