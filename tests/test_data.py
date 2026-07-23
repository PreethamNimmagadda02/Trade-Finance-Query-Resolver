# tests/test_data.py
from app.data import init_db, DEFAULT_DATASET_PATH

db = init_db(DEFAULT_DATASET_PATH)


def test_resolve_company_case_insensitive():
    c = db.resolve_company("acme exports")
    assert c["company_id"] == "C-101"
    assert c["destination_country"] == "Germany"
    assert c["kyc_status"] == "VERIFIED"


def test_resolve_company_not_found():
    assert db.resolve_company("Zenith Global") is None


def test_headroom_math():
    # ACME: 500000 - 180000 = 320000
    assert db.get_credit("ACME Exports")["available_headroom_usd"] == 320000
    # Pacific AgriCorp: 250000 - 250000 = 0 (fully utilised)
    assert db.get_credit("Pacific AgriCorp")["available_headroom_usd"] == 0


def test_clearance_three_states():
    assert db.get_clearance("Germany")["clearance"] == "CLEARED"
    assert db.get_clearance("Vietnam")["clearance"] == "RESTRICTED"
    assert db.get_clearance("Russia")["clearance"] == "PROHIBITED"


def test_clearance_not_found():
    assert db.get_clearance("Atlantis") is None


def test_shipments_for_meridian():
    ships = db.get_shipments("Meridian Logistics")["shipments"]
    # Meridian is C-103 -> S-003 (ACME, 300000) and S-004 (Pacific AgriCorp, 460000)
    top = max(ships, key=lambda s: s["value_usd"])
    assert top["counterparty_name"] == "Pacific AgriCorp"
    assert top["value_usd"] == 460000


def test_filter_companies_verified_germany():
    rows = db.filter_companies(destination_country="Germany", kyc_status="VERIFIED")
    names = {r["company_name"] for r in rows}
    assert names == {"ACME Exports", "Nordic Timber"}
