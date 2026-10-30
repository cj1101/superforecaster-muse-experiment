"""SF-BL-10469 regression tests: validate_forecast_payload must flag any
payload question id that is not a plain int (float/str/bool/...) as a
named non-int problem. Kill criterion: question=42.0 validated clean
against id 42, or question=True validated clean against id 1.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "metaculus_bot"))
import pytest
from metaculus_client import validate_forecast_payload

def q(qid):
    return {"id": qid, "type": "binary"}

def pay(qid):
    return {"question": qid, "probability_yes": 0.5,
            "probability_yes_per_category": None, "continuous_cdf": None}

@pytest.mark.parametrize("qid,against", [
    (42.0, 42), (1.0, 1), (0.0, 0), (42.5, 42),
    (float("nan"), 42), (float("inf"), 42),
    (True, 1), (False, 0),
    ("42", 42), ("1", 1),
    (None, 42), ([42], 42), ({"id": 42}, 42), (b"42", 42),
])
def test_non_int_question_id_flagged_named(qid, against):
    problems = validate_forecast_payload(q(against), pay(qid))
    assert problems, f"{qid!r} vs {against} validated clean"
    assert any("non-int" in p for p in problems), problems

@pytest.mark.parametrize("qid", [42, 1, 0])
def test_plain_int_question_id_clean(qid):
    assert validate_forecast_payload(q(qid), pay(qid)) == []

def test_mismatched_plain_int_still_flagged():
    assert validate_forecast_payload(q(42), pay(43))
