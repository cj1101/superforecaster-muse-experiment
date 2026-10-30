"""SF-BL-10547 regression tests: validate_submission_batch must raise a
named ValueError for NaN / non-finite / bool / non-integer-float question
ids, and duplicate detection must catch two distinct NaN objects.
Kill criterion: any NaN-id payload returns None from the gate.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "metaculus_bot"))
from decimal import Decimal
import numpy as np
import pytest
from metaculus_client import validate_submission_batch

P = "probability_yes"

@pytest.mark.parametrize("qid,needle", [
    (float("nan"), "nan"), (float("inf"), "non-finite"),
    (float("-inf"), "non-finite"), (True, "non-int"), (False, "non-int"),
    (1.5, "non-integer float"), (-0.25, "non-integer float"),
    (Decimal("NaN"), "nan"), (Decimal("Infinity"), "non-finite"),
    (np.float64("nan"), "nan"), (np.float32("nan"), "nan"),
    (np.float64("inf"), "non-finite"),
])
def test_bad_ids_raise_named(qid, needle):
    with pytest.raises(ValueError) as ei:
        validate_submission_batch([{"question": qid, P: 0.5}])
    assert needle in str(ei.value).lower()

def test_two_distinct_nan_objects_raise():
    a, b = float("nan"), float("nan")
    assert a is not b and a != b
    with pytest.raises(ValueError):
        validate_submission_batch([{"question": a, P: 0.5},
                                   {"question": b, P: 0.6}])

def test_valid_and_duplicate_ints_unchanged():
    validate_submission_batch([{"question": 123, P: 0.5}])
    with pytest.raises(ValueError, match="duplicate"):
        validate_submission_batch([{"question": 123, P: 0.5},
                                   {"question": 123, P: 0.6}])
