from syftr.studies import SearchSpace
from syftr.validation import are_valid_parameters


def test_validate_baseline(baseline_template):
    """Validates all of the baseline templates based on default search space."""
    ss = SearchSpace()
    assert are_valid_parameters(ss, baseline_template)
