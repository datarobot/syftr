import pytest
from tenacity import retry, stop_after_attempt, wait_exponential


class _TestException(Exception):
    pass


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=2, min=2, max=3),
    reraise=True,
)
def bad_function():
    raise _TestException()


def test_custom_exception():
    with pytest.raises(_TestException) as exc_info:
        bad_function()
    assert type(exc_info.value) is _TestException
