import pytest

from syftr.embeddings.timeouts import EmbeddingPreemptiveTimeoutError
from syftr.studies import TimeoutConfig


@pytest.mark.parametrize(
    "total_chunks,total_time,chunks_processed,should_time_out",
    [
        (0, 1000, 10, False),  # Too slow, but shouldn't check
        (1000, 10, 10, False),  # Fast enough to complete
        (1000, 10, 11, False),  # Fast enough to complete
        (1000, 11, 10, True),  # Not fast enough to complete
        (1000, 990, 995, False),  # Almost done
        (1000, 990, 10, True),  # Not enough chunks, but enough time elapsed
        (100000, 5, 101, True),  # Not enough time elapsed, but enough chunks
    ],
)
def test_timeout_logic(
    total_chunks, total_time, chunks_processed, should_time_out, bge_small
):
    bge_small.reset_timeouts(
        timeout_config=TimeoutConfig(
            embedding_timeout_active=True,
            embedding_max_time=1000,
            embedding_min_chunks_to_process=100,
            embedding_min_time_to_process=10,
        )
    )
    bge_small._total_chunks = total_chunks
    bge_small._total_time = total_time
    bge_small._chunks_processed = chunks_processed

    if should_time_out:
        with pytest.raises(EmbeddingPreemptiveTimeoutError):
            bge_small._check_remaining_time()
    else:
        bge_small._check_remaining_time()
