import json
import time
import typing as T

from llama_index.core.bridge.pydantic import PrivateAttr

from syftr.logger import logger
from syftr.studies import TimeoutConfig


class EmbeddingPreemptiveTimeoutError(Exception):
    """Embeddings expected to go over time."""

    def __init__(self, metadata):
        message = f"Embeddings expected to exceed max time limit: {json.dumps(metadata, sort_keys=True)}"
        super().__init__(message)


class EmbeddingTimeoutMixin:
    _chunks_processed: T.Any = PrivateAttr()
    _chunks_remaining: T.Any = PrivateAttr()
    _done_building_index: T.Any = PrivateAttr()
    _seconds_per_chunk: T.Any = PrivateAttr()
    _time_remaining: T.Any = PrivateAttr()
    _time_start: T.Any = PrivateAttr()
    _timeout_config: T.Any = PrivateAttr()
    _total_chunks: T.Any = PrivateAttr()
    _total_time: T.Any = PrivateAttr()
    _warning_logged: T.Any = PrivateAttr()

    def __init__(
        self,
        *args,
        timeout_config: TimeoutConfig = TimeoutConfig(),
        total_chunks: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._timeout_config = timeout_config
        self._total_chunks = total_chunks

        self._chunks_processed = 0
        self._time_start = None
        self._warning_logged = False

    def reset_timeouts(
        self,
        disable_timeouts: bool = False,
        total_chunks: int | None = None,
        timeout_config: TimeoutConfig | None = None,
    ):
        """Enable timeouts to be reset or disabled.

        Useful when index building has finished and we are moving to evaluation,
        or when reusing for a separate index.
        """
        self._chunks_processed = 0
        self._time_start = None
        self._done_building_index = bool(disable_timeouts)
        self._warning_logged = False

        if total_chunks is not None:
            self._total_chunks = total_chunks

        if timeout_config is not None:
            self._timeout_config = timeout_config

    @property
    def metadata(self) -> T.Dict[str, T.Any]:
        return {
            "_chunks_processed": self._chunks_processed,
            "_chunks_remaining": self._chunks_remaining,
            "_done_building_index": self._done_building_index,
            "_seconds_per_chunk": self._seconds_per_chunk,
            "_time_remaining": self._time_remaining,
            "_time_start": self._time_start,
            "_timeout_config": self._timeout_config.dict(),
            "_total_chunks": self._total_chunks,
            "_total_time": self._total_time,
            "_warning_logged": self._warning_logged,
        }

    def _check_remaining_time(self):
        if self._total_chunks == 0:
            return

        if self._done_building_index:
            return

        min_chunks_met = (
            self._chunks_processed
            > self._timeout_config.embedding_min_chunks_to_process
        )
        min_time_spent = (
            self._total_time > self._timeout_config.embedding_min_time_to_process
        )

        if min_chunks_met or min_time_spent:
            self._seconds_per_chunk = self._total_time / self._chunks_processed
            self._chunks_remaining = self._total_chunks - self._chunks_processed
            self._time_remaining = self._seconds_per_chunk * self._chunks_remaining
            if self._chunks_processed % 3200 == 0:
                logger.info(
                    "%s/%s chunks processed at %s it/s after %s s. Estimated time remaining is %s s",
                    self._chunks_processed,
                    self._total_chunks,
                    self._total_time,
                    self._chunks_processed / self._total_time,
                    self._time_remaining,
                )

            if (
                self._total_time + self._time_remaining
                > self._timeout_config.embedding_max_time
            ):
                if self._timeout_config.embedding_timeout_active:
                    raise EmbeddingPreemptiveTimeoutError(self.metadata)
                if not self._warning_logged:
                    logger.warning(
                        "Not raising embedding timeout because it is deactivated"
                    )
                    self._warning_logged = True

    def _embed(self, sentences: T.List[str], *args, **kwargs) -> T.List[T.List[float]]:
        if self._time_start is None:
            self._time_start = time.time()

        result = super()._embed(sentences, *args, **kwargs)  # type: ignore

        self._total_time = time.time() - self._time_start
        self._chunks_processed += len(sentences)
        self._check_remaining_time()
        return result

    def _compute_embeddings(
        self, texts: T.List[str], prefix: str | None = None
    ) -> T.List[T.List[float]]:
        if self._time_start is None:
            self._time_start = time.time()

        result = super()._compute_embeddings(texts, prefix=prefix)  # type: ignore

        self._total_time = time.time() - self._time_start
        self._chunks_processed += len(texts)
        self._check_remaining_time()
        return result
