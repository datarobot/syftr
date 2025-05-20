from itertools import batched
from typing import Any, List, Sequence

import requests
from llama_index.core.embeddings import BaseEmbedding
from pydantic.networks import HttpUrl
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_fixed,
    wait_random,
)
from transformers import AutoTokenizer

from syftr.embeddings.timeouts import EmbeddingTimeoutMixin


def _print_retry_error(retry_state):
    print(retry_state)


def _retry_if_not_specific_http_error(exception):
    if isinstance(exception, requests.exceptions.HTTPError):
        if exception.response.status_code == 429:
            return True
        elif exception.response.status_code >= 500:
            return True
        else:
            return False

    return True


class HFEndpointEmbeddings(BaseEmbedding, EmbeddingTimeoutMixin):
    def __init__(
        self,
        hf_api_url: HttpUrl,
        hf_api_key: str,
        max_length: int | None = None,
        hf_embedding_batch_size: int = 32,
        query_prefix: str = "",
        text_prefix: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._hf_api_url = hf_api_url
        self._hf_headers = {"Authorization": f"Bearer {hf_api_key}"}
        self._max_length = max_length
        self._query_prefix = query_prefix
        self._text_prefix = text_prefix
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.embed_batch_size = hf_embedding_batch_size

    def _truncate_tokens_to_max(
        self, text_batch: Sequence[str], tokenizer: AutoTokenizer, max_length=None
    ):
        truncated_text_batch = []
        for text in text_batch:
            tokens = tokenizer(
                text,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            )
            truncated_text_batch.append(
                tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            )
        # replace empty strings with "blank"
        # empty strings cause errors in the HF endpoint
        truncated_text_batch = [
            "blank" if not text else text for text in truncated_text_batch
        ]
        return truncated_text_batch

    def _query_hf_endpoint(self, payload):
        response = requests.post(
            self._hf_api_url, headers=self._hf_headers, json=payload
        )
        response.raise_for_status()
        return response.json()

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_fixed(60) + wait_random(0, 60),
        after=_print_retry_error,
        retry=retry_if_exception(_retry_if_not_specific_http_error),
    )
    def get_embeddings_with_retry(self, text_batch_truncated):
        embeds = self._query_hf_endpoint({"inputs": text_batch_truncated})
        return embeds

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embeddings([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._compute_embeddings(texts, prefix=self._text_prefix)

    def _get_query_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._compute_embeddings(texts, prefix=self._query_prefix)

    def _compute_embeddings(
        self, texts: List[str], prefix: str | None = None
    ) -> List[List[float]]:
        hf_embeds = []
        for batch in batched(texts, self.embed_batch_size):
            # add prefix
            text_batch = [prefix + text for text in batch] if prefix else batch
            text_batch_truncated = self._truncate_tokens_to_max(
                text_batch, self._tokenizer, self._max_length
            )

            embeds = self.get_embeddings_with_retry(text_batch_truncated)

            hf_embeds.extend(embeds)
        return hf_embeds
