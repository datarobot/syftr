import math
import os
import typing as T
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

import pandas as pd
from optuna import Trial
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    DiscreteUniformDistribution,
    FloatDistribution,
    IntDistribution,
    LogUniformDistribution,
    UniformDistribution,
)
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from syftr.configuration import NDIGITS, cfg
from syftr.helpers import (
    get_max_float,
    get_max_int,
    get_min_float,
    get_min_int,
    get_unique_bools,
    get_unique_strings,
)
from syftr.llm import LLMs
from syftr.storage import (
    CragTask3HF,
    DRDocsHF,
    FinanceBenchHF,
    HotPotQAHF,
    PartitionMap,
    SyftrQADataset,
)

ParamDict = T.Dict[str, str | int | float | bool]


class SearchSpaceMixin(ABC):
    """Common interface for all search space classes."""

    model_config = ConfigDict(extra="forbid")  # Forbids unknown fields

    @abstractmethod
    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        """Subclasses must return the distributions defining their parameter search space."""
        pass

    @abstractmethod
    def get_cardinality(self) -> int:
        """Subclasses must define a method to compute the cardinality of their space."""
        pass

    def sample(self, trial: Trial, prefix: str = "") -> ParamDict:
        """Sample concrete parameters from the search space distributions."""
        return {
            name: self._suggest_from_distribution(trial, name, dist)
            for name, dist in self.build_distributions(prefix).items()
        }

    def _suggest_from_distribution(
        self, trial: Trial, name: str, dist: BaseDistribution
    ) -> T.Any:
        if isinstance(dist, CategoricalDistribution):
            return trial.suggest_categorical(name, dist.choices)
        elif isinstance(dist, IntDistribution):
            return trial.suggest_int(
                name, low=dist.low, high=dist.high, step=dist.step, log=dist.log
            )
        elif isinstance(dist, FloatDistribution):
            value = trial.suggest_float(
                name, low=dist.low, high=dist.high, step=dist.step, log=dist.log
            )
            return round(value, ndigits=NDIGITS)
        elif isinstance(dist, DiscreteUniformDistribution):
            value = trial.suggest_discrete_uniform(
                name, low=dist.low, high=dist.high, q=dist.q
            )
            return round(value, ndigits=NDIGITS)
        elif isinstance(dist, LogUniformDistribution):
            return trial.suggest_loguniform(name, low=dist.low, high=dist.high)
        elif isinstance(dist, UniformDistribution):
            return trial.suggest_uniform(name, low=dist.low, high=dist.high)
        else:
            raise NotImplementedError(f"Unsupported distribution type: {type(dist)}")


class Splitter(BaseModel, SearchSpaceMixin):
    chunk_min_exp: int = Field(
        default=6, description="Minimum exponent for chunk size (2^6 = 64)."
    )
    chunk_max_exp: int = Field(
        default=12, description="Maximum exponent for chunk size (2^12 = 4096)."
    )
    chunk_overlap_frac_min: float = Field(
        default=0.0, description="Minimum fraction of overlap between chunks."
    )
    chunk_overlap_frac_max: float = Field(
        default=0.75, description="Maximum fraction of overlap between chunks."
    )
    chunk_overlap_frac_step: float = Field(
        default=0.25, description="Step size for chunk overlap fraction."
    )
    methods: T.List[str] = Field(
        default_factory=lambda: [
            "html",
            "recursive",
            "sentence",
            "token",
        ],
        description="List of available text splitting methods.",
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}splitter_method": "recursive",
            f"{prefix}splitter_chunk_exp": 10,
            f"{prefix}splitter_chunk_overlap_frac": 0.25,
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}splitter_method": CategoricalDistribution(self.methods),
            f"{prefix}splitter_chunk_exp": IntDistribution(
                self.chunk_min_exp, self.chunk_max_exp, step=1
            ),
            f"{prefix}splitter_chunk_overlap_frac": FloatDistribution(
                self.chunk_overlap_frac_min,
                self.chunk_overlap_frac_max,
                step=self.chunk_overlap_frac_step,
            ),
        }

    def sample(self, trial: Trial, prefix: str = "") -> ParamDict:
        params: ParamDict = {
            f"{prefix}splitter_method": trial.suggest_categorical(
                f"{prefix}splitter_method", self.methods
            ),
            f"{prefix}splitter_chunk_exp": trial.suggest_int(
                f"{prefix}splitter_chunk_exp", self.chunk_min_exp, self.chunk_max_exp
            ),
        }
        params[f"{prefix}splitter_chunk_overlap_frac"] = trial.suggest_float(
            f"{prefix}splitter_chunk_overlap_frac",
            self.chunk_overlap_frac_min,
            self.chunk_overlap_frac_max,
            step=self.chunk_overlap_frac_step,
        )
        return params

    def get_cardinality(self) -> int:
        chunk_exp_card = self.chunk_max_exp - self.chunk_min_exp + 1
        method_card = len(self.methods)
        overlap_card = get_dist_cardinality(
            self.chunk_overlap_frac_min,
            self.chunk_overlap_frac_max,
            self.chunk_overlap_frac_step,
        )
        return method_card * chunk_exp_card * overlap_card


LOCAL_EMBEDDING_MODELS = (
    [model.model_name for model in cfg.local_models.embedding]
    if cfg.local_models.embedding
    else []
)

DEFAULT_EMBEDDING_MODELS: T.List[str] = list(
    set(
        [
            "BAAI/bge-small-en-v1.5",  # first embedding model is the default
            "BAAI/bge-large-en-v1.5",
            "thenlper/gte-large",
            "mixedbread-ai/mxbai-embed-large-v1",
            "WhereIsAI/UAE-Large-V1",
            "avsolatorio/GIST-large-Embedding-v0",
            "w601sxs/b1ade-embed",
            "Labib11/MUG-B-1.6",
            "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "BAAI/bge-base-en-v1.5",
            "FinLang/finance-embeddings-investopedia",
            "baconnier/Finance2_embedding_small_en-V1.5",
            # "thomaskim1130/stella_en_400M_v5-FinanceRAG-v2",
            # "Alibaba-NLP/gte-large-en-v1.5",
            # "Alibaba-NLP/gte-base-en-v1.5",
            # "llmrails/ember-v1",
            # "jamesgpt1/sf_model_e5",
            # "mixedbread-ai/mxbai-embed-2d-large-v1",
            # "intfloat/e5-large-v2",
        ]
        + LOCAL_EMBEDDING_MODELS
    )
)


ALL_LLMS = list(LLMs.keys())

LOCAL_LLMS = (
    [model.model_name for model in cfg.local_models.generative]
    if cfg.local_models.generative
    else []
)

DEFAULT_LLMS: T.List[str] = list(
    set(
        [
            "gpt-4o-mini",  # first LLM is the default
            "anthropic-haiku-35",
            "gemini-flash",
            # "gemini-flash2",
            # "gemini-pro",
            # "llama-33-70B",   # not enough capacity
            # "mistral-large",  # not enough capacity
            # "phi-4",          # not enough capacity
            # "anthropic-sonnet-35",
            # "gpt-4o-std",
            "o3-mini",
        ]
        + LOCAL_LLMS
    )
)
assert set(DEFAULT_LLMS).issubset(set(ALL_LLMS))

RESPONSE_SYNTHESIZER_LLMS: T.List[str] = [
    "gpt-4o-mini",  # first LLM is the default
    "gpt-4o-std",
    "gpt-35-turbo",
    "anthropic-sonnet-35",
    "anthropic-haiku-35",
    "llama-33-70B",
    "gemini-pro",
    "gemini-flash",
    "gemini-flash2",
    # "gemini-flash-think-exp",
    "mistral-large",
    "together-r1",
    "together-V3",
] + LOCAL_LLMS
assert set(RESPONSE_SYNTHESIZER_LLMS).issubset(set(ALL_LLMS))

FUNCTION_CALLING_LLMS: T.List[str] = [
    "gpt-4o-mini",  # first LLM is the default
    "gpt-4o-std",
    "gpt-35-turbo",
    "anthropic-sonnet-35",
    "anthropic-haiku-35",
    "llama-33-70B",
    # "gemini-flash2",
    # "gemini-pro",
    # "gemini-flash",
    # "gemini-flash-think-exp",
    "mistral-large",
] + LOCAL_LLMS
assert set(FUNCTION_CALLING_LLMS).issubset(set(ALL_LLMS))

CHEAP_LLMS: T.List[str] = [
    "gpt-4o-mini",  # first LLM is the default
    "anthropic-haiku-35",
    "gemini-flash2",
] + LOCAL_LLMS
assert set(CHEAP_LLMS).issubset(set(ALL_LLMS))
assert set(CHEAP_LLMS).issubset(set(ALL_LLMS))

NON_REASONING_LLMS: T.List[str] = list(
    set(
        [
            "gpt-4o-mini",  # first LLM is the default
            "gpt-4o-std",
            "gpt-35-turbo",
            "anthropic-sonnet-35",
            "anthropic-haiku-35",
            "llama-33-70B",
            "gemini-pro",
            "gemini-flash",
            "gemini-flash2",
            "mistral-large",
        ]
        + LOCAL_LLMS
    )
)
assert set(NON_REASONING_LLMS).issubset(set(ALL_LLMS))

RAG_MODES: T.List[str] = [
    "rag",  #  first mode is the default
    "react_rag_agent",
    "critique_rag_agent",
    "sub_question_rag",
    "lats_rag_agent",
    "no_rag",
]

TEMPLATE_NAMES = [
    "default",  # first template is the default
    "concise",
    "CoT",
    "finance-expert",
]

PARAMETERS = [
    "rag_retriever",
    "splitter",
    "additional_context",
    "few_shot_retriever",
    "hyde",
    "reranker",
    "rag_mode",
    "sub_question_rag",
    "critique_rag_agent",
    "lats_rag_agent",
    "react_rag_agent",
    "response_synthesizer_llm",
    "template_name",
]


def get_dist_cardinality(min: int | float, max: int | float, step: int | float) -> int:
    """Returns the cardinality of an integer or float distribution"""
    assert min <= max
    assert step > 0
    return int((max - min) / step) + 1


class TopK(BaseModel, SearchSpaceMixin):
    kmin: int = Field(
        default=2, description="Minimum value for number of items to retrieve."
    )
    kmax: int = Field(
        default=20, description="Maximum value for number of items to retrieve."
    )
    log: bool = Field(
        default=False,
        description="Whether to use a logarithmic scale instead of linear for top_k.",
    )
    step: int = Field(default=1, description="Step size for top_k.")

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}top_k": 5,
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        name = f"{prefix}top_k"
        return {
            name: IntDistribution(self.kmin, self.kmax, log=self.log, step=self.step)
        }

    def get_cardinality(self) -> int:
        return get_dist_cardinality(self.kmin, self.kmax, self.step)


class Hybrid(BaseModel, SearchSpaceMixin):
    bm25_weight_min: float = Field(
        default=0.1, description="Minimum weight for BM25 in hybrid retrieval."
    )
    bm25_weight_max: float = Field(
        default=0.9, description="Maximum weight for BM25 in hybrid retrieval."
    )
    bm25_weight_step: float = Field(
        default=0.1, description="Step size for BM25 weight."
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}hybrid_bm25_weight": 0.5,
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        bm25_weight = f"{prefix}hybrid_bm25_weight"
        return {
            bm25_weight: FloatDistribution(
                self.bm25_weight_min, self.bm25_weight_max, step=self.bm25_weight_step
            ),
        }

    def get_cardinality(self) -> int:
        return get_dist_cardinality(
            self.bm25_weight_min, self.bm25_weight_max, self.bm25_weight_step
        )


class QueryDecomposition(BaseModel, SearchSpaceMixin):
    llm_names: T.List[str] = Field(
        default_factory=lambda: NON_REASONING_LLMS,
        description="List of LLM names to be used for query decomposition.",
    )
    num_queries_min: int = Field(
        default=2, description="Minimum number of sub-queries to generate."
    )
    num_queries_max: int = Field(
        default=20, description="Maximum number of sub-queries to generate."
    )
    num_queries_step: int = Field(
        default=2, description="Step size for the number of sub-queries."
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}query_decomposition_enabled": False,
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}query_decomposition_llm_name": CategoricalDistribution(
                self.llm_names
            ),
            f"{prefix}query_decomposition_num_queries": IntDistribution(
                self.num_queries_min,
                self.num_queries_max,
                step=self.num_queries_step,
            ),
        }

    def get_cardinality(self) -> int:
        return len(self.llm_names) * get_dist_cardinality(
            self.num_queries_min, self.num_queries_max, self.num_queries_step
        )


class FusionMode(BaseModel, SearchSpaceMixin):
    fusion_modes: T.List[str] = Field(
        default_factory=lambda: [
            "simple",
            "reciprocal_rerank",
            "relative_score",
            "dist_based_score",
        ],
        description="List of available fusion modes for combining results from multiple retrievers or queries.",
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {f"{prefix}fusion_mode": "simple"}

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}fusion_mode": CategoricalDistribution(self.fusion_modes),
        }

    def get_cardinality(self) -> int:
        return len(self.fusion_modes)


class Retriever(BaseModel, SearchSpaceMixin):
    top_k: TopK = Field(
        default_factory=TopK,
        description="Configuration for the number of items to retrieve.",
    )
    methods: T.List[str] = Field(
        default_factory=lambda: [
            "dense",
            "sparse",
            "hybrid",
        ],
        description="List of supported retrieval methods: dense (based on embedding models), sparse (BM25) or hybrid.",
    )
    embedding_models: T.List[str] = Field(
        default_factory=lambda: DEFAULT_EMBEDDING_MODELS,
        description="List of embedding models for dense retrieval.",
    )
    hybrid: Hybrid = Field(
        default_factory=Hybrid, description="Configuration for hybrid retrieval."
    )
    query_decomposition_enabled: list[bool] = Field(
        default_factory=lambda: [True, False],
        description="Whether query decomposition is enabled.",
    )
    query_decomposition: QueryDecomposition = Field(
        default_factory=QueryDecomposition,
        description="Configuration for query decomposition.",
    )
    fusion: FusionMode = Field(
        default_factory=FusionMode, description="Configuration for fusing results."
    )

    def defaults(self, prefix: str = "rag_") -> ParamDict:
        params = {
            f"{prefix}method": "dense",
            f"{prefix}embedding_model": self.embedding_models[0],
            **self.top_k.defaults(prefix=prefix),
            **self.query_decomposition.defaults(prefix=prefix),
            **self.hybrid.defaults(prefix=prefix),
            **self.fusion.defaults(prefix=prefix),
        }
        return T.cast(ParamDict, params)

    def build_distributions(
        self, prefix: str = "rag_"
    ) -> T.Dict[str, BaseDistribution]:
        distributions = {
            f"{prefix}method": CategoricalDistribution(self.methods),
            f"{prefix}query_decomposition_enabled": CategoricalDistribution(
                self.query_decomposition_enabled
            ),
            **self.top_k.build_distributions(prefix=prefix),
        }
        if "dense" in self.methods:
            distributions[f"{prefix}embedding_model"] = CategoricalDistribution(
                self.embedding_models
            )
        if "hybrid" in self.methods:
            distributions.update(**self.hybrid.build_distributions(prefix=prefix))

        if True in self.query_decomposition_enabled:
            distributions.update(
                **self.query_decomposition.build_distributions(prefix=prefix)
            )

        if "hybrid" in self.methods or True in self.query_decomposition_enabled:
            distributions.update(**self.fusion.build_distributions(prefix=prefix))

        return distributions

    def sample(self, trial: Trial, prefix: str = "rag_") -> ParamDict:
        method = f"{prefix}method"
        embedding_model = f"{prefix}embedding_model"
        use_query_decomp = f"{prefix}query_decomposition_enabled"

        params = {
            method: trial.suggest_categorical(method, self.methods),
            use_query_decomp: trial.suggest_categorical(
                use_query_decomp, self.query_decomposition_enabled
            ),
            **self.top_k.sample(trial, prefix=prefix),
        }

        if params[method] in ["dense", "hybrid"]:
            params[embedding_model] = trial.suggest_categorical(
                embedding_model, self.embedding_models
            )
        if params[method] == "hybrid":
            params.update(**self.hybrid.sample(trial, prefix=prefix))

        if params[use_query_decomp]:
            params.update(**self.query_decomposition.sample(trial, prefix=prefix))

        if params[method] == "hybrid" or params[use_query_decomp]:
            params.update(**self.fusion.sample(trial, prefix=prefix))

        return T.cast(ParamDict, params)

    def get_cardinality(self) -> int:
        card = (
            self.top_k.get_cardinality()
            * len(self.methods)
            * len(self.query_decomposition_enabled)
        )
        if "dense" in self.methods:
            card *= len(self.embedding_models)
        if "hybrid" in self.methods:
            card *= self.hybrid.get_cardinality()
        if True in self.query_decomposition_enabled:
            card *= self.query_decomposition.get_cardinality()
        if "hybrid" in self.methods or True in self.query_decomposition_enabled:
            card *= self.fusion.get_cardinality()
        return card


class FewShotRetriever(BaseModel, SearchSpaceMixin):
    top_k: TopK = Field(
        default_factory=TopK,
        description="Configuration for the number of few-shot examples to retrieve.",
    )
    embedding_models: T.List[str] = Field(
        default_factory=lambda: DEFAULT_EMBEDDING_MODELS,
        description="List of embedding models for few-shot example retrieval.",
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            **self.top_k.defaults(prefix=f"{prefix}few_shot_"),
            "{prefix}few_shot_embedding_model": self.embedding_models[0],
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            **self.top_k.build_distributions(prefix=f"{prefix}few_shot_"),
            f"{prefix}few_shot_embedding_model": CategoricalDistribution(
                self.embedding_models
            ),
        }

    def get_cardinality(self) -> int:
        return self.top_k.get_cardinality() * len(self.embedding_models)


class Reranker(BaseModel, SearchSpaceMixin):
    """
    Params:
        reranker_llm_name
        reranker_top_k
    """

    top_k: TopK = Field(
        default_factory=lambda: TopK(kmax=128, log=True),
        description="Configuration for the number of items to rerank.",
    )
    llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="List of LLMs to be used for reranking.",
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}reranker_llm_name": self.llms[0],
            **self.top_k.defaults(prefix=f"{prefix}reranker_"),
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}reranker_llm_name": CategoricalDistribution(self.llms),
            **self.top_k.build_distributions(prefix=f"{prefix}reranker_"),
        }

    def get_cardinality(self) -> int:
        return len(self.llms) * self.top_k.get_cardinality()


class Hyde(BaseModel, SearchSpaceMixin):
    llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="List of LLMs to be used for HyDE (Hypothetical Document Embeddings).",
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}hyde_llm_name": self.llms[0],
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}hyde_llm_name": CategoricalDistribution(self.llms),
        }

    def get_cardinality(self) -> int:
        return len(self.llms)


class AdditionalContext(BaseModel, SearchSpaceMixin):
    num_nodes_min: int = Field(
        default=2, description="Minimum number of additional context nodes."
    )
    num_nodes_max: int = Field(
        default=20, description="Maximum number of additional context nodes."
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {f"{prefix}additional_context_num_nodes": self.num_nodes_min}

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}additional_context_num_nodes": IntDistribution(
                self.num_nodes_min,
                self.num_nodes_max,
                log=True,
            )
        }

    def get_cardinality(self) -> int:
        return get_dist_cardinality(self.num_nodes_min, self.num_nodes_max, step=1)


class ReactRAGAgent(BaseModel, SearchSpaceMixin):
    subquestion_engine_llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="LLMs for the sub-question engine inside the agent.",
    )
    subquestion_response_synthesizer_llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="LLMs for synthesizing responses to subquestions.",
    )
    # TODO: use a fixed private parameter with value 10 and remove it from the search space.
    max_iterations_min: int = Field(
        default=10, description="Minimum number of ReAct agent iterations."
    )
    max_iterations_max: int = Field(
        default=11, description="Maximum number of ReAct agent iterations."
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}subquestion_engine_llm": self.subquestion_engine_llms[0],
            f"{prefix}subquestion_response_synthesizer_llm": self.subquestion_response_synthesizer_llms[
                0
            ],
            f"{prefix}max_iterations": 10,
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}subquestion_engine_llm": CategoricalDistribution(
                self.subquestion_engine_llms
            ),
            f"{prefix}subquestion_response_synthesizer_llm": CategoricalDistribution(
                self.subquestion_response_synthesizer_llms
            ),
            f"{prefix}max_iterations": IntDistribution(
                self.max_iterations_min, self.max_iterations_max, step=1
            ),
        }

    def get_cardinality(self) -> int:
        categorical_dists = [
            self.subquestion_engine_llms,
            self.subquestion_response_synthesizer_llms,
        ]
        return math.prod([len(dist) for dist in categorical_dists])


class CritiqueRAGAgent(BaseModel, SearchSpaceMixin):
    subquestion_engine_llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="LLMs for the sub-question engine inside the agent.",
    )
    subquestion_response_synthesizer_llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="LLMs for synthesizing responses to subquestions.",
    )
    critique_agent_llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS, description="LLMs for the critique agent."
    )
    reflection_agent_llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="LLMs for the reflection agent.",
    )
    # TODO: use a fixed private parameter with value 10 and remove it from the search space.
    max_iterations_min: int = Field(
        default=10, description="Minimum number of Critique agent iterations."
    )
    max_iterations_max: int = Field(
        default=11, description="Maximum number of Critique agent iterations."
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}subquestion_engine_llm": self.subquestion_engine_llms[0],
            f"{prefix}critique_agent_llm": self.critique_agent_llms[0],
            f"{prefix}reflection_agent_llm": self.reflection_agent_llms[0],
            f"{prefix}subquestion_response_synthesizer_llm": self.subquestion_response_synthesizer_llms[
                0
            ],
            f"{prefix}max_iterations": 10,
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}subquestion_engine_llm": CategoricalDistribution(
                self.subquestion_engine_llms
            ),
            f"{prefix}critique_agent_llm": CategoricalDistribution(
                self.critique_agent_llms
            ),
            f"{prefix}reflection_agent_llm": CategoricalDistribution(
                self.reflection_agent_llms
            ),
            f"{prefix}subquestion_response_synthesizer_llm": CategoricalDistribution(
                self.subquestion_response_synthesizer_llms
            ),
            f"{prefix}max_iterations": IntDistribution(
                self.max_iterations_min, self.max_iterations_max, step=1
            ),
        }

    def get_cardinality(self) -> int:
        categorical_dists = [
            self.subquestion_engine_llms,
            self.critique_agent_llms,
            self.reflection_agent_llms,
            self.subquestion_response_synthesizer_llms,
        ]
        return math.prod([len(dist) for dist in categorical_dists]) * (
            self.max_iterations_max - self.max_iterations_min + 1
        )


class SubQuestionRAGAgent(BaseModel, SearchSpaceMixin):
    subquestion_engine_llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="LLMs for the sub-question engine.",
    )
    subquestion_response_synthesizer_llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="LLMs for synthesizing responses to subquestions.",
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}subquestion_engine_llm": self.subquestion_engine_llms[0],
            f"{prefix}subquestion_response_synthesizer_llm": self.subquestion_response_synthesizer_llms[
                0
            ],
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}subquestion_engine_llm": CategoricalDistribution(
                self.subquestion_engine_llms
            ),
            f"{prefix}subquestion_response_synthesizer_llm": CategoricalDistribution(
                self.subquestion_response_synthesizer_llms
            ),
        }

    def get_cardinality(self) -> int:
        categorical_dists = [
            self.subquestion_engine_llms,
            self.subquestion_response_synthesizer_llms,
        ]
        return math.prod([len(dist) for dist in categorical_dists])


class LATSRagAgent(BaseModel, SearchSpaceMixin):
    num_expansions_min: int = Field(
        default=2,
        description="Minimum number of expansions in LATS (Language Agent Tree Search).",
    )
    num_expansions_max: int = Field(
        default=3, description="Maximum number of expansions in LATS."
    )
    num_expansions_step: int = Field(
        default=1, description="Step size for the number of expansions."
    )
    max_rollouts_min: int = Field(
        default=2, description="Minimum number of rollouts in LATS."
    )
    max_rollouts_max: int = Field(
        default=5, description="Maximum number of rollouts in LATS."
    )
    max_rollouts_step: int = Field(
        default=1, description="Step size for the number of rollouts."
    )

    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
        return {
            f"{prefix}lats_num_expansions": 2,
            f"{prefix}lats_max_rollouts": 2,
        }

    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
        return {
            f"{prefix}lats_num_expansions": IntDistribution(
                self.num_expansions_min,
                self.num_expansions_max,
                step=self.num_expansions_step,
            ),
            f"{prefix}lats_max_rollouts": IntDistribution(
                self.max_rollouts_min,
                self.max_rollouts_max,
                step=self.max_rollouts_step,
            ),
        }

    def get_cardinality(self) -> int:
        card = get_dist_cardinality(
            self.num_expansions_min, self.num_expansions_max, self.num_expansions_step
        )
        card *= get_dist_cardinality(
            self.max_rollouts_min, self.max_rollouts_max, self.max_rollouts_step
        )
        return card


class SearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Forbids unknown fields
    non_search_space_params: T.List[str] = Field(
        default_factory=lambda: [
            "enforce_full_evaluation",
            "retrievers",
        ],
        description="Parameters not part of the hyperparameter search space.",
    )
    rag_modes: T.List[str] = Field(
        default_factory=lambda: RAG_MODES,
        description="List of available RAG (Retrieval Augmented Generation) modes.",
    )
    template_names: T.List[str] = Field(
        default_factory=lambda: TEMPLATE_NAMES,
        description="List of available prompt template names.",
    )
    response_synthesizer_llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="List of LLMs for response synthesis.",
    )
    few_shot_enabled: T.List[bool] = Field(
        default_factory=lambda: [True, False],
        description="Whether few-shot learning is enabled.",
    )
    few_shot_retriever: FewShotRetriever = Field(
        default_factory=FewShotRetriever,
        description="Configuration for the few-shot retriever.",
    )
    rag_retriever: Retriever = Field(
        default_factory=lambda: Retriever(top_k=TopK(kmax=128, log=True)),
        description="Configuration for the RAG retriever.",
    )
    splitter: Splitter = Field(
        default_factory=Splitter, description="Configuration for the text splitter."
    )
    reranker_enabled: T.List[bool] = Field(
        default_factory=lambda: [True, False],
        description="Whether reranking is enabled.",
    )
    reranker: Reranker = Field(
        default_factory=Reranker, description="Configuration for the reranker."
    )
    hyde_enabled: T.List[bool] = Field(
        default_factory=lambda: [True, False], description="Whether HyDE is enabled."
    )
    hyde: Hyde = Field(default_factory=Hyde, description="Configuration for HyDE.")
    additional_context_enabled: T.List[bool] = Field(
        default_factory=lambda: [True, False],
        description="Whether additional context is enabled.",
    )
    additional_context: AdditionalContext = Field(
        default_factory=AdditionalContext,
        description="Configuration for additional context.",
    )
    react_rag_agent: ReactRAGAgent = Field(
        default_factory=ReactRAGAgent,
        description="Configuration for the ReAct RAG agent.",
    )
    critique_rag_agent: CritiqueRAGAgent = Field(
        default_factory=CritiqueRAGAgent,
        description="Configuration for the Critique RAG agent.",
    )
    sub_question_rag: SubQuestionRAGAgent = Field(
        default_factory=SubQuestionRAGAgent,
        description="Configuration for the Sub-question RAG agent.",
    )
    lats_rag_agent: LATSRagAgent = Field(
        default_factory=LATSRagAgent,
        description="Configuration for the LATS RAG agent.",
    )
    _custom_defaults: ParamDict = {}

    def _defaults(self) -> ParamDict:
        return {
            "rag_mode": self.rag_modes[0],
            "template_name": self.template_names[0],
            "response_synthesizer_llm": self.response_synthesizer_llms[0],
            "few_shot_enabled": False,
            "additional_context_enabled": False,
            "hyde_enabled": False,
            "reranker_enabled": False,
            **self.rag_retriever.defaults(),
            **self.splitter.defaults(),
            **self.react_rag_agent.defaults(),
            **self.critique_rag_agent.defaults(),
            **self.sub_question_rag.defaults(),
            **self.lats_rag_agent.defaults(),
        }

    def update_defaults(self, defaults: ParamDict) -> None:
        self._custom_defaults.update(defaults)

    def defaults(self) -> ParamDict:
        return {
            **self._defaults(),
            **self._custom_defaults,
        }

    def param_names(
        self, params: T.Dict[str, T.Any] | T.List[str] | None = None
    ) -> T.List[str]:
        return list(self.build_distributions(params=params).keys())

    def build_distributions(
        self, params: T.Dict[str, T.Any] | T.List[str] | None = None
    ) -> T.Dict[str, BaseDistribution]:
        distributions: dict[str, BaseDistribution] = {
            "rag_mode": CategoricalDistribution(self.rag_modes),
            "template_name": CategoricalDistribution(self.template_names),
            "response_synthesizer_llm": CategoricalDistribution(
                self.response_synthesizer_llms
            ),
            "few_shot_enabled": CategoricalDistribution(self.few_shot_enabled),
            "hyde_enabled": CategoricalDistribution(self.hyde_enabled),
            "reranker_enabled": CategoricalDistribution(self.reranker_enabled),
            "additional_context_enabled": CategoricalDistribution(
                self.additional_context_enabled
            ),
        }
        if True in self.few_shot_enabled:
            distributions.update(self.few_shot_retriever.build_distributions())
        if True in self.hyde_enabled:
            distributions.update(self.hyde.build_distributions())
        distributions.update(self.rag_retriever.build_distributions())
        distributions.update(self.splitter.build_distributions())
        if True in self.reranker_enabled:
            distributions.update(self.reranker.build_distributions())
        # some agents redefine parameters
        if True in self.additional_context_enabled:
            distributions.update(self.additional_context.build_distributions())
        distributions.update(self.react_rag_agent.build_distributions())
        distributions.update(self.critique_rag_agent.build_distributions())
        distributions.update(self.sub_question_rag.build_distributions())
        distributions.update(self.lats_rag_agent.build_distributions())

        if params is not None:
            reduced_distributions = {
                key: val for key, val in distributions.items() if key in params
            }
            return reduced_distributions

        return distributions

    def sample(self, trial: Trial, parameters: T.List[str] = PARAMETERS) -> ParamDict:
        for param in parameters:
            assert param in PARAMETERS, f"Invalid parameter: {param}"

        params: ParamDict = {
            "few_shot_enabled": False,
        }
        defaults = self.defaults()

        if "rag_mode" in parameters:
            params["rag_mode"] = trial.suggest_categorical("rag_mode", self.rag_modes)
        else:
            params["rag_mode"] = defaults["rag_mode"]

        if "template_name" in parameters:
            params["template_name"] = trial.suggest_categorical(
                "template_name", self.template_names
            )
        else:
            params["template_name"] = defaults["template_name"]

        if "response_synthesizer_llm" in parameters:
            params["response_synthesizer_llm"] = trial.suggest_categorical(
                "response_synthesizer_llm", self.response_synthesizer_llms
            )
        else:
            params["response_synthesizer_llm"] = defaults["response_synthesizer_llm"]

        # No-RAG general parameters
        if params["rag_mode"] != "no_rag":
            if "rag_retriever" in parameters:
                params.update(**self.rag_retriever.sample(trial))
            else:
                params.update(**self.rag_retriever.defaults())

            if "splitter" in parameters:
                params.update(**self.splitter.sample(trial))
            else:
                params.update(**self.splitter.defaults())

            if "reranker" in parameters:
                params["reranker_enabled"] = trial.suggest_categorical(
                    "reranker_enabled", self.reranker_enabled
                )
                if params["reranker_enabled"]:
                    params.update(**self.reranker.sample(trial))
            else:
                params["reranker_enabled"] = False

            if "hyde" in parameters:
                params["hyde_enabled"] = trial.suggest_categorical(
                    "hyde_enabled", self.hyde_enabled
                )
                if params["hyde_enabled"]:
                    params.update(**self.hyde.sample(trial))
            else:
                params["hyde_enabled"] = False

            if "additional_context" in parameters:
                params["additional_context_enabled"] = trial.suggest_categorical(
                    "additional_context_enabled", self.additional_context_enabled
                )
                if params["additional_context_enabled"]:
                    params.update(**self.additional_context.sample(trial))
            else:
                params["additional_context_enabled"] = False

        if params["rag_mode"] == "react_rag_agent":
            if "react_rag_agent" in parameters:
                params.update(**self.react_rag_agent.sample(trial))
            else:
                params.update(**self.react_rag_agent.defaults())
        elif params["rag_mode"] == "critique_rag_agent":
            if "critique_rag_agent" in parameters:
                params.update(**self.critique_rag_agent.sample(trial))
            else:
                params.update(**self.critique_rag_agent.defaults())
        elif params["rag_mode"] == "sub_question_rag":
            if "sub_question_rag" in parameters:
                params.update(**self.sub_question_rag.sample(trial))
            else:
                params.update(**self.sub_question_rag.defaults())
        elif params["rag_mode"] == "lats_rag_agent":
            if "lats_rag_agent" in parameters:
                params.update(**self.lats_rag_agent.sample(trial))
            else:
                params.update(**self.lats_rag_agent.defaults())
            params.update(**self.lats_rag_agent.sample(trial))

        if few_shot_enabled := trial.suggest_categorical(
            "few_shot_enabled", self.few_shot_enabled
        ):
            params["few_shot_enabled"] = few_shot_enabled
            params.update(**self.few_shot_retriever.sample(trial))

        return params

    def get_cardinality(self) -> int:
        card = 0
        for rag_mode in self.rag_modes:
            sub_card = (
                len(self.template_names)
                * len(self.response_synthesizer_llms)
                * len(self.few_shot_enabled)
                * len(self.hyde_enabled)
                * len(self.additional_context_enabled)
                * len(self.reranker_enabled)
            )
            if rag_mode != "no_rag":
                sub_card *= self.rag_retriever.get_cardinality()
                sub_card *= self.splitter.get_cardinality()
                if True in self.reranker_enabled:
                    sub_card *= self.reranker.get_cardinality()
                if True in self.hyde_enabled:
                    sub_card *= self.hyde.get_cardinality()
                if True in self.additional_context_enabled:
                    sub_card *= self.additional_context.get_cardinality()

            if rag_mode == "react_rag_agent":
                sub_card *= self.react_rag_agent.get_cardinality()
            elif rag_mode == "critique_rag_agent":
                sub_card *= self.critique_rag_agent.get_cardinality()
            elif rag_mode == "sub_question_rag":
                sub_card *= self.sub_question_rag.get_cardinality()
            elif rag_mode == "lats_rag_agent":
                sub_card *= self.lats_rag_agent.get_cardinality()

            if True in self.few_shot_enabled:
                sub_card *= self.few_shot_retriever.get_cardinality()
            card += sub_card

        return card

    def is_few_shot(self, params: T.Dict) -> bool:
        return params.get("few_shot_enabled", False)


class RetrieverSearchSpace(BaseModel):
    """Search space over retrievers."""

    model_config = ConfigDict(extra="forbid")  # Forbids unknown fields
    rag_modes: T.List[str] = Field(
        default_factory=lambda: ["rag"],
        description='List of RAG modes, restricted to "rag" for this specific search space.',
    )
    non_search_space_params: T.List[str] = Field(
        default_factory=lambda: ["enforce_full_evaluation"],
        description="Parameters not part of the hyperparameter search space.",
    )
    response_synthesizer_llms: T.List[str] = Field(
        default_factory=lambda: DEFAULT_LLMS,
        description="LLMs used for response synthesis.",
    )
    rag_retriever: Retriever = Field(
        default_factory=lambda: Retriever(top_k=TopK(kmax=128, log=True)),
        description="Configuration for the RAG retriever.",
    )
    splitter: Splitter = Field(
        default_factory=Splitter, description="Configuration for the text splitter."
    )
    hyde_enabled: T.List[bool] = Field(
        default_factory=lambda: [True, False], description="Whether HyDE is enabled."
    )
    hyde: Hyde = Field(default_factory=Hyde, description="Configuration for HyDE.")
    additional_context_enabled: T.List[bool] = Field(
        default_factory=lambda: [True, False],
        description="Whether additional context is enabled.",
    )
    additional_context: AdditionalContext = Field(
        default_factory=AdditionalContext,
        description="Configuration for additional context.",
    )

    def defaults(self) -> ParamDict:
        return {
            "rag_mode": self.rag_modes[0],
            "response_synthesizer_llm": self.response_synthesizer_llms[0],
            "additional_context_enabled": False,
            "hyde_enabled": False,
            **self.rag_retriever.defaults(),
            **self.splitter.defaults(),
        }

    def build_distributions(
        self, params: T.Dict[str, T.Any] | T.List[str] | None = None
    ) -> T.Dict[str, BaseDistribution]:
        distributions: dict[str, BaseDistribution] = {
            "rag_mode": CategoricalDistribution(self.rag_modes),
            "response_synthesizer_llm": CategoricalDistribution(
                self.response_synthesizer_llms
            ),
            "hyde_enabled": CategoricalDistribution(self.hyde_enabled),
            "additional_context_enabled": CategoricalDistribution(
                self.additional_context_enabled
            ),
            **self.rag_retriever.build_distributions(prefix="rag_"),
            **self.splitter.build_distributions(),
        }
        if True in self.hyde_enabled:
            distributions.update(self.hyde.build_distributions())
        if True in self.additional_context_enabled:
            distributions.update(self.additional_context.build_distributions())

        if params is not None:
            reduced_distributions = {
                key: val for key, val in distributions.items() if key in params
            }
            return reduced_distributions

        return distributions

    def sample(self, trial: Trial, prefix: str = "") -> ParamDict:
        params: ParamDict = {
            "rag_mode": trial.suggest_categorical("rag_mode", self.rag_modes),
            "response_synthesizer_llm": trial.suggest_categorical(
                "response_synthesizer_llm", self.response_synthesizer_llms
            ),
            "hyde_enabled": trial.suggest_categorical(
                "hyde_enabled", self.hyde_enabled
            ),
            "additional_context_enabled": trial.suggest_categorical(
                "additional_context_enabled", self.additional_context_enabled
            ),
            **self.rag_retriever.sample(trial, prefix="rag_"),
            **self.splitter.sample(trial),
        }
        if params["hyde_enabled"]:
            params.update(**self.hyde.sample(trial))
        if params["additional_context_enabled"]:
            params.update(**self.additional_context.sample(trial))
        return params

    def get_cardinality(self) -> int:
        return (
            self.rag_retriever.get_cardinality()
            * self.splitter.get_cardinality()
            * self.hyde.get_cardinality()
            * self.additional_context.get_cardinality()
            * len(self.hyde_enabled)
            * len(self.additional_context_enabled)
        )


class AgentSearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Forbids unknown fields
    llms: T.List[str] = [
        "gpt-4o-mini",
    ]
    rag_modes: T.List[str] = [
        "no_rag",
        "rag",
    ]
    prompt_names: T.List[str] = [
        "default",
        "concise",
        "aggressive",
    ]
    embedding_models: T.List[str] = [
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L12-v2",
    ]
    splitter: Splitter = Splitter()


EmbeddingDeviceType = T.Literal["onnx-cpu", "cpu", "mps", "cuda", None]


class Block(BaseModel):
    name: str = Field(default="global", description="Block name")
    num_trials: int = Field(default=1000, description="Number of trials.")
    components: T.List[str] = Field(
        default_factory=lambda: PARAMETERS, description="Block components"
    )


class OptimizationConfig(BaseModel):
    method: T.Literal["expanding", "knee"] = Field(
        default="expanding",
        description="Method for optimization, e.g., expanding window or knee point detection.",
    )
    blocks: T.List[Block] = Field(
        default_factory=lambda: [Block()], description="List of optimization blocks."
    )
    shuffle_blocks: bool = Field(
        default=False,
        description="Whether to shuffle the order of optimization blocks.",
    )
    num_trials: int = Field(
        default=1000, description="Total number of optimization trials."
    )
    model_config = ConfigDict(extra="forbid")  # Forbids unknown fields
    baselines: T.List[T.Dict[str, T.Any]] = Field(
        default_factory=list,
        description="List of baseline configurations to compare against.",
    )
    shuffle_baselines: bool = Field(
        default=True, description="Whether to shuffle the order of baselines."
    )
    cpus_per_trial: int = Field(
        default=2, description="Number of CPUs allocated per trial."
    )
    gpus_per_trial: int | float = Field(
        default=0.0, description="Number of GPUs allocated per trial."
    )
    embedding_device: EmbeddingDeviceType = Field(
        default=None,
        description="Device to use for embeddings (e.g., 'cpu', 'cuda', 'onnx-cpu'). Use `None` to auto-detect.",
    )
    use_hf_embedding_models: bool = Field(
        default=False, description="Whether to use HuggingFace embedding models."
    )
    raise_on_failed_trial: bool = Field(
        default=False, description="Whether to raise an exception if a trial fails."
    )
    max_concurrent_trials: int = Field(
        default=10, description="Maximum number of trials to run concurrently."
    )
    num_eval_samples: int = Field(
        default=500, description="Number of samples to use for evaluation."
    )
    num_eval_batch: int = Field(default=5, description="Batch size for evaluation.")
    max_eval_failure_rate: float = Field(
        default=0.5, description="Maximum allowed failure rate during evaluation."
    )
    max_trial_cost: float = Field(
        default=10.00, description="Maximum allowed cost per trial."
    )
    num_random_trials: int = Field(
        default=100, description="Number of random trials to run initially."
    )
    num_retries_unique_params: int = Field(
        default=100,
        description="Number of retries to find unique parameters for a trial.",
    )
    num_prompt_optimization_batch: int = Field(
        default=50, description="Batch size for prompt optimization."
    )
    rate_limiter_max_coros: int = Field(
        default=3, description="Maximum number of coroutines for the rate limiter."
    )
    rate_limiter_period: int = Field(
        default=10, description="Period in seconds for the rate limiter."
    )
    skip_existing: bool = Field(
        default=True, description="Whether to skip trials with existing results."
    )
    num_warmup_steps_timeout: int = Field(
        default=3, description="Number of warmup steps for timeout pruner."
    )
    num_warmup_steps_costout: int = Field(
        default=2, description="Number of warmup steps for cost pruner."
    )
    num_warmup_steps_pareto: int = Field(
        default=30, description="Number of warmup steps for Pareto pruner."
    )
    use_pareto_pruner: bool = Field(
        default=True, description="Whether to use the Pareto pruner."
    )
    use_cost_pruner: bool = Field(
        default=True, description="Whether to use the cost pruner."
    )
    use_runtime_pruner: bool = Field(
        default=True, description="Whether to use the runtime pruner."
    )
    pareto_pruner_success_rate: float = Field(
        default=0.9, description="Success rate threshold for Pareto pruner."
    )
    pareto_eval_success_rate: float = Field(
        default=0.9, description="Success rate threshold for Pareto evaluation."
    )
    raise_on_invalid_baseline: bool = Field(
        default=False,
        description="Whether to raise an exception for invalid baselines.",
    )
    baselines_cycle_llms: bool = Field(
        default=False, description="Whether to cycle through LLMs for baselines."
    )
    use_toy_baselines: bool = Field(
        default=False, description="Whether to use toy baselines."
    )
    use_individual_baselines: bool = Field(
        default=True, description="Whether to use individual component baselines."
    )
    use_agent_baselines: bool = Field(
        default=True, description="Whether to use agent-specific baselines."
    )
    use_variations_of_baselines: bool = Field(
        default=True, description="Whether to use variations of baselines."
    )
    use_pareto_baselines: bool = Field(
        default=False,
        description="Whether to use baselines from the Pareto front, switch to True for transfer learning",
    )
    objective_1_name: T.Literal["accuracy", "retriever_recall"] = Field(
        default="accuracy", description="Name of the first optimization objective."
    )
    objective_2_name: T.Literal[
        "p80_time", "llm_cost_mean", "retriever_context_length"
    ] = Field(
        default="llm_cost_mean",
        description="Name of the second optimization objective.",
    )
    obj1_zscore: float = Field(
        default=1.645,
        description="Z-score for the first objective (e.g., for confidence interval).",
    )
    obj2_zscore: float = Field(
        default=1.645, description="Z-score for the second objective."
    )
    sampler: T.Literal["tpe", "hierarchical"] = Field(
        default="tpe",
        description='Type of sampler to use (e.g., "tpe", "hierarchical").',
    )
    ############################
    # seeder_timeout settings
    # --------------------------
    # 1 hour: 3600
    # 1 day: 86400
    # no wait: 0
    # wait until finished: None
    # -------------------------
    # main optimization starts in parallel after timeout
    # while seeding continues
    seeder_timeout: float | None = Field(
        default=3600,
        description="Timeout in seconds for the seeder process. None means wait indefinitely.",
    )


class TransferLearningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Forbids unknown fields
    studies: T.List[str] = Field(
        default_factory=list,
        description="List of study names to use for transfer learning.",
    )
    max_fronts: int = Field(
        default=2,
        description="Maximum number of Pareto fronts to consider from previous studies.",
    )
    max_total: int = Field(
        default=100, description="Maximum total number of configurations to transfer."
    )
    success_rate: float = Field(
        default=0.9, description="Minimum success rate for transferred configurations."
    )
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Embedding model used for comparing configurations in transfer learning.",
    )


class TimeoutConfig(BaseModel):
    embedding_timeout_active: bool = Field(
        default=False, description="Whether embedding timeout is active."
    )
    embedding_max_time: int = Field(
        default=3600 * 4, description="Maximum time in seconds for embeddings."
    )
    embedding_min_chunks_to_process: int = Field(
        default=100,
        description="Minimum number of chunks to process before embedding timeout.",
    )
    embedding_min_time_to_process: int = Field(
        default=120,
        description="Minimum time in seconds to process before embedding timeout.",
    )
    eval_timeout: int = Field(
        default=3600 * 10,
        description="Maximum time in seconds for the entire evaluation process.",
    )
    single_eval_timeout: int = Field(
        default=3600 * 2,
        description="Maximum time in seconds for a single evaluation run.",
    )
    onnx_timeout: int = Field(
        default=600, description="Maximum time in seconds for ONNX model operations."
    )


class Evaluation(BaseModel):
    mode: T.Literal["single", "random", "consensus", "retriever"] = Field(
        default="single", description="Evaluation mode."
    )
    llms: T.List[str] = Field(
        default_factory=lambda: ["gpt-4o-mini"],
        description="List of LLMs to use for evaluation. If 'single' mode is chosen, the first list item will be used.",
    )
    raise_on_exception: bool = Field(
        default=False,
        description="Whether to raise an exception if an error occurs during evaluation.",
    )
    use_tracing_metrics: bool = Field(
        default=False, description="Whether to use tracing metrics during evaluation."
    )
    min_reporting_success_rate: float = Field(
        default=0.5,
        description="Minimum success rate for reporting evaluation results.",
    )


class ParetoConfig(BaseModel):
    """
    Parameters that are used to override the study config for the Pareto front evaluation,
    for instance, `optimization__skip_existing` is used to override `optimization.skip_existing`.
    """

    name: str = Field(description="Name of the Pareto configuration/study.")
    raise_on_same_study: bool = Field(
        default=True,
        description="Whether to raise an error if the Pareto study name is the same as the main study.",
    )
    reuse_study: bool = Field(
        default=False, description="Whether to reuse an existing Pareto study."
    )
    optimization__skip_existing: bool = Field(
        default=True,
        description="Override for optimization.skip_existing for Pareto evaluation, switch to false when using same study.",
    )
    optimization__use_pareto_pruner: bool = Field(
        default=False,
        description="Override for optimization.use_pareto_pruner for Pareto evaluation.",
    )
    optimization__use_cost_pruner: bool = Field(
        default=False,
        description="Override for optimization.use_cost_pruner for Pareto evaluation.",
    )
    optimization__use_runtime_pruner: bool = Field(
        default=False,
        description="Override for optimization.use_runtime_pruner for Pareto evaluation.",
    )
    optimization__num_eval_samples: int = Field(
        description="Override for optimization.num_eval_samples for Pareto evaluation."
    )  # No default
    replacement_llm_name: str = Field(
        default="",
        description="LLM name to replace in configurations for Pareto evaluation (e.g., to test a new LLM on existing good configurations).",
    )
    dataset__partition_map: PartitionMap = Field(
        default_factory=lambda: PartitionMap(
            sample="sample",
            train="test",
            test="holdout",
            holdout="holdout",
        ),
        description="Override for dataset partition mapping for Pareto evaluation.",
    )


class StudyConfig(BaseSettings):
    name: str = Field(description="Name of the Optuna study.")
    dataset: T.Annotated[  # type: ignore
        T.Union[
            *SyftrQADataset.__subclasses__(),  # type: ignore
            *HotPotQAHF.__subclasses__(),  # type: ignore
            *FinanceBenchHF.__subclasses__(),  # type: ignore
            *CragTask3HF.__subclasses__(),  # type: ignore
            *DRDocsHF.__subclasses__(),  # type: ignore
        ],
        Field(discriminator="xname"),
    ] = Field(description="Dataset configuration.")
    evaluation: Evaluation = Field(
        default_factory=Evaluation, description="LLM-as-a-judge configuration."
    )
    reuse_study: bool = Field(
        default=True, description="Whether to reuse an existing study."
    )
    recreate_study: bool = Field(
        default=True,
        description="Whether to recreate the study if it already exists (potentially deleting old data).",
    )
    search_space: SearchSpace = Field(
        default_factory=SearchSpace,
        description="Search space configuration for the optimization.",
    )
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optimization process configuration.",
    )
    pareto: T.Optional[ParetoConfig] = Field(
        default=None, description="Optional configuration for Pareto front evaluation."
    )
    transfer_learning: TransferLearningConfig = Field(
        default_factory=TransferLearningConfig,
        description="Transfer learning configuration.",
    )
    timeouts: TimeoutConfig = Field(
        default_factory=TimeoutConfig,
        description="Timeout configurations for various stages.",
    )
    toy_mode: bool = Field(
        default=False, description="Whether to run in toy mode (with smaller dataset)."
    )

    model_config = SettingsConfigDict(
        extra="forbid",  # Forbids unknown fields
        yaml_file=cfg.study_config_file or Path("Idontexist"),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: T.Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> T.Tuple[PydanticBaseSettingsSource, ...]:
        """Study config can be loaded from a yaml file.

        Use SYFTR_STUDY_CONFIG_FILE env var or
        'study_config_file: <path> in the top-level of config.yaml
        to choose a study config file, or use the from_file factory method.

        Parameters passed to StudyConfig.__init__ will take precedence
        over the yaml file.
        """
        if cfg.study_config_file and not cfg.study_config_file.exists():
            raise ValueError(
                f"Study configuration file cannot be found at {cfg.study_config_file.resolve()}"
            )

        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    @classmethod
    def from_file(cls, path: Path | str, *args, **kwargs) -> "StudyConfig":
        """Use from_file to load from a given config file path.

        *args and **kwargs are the same as the StudyConfig constructor
        and take precedence over values loaded from the config file.

        cfg.study_config_file is ignored when this method is used.
        """
        if not Path(path).exists():
            raise ValueError(
                f"Study configuration file cannot be found at {Path(path).resolve()}"
            )

        klass = deepcopy(cls)
        _orig = klass.model_config.pop("yaml_file", None)
        klass.model_config = SettingsConfigDict(**cls.model_config, yaml_file=path)
        instance = klass(*args, **kwargs)
        klass.model_config["yaml_file"] = _orig
        return instance

    def replace_llm_name(self, params: T.Dict[str, T.Any]):
        """
        Replace the LLM name in the params with the replacement_llm_name.
        With this functionality, we can easily run historical flows with a different LLM.
        """
        assert self.pareto, "No Pareto config is set"
        assert self.pareto.replacement_llm_name, "No replacement LLM name is set"

        replacement_llm_name = self.pareto.replacement_llm_name
        params["response_synthesizer_llm"] = replacement_llm_name

    @property
    def is_retriever_study(self) -> bool:
        return isinstance(self.search_space, RetrieverSearchSpace)


class AgentStudyConfig(BaseSettings):
    name: str = Field(description="Name of the agent study.")
    datasets: T.List[  # type: ignore
        T.Annotated[
            T.Union[*SyftrQADataset.__subclasses__()],
            Field(discriminator="xname"),
        ]
    ] = Field(
        default_factory=list,
        description="List of dataset configurations for the agent study.",
    )
    evaluation: Evaluation = Field(
        default_factory=Evaluation, description="LLM-as-a-judge configuration."
    )
    reuse_study: bool = Field(
        default=True, description="Whether to reuse an existing agent study."
    )
    search_space: AgentSearchSpace = Field(
        default_factory=AgentSearchSpace,
        description="Search space configuration specific to agents.",
    )
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optimization process configuration.",
    )
    transfer_learning: TransferLearningConfig = Field(
        default_factory=TransferLearningConfig,
        description="Transfer learning configuration.",
    )
    timeouts: TimeoutConfig = Field(
        default_factory=TimeoutConfig, description="Timeout configurations."
    )
    toy_mode: bool = Field(default=False, description="Whether to run in toy mode.")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: T.Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> T.Tuple[PydanticBaseSettingsSource, ...]:
        if cfg.study_config_file and not cfg.study_config_file.exists():
            raise ValueError(
                f"Study configuration file cannot be found at {cfg.study_config_file.resolve()}"
            )

        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    @classmethod
    def from_file(cls, path: Path | str, *args, **kwargs) -> "AgentStudyConfig":
        if not Path(path).exists():
            raise ValueError(
                f"Study configuration file cannot be found at {Path(path).resolve()}"
            )

        klass = deepcopy(cls)
        _orig = klass.model_config.pop("yaml_file", None)
        klass.model_config = SettingsConfigDict(**cls.model_config, yaml_file=path)
        instance = klass(*args, **kwargs)
        klass.model_config["yaml_file"] = _orig
        return instance

    @property
    def is_retriever_study(self) -> bool:
        return False


class RetrieverStudyConfig(StudyConfig):
    search_space: RetrieverSearchSpace = Field(  # type: ignore
        default_factory=RetrieverSearchSpace,
        description="Search space of retriever study,",
    )


def get_default_study_name():
    if os.path.exists("studies/private.yaml"):
        return "studies/private.yaml"
    return "studies/hotpot-toy.yaml"


def get_pareto_study_config(study_config: StudyConfig) -> StudyConfig:
    assert study_config.pareto is not None, "Pareto config is not set"
    pareto_study_config = study_config.model_copy()
    pareto_study_config.name = study_config.pareto.name
    pareto_study_config.reuse_study = study_config.pareto.reuse_study
    pareto_study_config.optimization.use_pareto_pruner = (
        study_config.pareto.optimization__use_pareto_pruner
    )
    pareto_study_config.optimization.use_cost_pruner = (
        study_config.pareto.optimization__use_cost_pruner
    )
    pareto_study_config.optimization.use_runtime_pruner = (
        study_config.pareto.optimization__use_runtime_pruner
    )
    pareto_study_config.optimization.num_eval_samples = (
        study_config.pareto.optimization__num_eval_samples
    )
    pareto_study_config.dataset.partition_map = (
        study_config.pareto.dataset__partition_map
    )

    # sometimes we want to run only the pareto frontier flows
    # but replace the LLMs with a different one to see if a new LLM
    # can potentially improve results. This is a cheaper way to test a new LLM
    # without running a full fledged search but less effective.
    # We need to make sure the that replacement LLM is in the search space.
    if study_config.pareto.replacement_llm_name:
        pareto_study_config.search_space.response_synthesizer_llms = list(
            set(
                pareto_study_config.search_space.response_synthesizer_llms
                + [study_config.pareto.replacement_llm_name]
            )
        )

    return pareto_study_config


# optimize this using training data
def get_critique_template() -> str:
    return """
Your job is to judge the correctness of the answer to the provided question. 
Write PASS if you think that the answer is correct, otherwise return FAIL.
Do not return anything else. Here's the text: {input_str}
"""


# optimize this using training data
def get_react_template() -> str:
    return """
You are designed to help with answering questions.

## Tools

You have access to a wide variety of tools that can be used to look up relevant information
about provided questions.

You have access to the following tools:
{tool_desc}

## Output Format

Please answer in the same language as the question and use the following format:

```
Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

NEVER surround your response with markdown code markers.
You may use code markers within your response if you need to.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the tool will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format till you have enough information to answer the question without using any more tools.
At that point, you MUST respond in one of the following two formats:

```
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: [your answer here (In the same language as the user's question)]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]
```

## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages.
"""


def get_template_name(params: T.Dict[str, T.Any], prefix="") -> str:
    if prefix + "template_name" in params:
        return params[prefix + "template_name"]
    # backward compatibility
    rag_mode = params[prefix + "rag_mode"]
    match rag_mode:
        case "no_rag":
            return params[prefix + "no_rag_template_name"]
        case "rag":
            return params[prefix + "rag_template_name"]
        case "react_rag_agent":
            return params[prefix + "react_rag_agent_template_name"]
        case "critique_rag_agent":
            return params[prefix + "critique_rag_agent_template_name"]
        case "sub_question_rag":
            return params[prefix + "sub_question_rag_template_name"]
        case "lats_rag_agent":
            return params[prefix + "lats_rag_agent_template_name"]
        case _:
            raise ValueError(f"Invalid RAG mode: {rag_mode}")


def get_response_synthesizer_llm(params: T.Dict[str, T.Any], prefix="") -> str:
    if prefix + "response_synthesizer_llm" in params:
        return params[prefix + "response_synthesizer_llm"]
    # backward compatibility
    rag_mode = params[prefix + "rag_mode"]
    match rag_mode:
        case "no_rag":
            return params[prefix + "no_rag_response_synthesizer_llm"]
        case "rag":
            return params[prefix + "rag_response_synthesizer_llm"]
        case "react_rag_agent":
            return params[prefix + "react_rag_agent_response_synthesizer_llm"]
        case "critique_rag_agent":
            return params[prefix + "critique_rag_agent_response_synthesizer_llm"]
        case "sub_question_rag":
            return params[prefix + "sub_question_rag_response_synthesizer_llm"]
        case "lats_rag_agent":
            return params[prefix + "lats_rag_agent_response_synthesizer_llm"]
        case _:
            raise ValueError(f"Invalid RAG mode: {rag_mode}")


def get_subspace(df_trials: pd.DataFrame, search_space: SearchSpace) -> SearchSpace:
    """
    Given a results dataframe, return a SearchSpace object that is
    spanned by the trials in the dataframe.
    """
    params = {}

    if rag_modes := get_unique_strings(df_trials, "rag_mode"):
        params["rag_modes"] = rag_modes

    if template_names := get_unique_strings(df_trials, "template_name"):
        params["template_names"] = template_names

    if response_synthesizer_llms := get_unique_strings(
        df_trials, "response_synthesizer_llm"
    ):
        params["response_synthesizer_llms"] = response_synthesizer_llms

    if few_shot_enabled := get_unique_bools(df_trials, "few_shot_enabled"):
        params["few_shot_enabled"] = few_shot_enabled  # type: ignore

    if True in few_shot_enabled:
        if embedding_models := get_unique_strings(
            df_trials, "few_shot_embedding_model"
        ):
            params["few_shot_retriever"] = FewShotRetriever(  # type: ignore
                top_k=TopK(
                    kmin=get_min_int(
                        df_trials,
                        "few_shot_top_k",
                        search_space.few_shot_retriever.top_k.kmin,
                    ),
                    kmax=get_max_int(
                        df_trials,
                        "few_shot_top_k",
                        search_space.few_shot_retriever.top_k.kmax,
                    ),
                    log=search_space.few_shot_retriever.top_k.log,
                    step=search_space.few_shot_retriever.top_k.step,
                ),
                embedding_models=embedding_models,
            )

    if rag_methods := get_unique_strings(df_trials, "rag_method"):
        kmin = get_min_int(
            df_trials, "rag_top_k", search_space.rag_retriever.top_k.kmin
        )
        kmax = get_max_int(
            df_trials, "rag_top_k", search_space.rag_retriever.top_k.kmax
        )
        bm25_weight_min = get_min_float(
            df=df_trials,
            col="rag_hybrid_bm25_weight",
            default=search_space.rag_retriever.hybrid.bm25_weight_min,
            ndigits=NDIGITS,
        )
        bm25_weight_max = get_max_float(
            df=df_trials,
            col="rag_hybrid_bm25_weight",
            default=search_space.rag_retriever.hybrid.bm25_weight_max,
            ndigits=NDIGITS,
        )
        retriever_params = {
            "top_k": TopK(
                kmin=kmin, kmax=kmax, log=search_space.rag_retriever.top_k.log
            ),
            "methods": rag_methods,
            "hybrid": Hybrid(
                bm25_weight_min=bm25_weight_min,
                bm25_weight_max=bm25_weight_max,
                bm25_weight_step=search_space.rag_retriever.hybrid.bm25_weight_step,
            ),
        }
        if embedding_models := get_unique_strings(df_trials, "rag_embedding_model"):
            retriever_params["embedding_models"] = embedding_models
        if query_decomposition_enabled := get_unique_bools(
            df_trials, "rag_query_decomposition_enabled"
        ):
            retriever_params["rag_query_decomposition_enabled"] = (
                query_decomposition_enabled
            )
            if True in query_decomposition_enabled:
                retriever_params["rag_query_decomposition"] = QueryDecomposition(
                    llm_names=get_unique_strings(
                        df_trials, "rag_query_decomposition_llm_name"
                    ),
                    num_queries_min=get_min_int(
                        df_trials,
                        "rag_query_decomposition_num_queries",
                        search_space.rag_retriever.query_decomposition.num_queries_min,
                    ),
                    num_queries_max=get_max_int(
                        df_trials,
                        "rag_query_decomposition_num_queries",
                        search_space.rag_retriever.query_decomposition.num_queries_max,
                    ),
                )
            if fusion_modes := get_unique_strings(df_trials, "rag_fusion_mode"):
                retriever_params["fusion"] = FusionMode(
                    fusion_modes=fusion_modes,
                )
        params["rag_retriever"] = Retriever(**retriever_params)  # type: ignore

    if splitter_methods := get_unique_strings(df_trials, "splitter_method"):
        params["splitter"] = Splitter(  # type: ignore
            chunk_min_exp=get_min_int(
                df_trials, "splitter_chunk_min_exp", search_space.splitter.chunk_min_exp
            ),
            chunk_max_exp=get_max_int(
                df_trials, "splitter_chunk_max_exp", search_space.splitter.chunk_max_exp
            ),
            chunk_overlap_frac_min=get_min_float(
                df=df_trials,
                col="splitter_chunk_overlap_frac",
                default=search_space.splitter.chunk_overlap_frac_min,
                ndigits=NDIGITS,
            ),
            chunk_overlap_frac_max=get_max_float(
                df=df_trials,
                col="splitter_chunk_overlap_frac",
                default=search_space.splitter.chunk_overlap_frac_max,
                ndigits=NDIGITS,
            ),
            chunk_overlap_frac_step=search_space.splitter.chunk_overlap_frac_step,
            methods=splitter_methods,
        )

    if reranker_enabled := get_unique_bools(df_trials, "reranker_enabled"):
        params["reranker_enabled"] = reranker_enabled  # type: ignore
        if True in reranker_enabled:
            if reranker_llm_name := get_unique_strings(df_trials, "reranker_llm_name"):
                params["reranker"] = Reranker(  # type: ignore
                    top_k=TopK(
                        kmax=get_max_int(
                            df_trials,
                            "reranker_top_k",
                            search_space.reranker.top_k.kmax,
                        ),
                        kmin=get_min_int(
                            df_trials,
                            "reranker_top_k",
                            search_space.reranker.top_k.kmin,
                        ),
                        log=search_space.reranker.top_k.log,
                        step=search_space.reranker.top_k.step,
                    ),
                    llms=reranker_llm_name,
                )

    if hyde_enabled := get_unique_bools(df_trials, "hyde_enabled"):
        params["hyde_enabled"] = hyde_enabled  # type: ignore
        if True in hyde_enabled:
            if hyde_llm_name := get_unique_strings(df_trials, "hyde_llm_name"):
                params["hyde"] = Hyde(  # type: ignore
                    llms=hyde_llm_name,
                )

    if additional_context_enabled := get_unique_bools(
        df_trials, "additional_context_enabled"
    ):
        params["additional_context_enabled"] = additional_context_enabled  # type: ignore
        if True in additional_context_enabled:
            params["additional_context"] = AdditionalContext(  # type: ignore
                num_nodes_min=get_min_int(
                    df_trials,
                    "additional_context_num_nodes",
                    search_space.additional_context.num_nodes_min,
                ),
                num_nodes_max=get_max_int(
                    df_trials,
                    "additional_context_num_nodes",
                    search_space.additional_context.num_nodes_max,
                ),
            )

    if "react_rag_agent" in rag_modes:
        params["react_rag_agent"] = ReactRAGAgent(  # type: ignore
            subquestion_engine_llms=get_unique_strings(
                df_trials, "subquestion_engine_llm"
            ),
            subquestion_response_synthesizer_llms=get_unique_strings(
                df_trials, "subquestion_response_synthesizer_llm"
            ),
            max_iterations_min=get_min_int(
                df_trials,
                "max_iterations",
                search_space.react_rag_agent.max_iterations_min,
            ),
            max_iterations_max=get_max_int(
                df_trials,
                "max_iterations",
                search_space.react_rag_agent.max_iterations_max,
            ),
        )

    if "critique_rag_agent" in rag_modes:
        params["critique_rag_agent"] = CritiqueRAGAgent(  # type: ignore
            subquestion_engine_llms=get_unique_strings(
                df_trials, "subquestion_engine_llm"
            ),
            subquestion_response_synthesizer_llms=get_unique_strings(
                df_trials, "subquestion_response_synthesizer_llm"
            ),
            critique_agent_llms=get_unique_strings(df_trials, "critique_agent_llm"),
            reflection_agent_llms=get_unique_strings(df_trials, "reflection_agent_llm"),
            max_iterations_min=get_min_int(
                df_trials,
                "max_iterations",
                search_space.critique_rag_agent.max_iterations_min,
            ),
            max_iterations_max=get_max_int(
                df_trials,
                "max_iterations",
                search_space.critique_rag_agent.max_iterations_max,
            ),
        )

    if "sub_question_rag" in rag_modes:
        params["sub_question_rag"] = SubQuestionRAGAgent(  # type: ignore
            subquestion_engine_llms=get_unique_strings(
                df_trials, "subquestion_engine_llm"
            ),
            subquestion_response_synthesizer_llms=get_unique_strings(
                df_trials, "subquestion_response_synthesizer_llm"
            ),
        )

    if "lats_rag_agent" in rag_modes:
        params["lats_rag_agent"] = LATSRagAgent(  # type: ignore
            num_expansions_min=get_min_int(
                df_trials,
                "num_expansions",
                search_space.lats_rag_agent.num_expansions_min,
            ),
            num_expansions_max=get_max_int(
                df_trials,
                "num_expansions",
                search_space.lats_rag_agent.num_expansions_max,
            ),
            max_rollouts_min=get_min_int(
                df_trials, "max_rollouts", search_space.lats_rag_agent.max_rollouts_min
            ),
            max_rollouts_max=get_max_int(
                df_trials, "max_rollouts", search_space.lats_rag_agent.max_rollouts_max
            ),
        )

    return SearchSpace(**params)  # type: ignore
