import typing as T
from functools import partial
from pathlib import Path

import datasets
import llama_index.core.instrumentation as instrument
import pandas as pd
import pytest
from fsspec.implementations.local import LocalFileSystem
from llama_index.core import Document, Settings, SimpleDirectoryReader
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.llms.azure_openai import AzureOpenAI

from syftr.configuration import AzureOpenAILLM, LLMCostTokens, LLMMetadata, cfg
from syftr.configuration import Settings as CfgSettings
from syftr.flows import Flow, RAGFlow, ReActAgentFlow
from syftr.huggingface_helper import get_embedding_model
from syftr.llm import get_llm, load_configured_llms
from syftr.retrievers.build import _build_dense_index, build_rag_retriever
from syftr.storage import HotPotQAHF, PartitionMap, QAPair, SyftrQADataset
from syftr.studies import (
    EmbeddingDeviceType,
    OptimizationConfig,
    ParamDict,
    StudyConfig,
    TimeoutConfig,
)
from syftr.templates import get_template

dispatcher = instrument.get_dispatcher()


@pytest.fixture(scope="session")
def bge_small():
    model, _ = get_embedding_model(
        "BAAI/bge-small-en-v1.5", use_hf_endpoint_models=False
    )
    return model


@pytest.fixture(scope="session")
def gpt_4o_mini() -> T.Tuple[AzureOpenAI, str]:
    return get_llm("gpt-4o-mini"), "gpt-4o-mini"


@pytest.fixture(scope="session")
def basic_flow(gpt_4o_mini):
    llm, _ = gpt_4o_mini
    return Flow(response_synthesizer_llm=llm, template=get_template("default"))


@pytest.fixture(
    scope="session",
    params=[
        (HotPotQAHF, "dev", "sample", "A collection of various trivia from Wikipedia"),
        # (CragTask3HF, "music", "sample", "Various information related to music"),
        # (
        #     FinanceBenchHF,
        #     None,
        #     "sample",
        #     "A collection of financial data and statements",
        # ),
    ],
)
def real_dataset(request) -> SyftrQADataset:
    dataset_class, subset, partition, description = request.param
    if partition is not None:
        p_map = PartitionMap(test=partition)
    else:
        p_map = PartitionMap()
    return dataset_class(subset=subset, partition_map=p_map, description=description)


@pytest.fixture(scope="session")
def tiny_dataset() -> SyftrQADataset:
    class TinyDataset(SyftrQADataset):
        xname: str = "tiny-wiki-llm-page"  # type: ignore

        path_root: Path = cfg.paths.root_dir / "tests/functional/data/datasets"  # type: ignore
        dataset_dir: str = "partitioned"
        grounding_data_path: str = "grounding_data"
        examples_data_path: str = "shouldnotexist"

        def _row_to_qapair(self, row: T.Dict[str, T.Any]) -> QAPair:
            """Dataset-specific conversion of row to QAPair struct.

            Invoked by iter_examples.

            Default implementation assumes row is already in QAPair format.
            """
            return QAPair(**row)

        def iter_examples(self, partition="test", use_ray=False) -> T.Iterator[QAPair]:
            examples = self.load_examples()
            _iter: T.Callable[[], T.Iterator[T.Dict[str, T.Any]]]
            if use_ray:
                # Ray ray.datasets.Dataset.iter_rows
                _iter = examples.iter_rows
            else:
                # Huggingface datasets.Dataset.iter
                assert isinstance(examples, datasets.Dataset)

                def _iter():
                    for row in examples.iter(batch_size=1):
                        yield {key: value[0] for key, value in row.items()}

            for row in _iter():
                yield self._row_to_qapair(row)

        def iter_grounding_data(self, partition="notused") -> T.Iterator[Document]:
            """Iterate over grounding data examples."""
            grounding_data = self.load_grounding_data()
            # _iter: T.Callable[[], T.Iterator[Document]]

            def _iter():
                return iter(grounding_data)

            for row in _iter():
                yield row

        @property
        def storage(self) -> LocalFileSystem:
            return LocalFileSystem()

        def load_examples(
            self,
        ) -> datasets.Dataset:
            data = [
                {
                    "question": "What are some commonly used benchmarking datasets?",
                    "answer": "TruthfulQA, Web Questions, TriviaQA, and SQuAD",
                    "id": "1",
                    "context": {},
                    "supporting_facts": [
                        "Some examples of commonly used question answering datasets include TruthfulQA, Web Questions, TriviaQA, and SQuAD."
                    ],
                    "difficulty": "easy",
                    "qtype": "simple",
                },
                {
                    "question": "How much compute do you need to train llama 3.1",
                    "answer": "The 405B version of Llama 3.1 took 31 million hours on H100-80GB GPUs.",
                    "id": "2",
                    "context": {},
                    "supporting_facts": [
                        "405B version took 31 million hours on H100-80GB, at 3.8E25 FLOPs."
                    ],
                    "difficulty": "easy",
                    "qtype": "simple",
                },
            ]
            return datasets.Dataset.from_pandas(pd.DataFrame(data))

        def load_grounding_data(
            self,
            **load_kwargs,
        ) -> T.List[Document]:
            reader = SimpleDirectoryReader(
                input_dir="tests/functional/data/datasets/partitioned/tiny-wiki-llm-page/grounding_data/test",
            )
            return reader.load_data(**load_kwargs)

    return TinyDataset()


@pytest.fixture(scope="session")
def real_dataset_study_config(request, real_dataset) -> StudyConfig:
    device: EmbeddingDeviceType = (
        "cuda" if request.config.getoption("--gpu") else "onnx-cpu"
    )
    study_config = StudyConfig(
        name=f"functional_test_{real_dataset.name}",
        dataset=real_dataset,
        toy_mode=True,
        optimization=OptimizationConfig(embedding_device=device),
    )
    study_config.optimization.use_hf_embedding_models = False
    study_config.optimization.num_eval_samples = 10
    return study_config


@pytest.fixture(scope="session")
def tiny_dataset_study_config(hotpot_toy_study_config, tiny_dataset) -> StudyConfig:
    config: StudyConfig = hotpot_toy_study_config.copy()
    config.dataset = tiny_dataset
    config.optimization.use_hf_embedding_models = False
    return config


@pytest.fixture(scope="session")
def real_sparse_retriever(
    real_dataset_study_config,
) -> T.Tuple[BaseRetriever, BaseDocumentStore, StudyConfig]:
    params: ParamDict = {
        "rag_llm_name": "gpt-4o-mini",
        "rag_mode": "rag",
        "response_synthesizer_llm": "gpt-4o-mini",
        "rag_method": "sparse",
        "rag_top_k": 5,
        "rag_query_decomposition_enabled": True,
        "rag_query_decomposition_num_queries": 2,
        "rag_query_decomposition_llm_name": "gemini-flash2",
        "rag_fusion_mode": "simple",
        "splitter_method": "sentence",
        "splitter_chunk_exp": 9,
        "splitter_chunk_overlap_frac": 0.5,
    }
    retriever, docstore = build_rag_retriever(real_dataset_study_config, params)
    return retriever, docstore, real_dataset_study_config


@pytest.fixture(scope="session")
def real_dense_retriever(
    real_dataset_study_config,
) -> T.Tuple[BaseRetriever, BaseDocumentStore, StudyConfig]:
    params: ParamDict = {
        "rag_llm_name": "gpt-4o-mini",
        "rag_mode": "rag",
        "response_synthesizer_llm": "gpt-4o-mini",
        "rag_method": "dense",
        "rag_embedding_model": "BAAI/bge-small-en-v1.5",
        "rag_query_decomposition_enabled": False,
        "rag_top_k": 5,
        "splitter_method": "recursive",
        "splitter_chunk_exp": 9,
        "splitter_chunk_overlap_frac": 0.5,
    }
    retriever, docstore = build_rag_retriever(real_dataset_study_config, params)
    return retriever, docstore, real_dataset_study_config


@pytest.fixture(scope="session")
def tiny_sparse_retriever(
    tiny_dataset_study_config,
) -> T.Tuple[BaseRetriever, BaseDocumentStore]:
    params: ParamDict = {
        "rag_llm_name": "gpt-4o-mini",
        "rag_mode": "rag",
        "response_synthesizer_llm": "gpt-4o-mini",
        "rag_method": "sparse",
        "rag_top_k": 5,
        "rag_query_decomposition_enabled": False,
        "splitter_method": "sentence",
        "splitter_chunk_exp": 9,
        "splitter_chunk_overlap_frac": 0.5,
    }
    return build_rag_retriever(tiny_dataset_study_config, params)


@pytest.fixture(scope="session")
def tiny_dense_retriever(
    tiny_dataset_study_config,
) -> T.Tuple[BaseRetriever, BaseDocumentStore]:
    params: ParamDict = {
        "rag_llm_name": "gpt-4o-mini",
        "rag_mode": "rag",
        "response_synthesizer_llm": "gpt-4o-mini",
        "rag_method": "dense",
        "rag_embedding_model": "BAAI/bge-small-en-v1.5",
        "rag_query_decomposition_enabled": False,
        "rag_top_k": 5,
        "splitter_method": "recursive",
        "splitter_chunk_exp": 9,
        "splitter_chunk_overlap_frac": 0.5,
    }
    return build_rag_retriever(tiny_dataset_study_config, params)


@pytest.fixture(scope="session")
def real_hybrid_retriever(
    real_dataset_study_config,
) -> T.Tuple[BaseRetriever, BaseDocumentStore, StudyConfig]:
    params: ParamDict = {
        "rag_llm_name": "gpt-4o-mini",
        "rag_mode": "rag",
        "rag_method": "hybrid",
        "response_synthesizer_llm": "gpt-4o-mini",
        "rag_embedding_model": "BAAI/bge-small-en-v1.5",
        "rag_top_k": 5,
        "rag_query_decomposition_enabled": True,
        "rag_query_decomposition_llm_name": "gpt-4o-mini",
        "rag_query_decomposition_num_queries": 2,
        "rag_fusion_mode": "simple",
        "rag_hybrid_bm25_weight": 0.5,
        "splitter_method": "sentence",
        "splitter_chunk_exp": 9,
        "splitter_chunk_overlap_frac": 0.5,
    }
    retriever, docstore = build_rag_retriever(real_dataset_study_config, params)
    return retriever, docstore, real_dataset_study_config


@pytest.fixture(scope="session")
def rag_template():
    return get_template("default", with_context=True)


@pytest.fixture(scope="session")
def few_shot_rag_template():
    return get_template("default", with_context=True, with_few_shot_prompt=True)


@pytest.fixture(scope="session")
def bge_small_no_hf():
    model, _ = get_embedding_model(
        "BAAI/bge-small-en-v1.5", use_hf_endpoint_models=False
    )
    return model


@pytest.fixture(scope="session")
def correctness_evaluator(gpt_4o_mini):
    llm, _ = gpt_4o_mini
    evaluator = CorrectnessEvaluator(llm)
    evaluator.evaluate = dispatcher.span(evaluator.evaluate)
    evaluator.aevaluate = dispatcher.span(evaluator.aevaluate)
    return evaluator


@pytest.fixture
def hotpot_dense_flow(hotpot_dense_retriever, gpt_4o_mini, rag_template) -> RAGFlow:
    llm, _ = gpt_4o_mini
    hotpot_dense_retriever, _ = hotpot_dense_retriever
    return RAGFlow(
        retriever=hotpot_dense_retriever,
        response_synthesizer_llm=llm,
        template=rag_template,
    )


@pytest.fixture
def hotpot_fusion_flow(hotpot_fusion_retriever, gpt_4o_mini, rag_template) -> RAGFlow:
    llm, _ = gpt_4o_mini
    hotpot_fusion_retriever, _ = hotpot_fusion_retriever
    return RAGFlow(
        retriever=hotpot_fusion_retriever,
        response_synthesizer_llm=llm,
        template=rag_template,
    )


@pytest.fixture
def hotpot_reranker_flow(hotpot_dense_retriever, gpt_4o_mini, rag_template) -> RAGFlow:
    llm, _ = gpt_4o_mini
    retriever, _ = hotpot_dense_retriever
    return RAGFlow(
        retriever=retriever,
        response_synthesizer_llm=llm,
        template=rag_template,
        reranker_llm=llm,
        reranker_top_k=5,
    )


@pytest.fixture(scope="session")
def hotpot_toy_study_config() -> StudyConfig:
    return StudyConfig.from_file(cfg.paths.test_studies_dir / "test-hotpot-toy.yaml")


@pytest.fixture(scope="session")
def hotpot_sparse_retriever(
    hotpot_toy_study_config,
) -> T.Tuple[BaseRetriever, BaseDocumentStore]:
    params: ParamDict = {
        "rag_mode": "rag",
        "rag_method": "sparse",
        "rag_top_k": 5,
        "rag_query_decomposition_enabled": False,
        "splitter_method": "sentence",
        "splitter_chunk_exp": 9,
        "splitter_chunk_overlap_frac": 0.5,
    }
    return build_rag_retriever(hotpot_toy_study_config, params)


@pytest.fixture(scope="session")
def hotpot_dense_retriever(
    hotpot_toy_study_config,
) -> T.Tuple[BaseRetriever, BaseDocumentStore]:
    params: ParamDict = {
        "response_synthesizer_llm": "gpt-4o-mini",
        "rag_mode": "rag",
        "rag_method": "dense",
        "rag_embedding_model": "BAAI/bge-small-en-v1.5",
        "rag_top_k": 5,
        "rag_query_decomposition_enabled": False,
        "splitter_method": "sentence",
        "splitter_chunk_exp": 9,
        "splitter_chunk_overlap_frac": 0.5,
    }
    return build_rag_retriever(hotpot_toy_study_config, params)


@pytest.fixture(scope="session")
def hotpot_dense_retriever_recursive(
    hotpot_toy_study_config,
) -> T.Tuple[BaseRetriever, BaseDocumentStore]:
    params: ParamDict = {
        "rag_mode": "rag",
        "rag_method": "dense",
        "rag_embedding_model": "BAAI/bge-small-en-v1.5",
        "rag_top_k": 5,
        "rag_query_decomposition_enabled": False,
        "splitter_method": "recursive",
        "splitter_chunk_exp": 9,
        "splitter_chunk_overlap_frac": 0.5,
    }
    return build_rag_retriever(hotpot_toy_study_config, params)


@pytest.fixture(scope="session")
def hotpot_fusion_retriever(
    hotpot_toy_study_config,
) -> T.Tuple[BaseRetriever, BaseDocumentStore]:
    params: ParamDict = {
        "rag_mode": "rag",
        "rag_method": "fusion",
        "rag_embedding_model": "BAAI/bge-small-en-v1.5",
        "rag_top_k": 5,
        "rag_query_decomposition_enabled": False,
        "rag_fusion_llm_name": "gpt-4o-mini",
        "rag_fusion_num_queries": 2,
        "rag_fusion_mode": "simple",
        "splitter_method": "sentence",
        "splitter_chunk_exp": 9,
        "splitter_chunk_overlap_frac": 0.5,
    }
    return build_rag_retriever(hotpot_toy_study_config, params)


@pytest.fixture(scope="session")
def llama_debug():
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    Settings.callback_manager = callback_manager
    return llama_debug


@pytest.fixture
def tiny_flow(tiny_dense_retriever, gpt_4o_mini, rag_template) -> RAGFlow:
    llm, _ = gpt_4o_mini
    retriever, _ = tiny_dense_retriever
    return RAGFlow(
        retriever=retriever,
        response_synthesizer_llm=llm,
        template=rag_template,
        params={},
    )


@pytest.fixture
def hotpot_dev(hotpot_toy_study_config):
    return hotpot_toy_study_config.dataset


@pytest.fixture
def build_index_that_will_time_out(bge_small_no_hf, hotpot_dev):
    """Only use if you want to test the embeddings timeout functionality."""
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
    transforms = [splitter]

    bge_small_no_hf.reset_timeouts(
        timeout_config=TimeoutConfig(
            embedding_timeout_active=True,
            embedding_max_time=2,
            embedding_min_chunks_to_process=20,
            embedding_min_time_to_process=1,
        )
    )

    return partial(
        _build_dense_index,
        list(hotpot_dev.iter_grounding_data()),
        transforms,
        bge_small_no_hf,
    )


@pytest.fixture
def build_index_that_will_skip_time_out(bge_small_no_hf, hotpot_dev):
    """Only use if you want to test the embeddings timeout functionality."""
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
    transforms = [splitter]

    bge_small_no_hf.reset_timeouts(
        timeout_config=TimeoutConfig(
            embedding_timeout_active=False,
            embedding_max_time=2,
            embedding_min_chunks_to_process=20,
            embedding_min_time_to_process=1,
        )
    )

    return partial(
        _build_dense_index,
        list(hotpot_dev.iter_grounding_data()),
        transforms,
        bge_small_no_hf,
    )


@pytest.fixture
def build_index_that_will_not_time_out(bge_small_no_hf, hotpot_dev):
    """Only use if you want to test the embeddings timeout functionality."""
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
    transforms = [splitter]

    bge_small_no_hf.reset_timeouts(
        timeout_config=TimeoutConfig(
            embedding_max_time=1000,
            embedding_min_chunks_to_process=50,
            embedding_min_time_to_process=2,
        )
    )

    return partial(
        _build_dense_index,
        list(hotpot_dev.iter_grounding_data()),
        transforms,
        bge_small_no_hf,
    )


@pytest.fixture(scope="session")
def sentence_splitter() -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=256,
        chunk_overlap=64,
    )


@pytest.fixture
def hotpot_study_config():
    study_config = StudyConfig.from_file(
        "tests/functional/data/studies/test-hotpot-toy.yaml"
    )
    return study_config


@pytest.fixture(scope="session")
def react_agent_flow(
    hotpot_dense_retriever, hotpot_toy_study_config, gpt_4o_mini, rag_template
) -> T.Tuple[ReActAgentFlow, StudyConfig]:
    llm, _ = gpt_4o_mini
    hotpot_dense_retriever, _ = hotpot_dense_retriever
    return (
        ReActAgentFlow(
            retriever=hotpot_dense_retriever,
            response_synthesizer_llm=llm,
            subquestion_engine_llm=llm,
            subquestion_response_synthesizer_llm=llm,
            template=rag_template,
            dataset_name=hotpot_toy_study_config.dataset.name,
            dataset_description=hotpot_toy_study_config.dataset.description,
        ),
        hotpot_toy_study_config,
    )


def config_with_models() -> CfgSettings:
    custom_config = CfgSettings(
        generative_models={
            "test_gpt_4o_mini": AzureOpenAILLM(
                deployment_name="gpt-4o-mini",
                api_version="2024-06-01",
                additional_kwargs={"user": "syftr"},
                cost=LLMCostTokens(input=0.15, output=0.60),
                metadata=LLMMetadata(
                    model_name="gpt-4o-mini",
                    context_window=128000,
                    is_function_calling_model=True,
                ),
            ),
            "test_gpt_4o": AzureOpenAILLM(
                deployment_name="gpt-4o",
                api_version="2024-06-01",
                additional_kwargs={"user": "syftr"},
                cost=LLMCostTokens(input=2.5, output=10.00),
                metadata=LLMMetadata(
                    model_name="gpt-4o",
                    context_window=128000,
                    is_function_calling_model=True,
                ),
            ),
        }
    )
    custom_config.generative_models = {
        key: value
        for key, value in custom_config.generative_models.items()
        if "test_" in key
    }
    return custom_config


@pytest.fixture(
    params=[name for name in config_with_models().generative_models.keys()],
    ids=[name for name in config_with_models().generative_models.keys()],
)
def configured_models(request) -> T.Tuple[str, LLM]:
    name = request.param
    llms = load_configured_llms(config_with_models())
    return name, llms[name]
