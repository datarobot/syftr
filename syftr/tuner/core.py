import typing as T

import optuna
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import (
    CodeSplitter,
    LangchainNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.node_parser.interface import NodeParser
from tenacity import retry, stop_after_attempt, wait_fixed
from transformers import AutoTokenizer

from syftr.flows import Flows
from syftr.llm import get_tokenizer
from syftr.optuna_helper import set_metrics
from syftr.studies import StudyConfig


def get_flow_name(rag_mode: str):
    match rag_mode:
        case "no_rag":
            return Flows.GENERATOR_FLOW.value.__name__
        case "rag":
            return Flows.RAG_FLOW.value.__name__
        case "sub_question_rag":
            return Flows.LLAMA_INDEX_SUB_QUESTION_FLOW.value.__name__
        case "react_rag_agent":
            return Flows.LLAMA_INDEX_REACT_AGENT_FLOW.value.__name__
        case "critique_rag_agent":
            return Flows.LLAMA_INDEX_CRITIQUE_AGENT_FLOW.value.__name__
        case "lats_rag_agent":
            return Flows.LLAMA_INDEX_LATS_RAG_AGENT.value.__name__
        case _:
            raise RuntimeError("Cannot identify flow")


def build_splitter(study_config: StudyConfig, params: T.Dict[str, T.Any]) -> NodeParser:
    chunk_size = 2 ** int(params["splitter_chunk_exp"])
    overlap = int(params["splitter_chunk_overlap_frac"] * chunk_size)
    llm_name = params["response_synthesizer_llm"]
    match params["splitter_method"]:
        case "html":
            return CodeSplitter(
                language="html",
                max_chars=4 * chunk_size,
            )
        case "sentence":
            return SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                tokenizer=get_tokenizer(llm_name),
            )
        case "token":
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                tokenizer=get_tokenizer(llm_name),
            )
        case "recursive":
            return LangchainNodeParser(
                RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                    tokenizer=AutoTokenizer.from_pretrained(
                        params.get("rag_embedding_model", "BAAI/bge-small-en-v1.5")
                    ),
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                )
            )
        case _:
            raise ValueError("Invalid splitter")


@retry(stop=stop_after_attempt(6), wait=wait_fixed(10))
def set_trial(
    trial: optuna.trial.FrozenTrial | optuna.trial.Trial,
    study_config: StudyConfig | None = None,
    params: dict[str, str | bool | int | float] | None = None,
    is_seeding: bool | None = None,
    metrics: T.Dict[str, float] | None = None,
    flow_json: str | None = None,
):
    if params:
        flow_name = get_flow_name(str(params["rag_mode"]))
        trial.set_user_attr("flow_name", flow_name)
    if study_config:
        trial.set_user_attr("dataset", study_config.dataset.name)
    if is_seeding is not None:
        trial.set_user_attr("is_seeding", is_seeding)
    if flow_json:
        trial.set_user_attr("flow", flow_json)
    if metrics:
        set_metrics(trial, metrics)
