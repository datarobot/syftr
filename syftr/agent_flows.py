import functools
import json
import time
import typing as T
from typing import Protocol

from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptType
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from syftr.logger import logger


class AgentFlow:
    """Base class for agentic flows, containing common methods."""

    reranker_top_n: int
    reranker_llm: FunctionCallingLLM | None
    get_examples: T.Callable[[str], str]

    def __init__(
        self,
        llm: FunctionCallingLLM,
        template: str | None = None,
        **kwargs,
    ):
        self.llm = llm
        self.template = template
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.node_postprocessors = []
        if hasattr(self, "reranker_top_n"):
            self.node_postprocessors.append(
                LLMRerank(top_n=self.reranker_top_n, llm=self.reranker_llm)
            )
        self.name = self.__class__.__name__

    def get_prompt(self, query_str: str) -> str:
        if self.template is None:
            return query_str
        if not hasattr(self, "get_examples"):
            return self.template.format(query_str=query_str)
        examples = self.get_examples(query_str)  # type: ignore
        assert examples, "No examples found for few-shot prompting"
        return self.template.format(
            query_str=query_str,
            few_shot_examples=examples,
        )

    def get_prompt_template(self) -> PromptTemplate | None:
        prompt_template = None
        function_mappings = None
        if hasattr(self, "get_examples"):
            function_mappings = {"few_shot_examples": self.get_examples}
        if self.template:
            logger.debug("Creating prompt template from '%s'", self.template)
            prompt_template = PromptTemplate(
                template=self.template,
                prompt_type=PromptType.QUESTION_ANSWER,
                function_mappings=function_mappings,
            )
        return prompt_template


class AgentProtocol(Protocol):
    """Typing protocol for type-hinting Agent flows.

    usage:

        def foo(agent: AgentProtocol):
            agent.generate('hi')
    """

    llm: FunctionCallingLLM
    llm_name: str
    query_engine: RetrieverQueryEngine | TransformQueryEngine
    node_postprocessors: T.List[BaseNodePostprocessor] | None
    agent: ReActAgent
    name: str
    use_hyde: bool
    hyde_llm: FunctionCallingLLM | None

    def get_prompt(self, query_str: str) -> str:
        pass

    def get_prompt_template(self) -> PromptTemplate:
        pass

    def generate(self, query_str: str) -> T.Tuple[str, float]:
        pass

    async def agenerate(self, query_str: str) -> str:
        pass


class AgentMixin:
    """Default agentic generate method implementations.

    Add this to your class inheritance when making a new agent flow:

        class MyAgentFlow(AgentFlow, AgentMixin):
            ...
    """

    def generate(self: AgentProtocol, query_str: str) -> T.Tuple[str, float]:
        start_time = time.perf_counter()
        response = self.agent.chat(query_str)
        duration = time.perf_counter() - start_time
        return response, duration

    async def agenerate(self: AgentProtocol, query_str: str) -> T.Tuple[str, float]:
        start_time = time.perf_counter()
        response = await self.agent.achat(query_str)
        duration = time.perf_counter() - start_time
        return response, duration


class LlamaIndexReactRAGAgentFlow(AgentFlow, AgentMixin):
    def __init__(
        self: AgentProtocol,
        indexes: T.List[T.Tuple[str, str, VectorStoreIndex]],
        llm: FunctionCallingLLM,
        system_prompt: PromptTemplate | None = None,
        template: str | None = None,
        **kwargs,
    ):
        super().__init__(llm=llm, template=template, **kwargs)  # type: ignore

        prompt_template: PromptTemplate = self.get_prompt_template()
        query_engines = [
            index.as_query_engine(
                llm=self.llm,
                text_qa_template=prompt_template,
                node_postprocessors=self.node_postprocessors,
            )
            for _, _, index in indexes
        ]
        query_engine_tools = [
            QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name="Grounding_%s" % name,
                    description="Grounding data related to %s" % desc,
                ),
            )
            for (name, desc, _), query_engine in zip(indexes, query_engines)
        ]
        self.agent = ReActAgent.from_tools(
            query_engine_tools,  # type: ignore
            llm=self.llm,
            verbose=True,
        )
        if system_prompt:
            self.agent.update_prompts({"agent_worker:system_prompt": system_prompt})
            self.agent.reset()


class FlowJSONDecodeError(Exception):
    pass


def chain_hasattr(obj, attr):
    for a in attr.split("."):
        if hasattr(obj, a):
            obj = getattr(obj, a)
        else:
            return False
    return True


def chain_getattr(obj, attr):
    return functools.reduce(
        lambda obj, attr: getattr(obj, attr), [obj] + attr.split(".")
    )


class FlowJSONHandler(json.JSONEncoder):
    copy_args: T.List[str] = [
        "fusion_mode",
        "num_queries",
        "retriever_weights",
        "similarity_top_k",
        "use_hyde",
        "use_reranker",
        "reranker_top_n",
        "retriever",
        "embedding_model",
        "splitter",
        "template_name",
        "rag_mode",
        "llm_name",
        "reranker_llm_name",
        "hyde_llm_name",
        "chunk_size",
        "chunk_overlap",
    ]

    def default(self, obj):
        out = {}
        for arg in self.copy_args:
            if hasattr(obj, arg):
                out[arg] = getattr(obj, arg)
        retrievers = []
        if chain_hasattr(obj, "query_engine.retriever"):
            retrievers.append({"name": obj.query_engine.retriever.__class__.__name__})
        elif chain_hasattr(obj, "query_engine._query_engine.retriever"):
            retrievers.append(
                {"name": obj.query_engine._query_engine.retriever.__class__.__name__}
            )
        if retrievers:
            out["retrievers"] = retrievers
        if hasattr(obj, "retrievers"):
            retrievers += obj.retrievers
            out["retrievers"] = retrievers
        if hasattr(obj, "agent"):
            out["agent"] = obj.agent.__class__.__name__
        return out
