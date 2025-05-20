import time
import typing as T
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import cached_property
from uuid import uuid4

import llama_index.core.instrumentation as instrument
from llama_index.agent.introspective import (
    IntrospectiveAgentWorker,
    ToolInteractiveReflectionAgentWorker,
)
from llama_index.agent.introspective.reflective.tool_interactive_reflection import (
    StoppingCallable,
)
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core import (
    PromptTemplate,
    QueryBundle,
    Response,
    get_response_synthesizer,
)
from llama_index.core.agent import (
    AgentChatResponse,
    AgentRunner,
    FunctionCallingAgentWorker,
    ReActAgent,
)
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.llms import ChatMessage, CompletionResponse, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor import LLMRerank, PrevNextNodePostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptType
from llama_index.core.query_engine import (
    BaseQueryEngine,
    RetrieverQueryEngine,
    SubQuestionQueryEngine,
    TransformQueryEngine,
)
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.tools import BaseTool, QueryEngineTool, ToolMetadata
from numpy import ceil

from syftr.configuration import cfg
from syftr.instrumentation.arize import instrument_arize
from syftr.instrumentation.tokens import LLMCallData, TokenTrackingEventHandler
from syftr.llm import get_llm_name, get_tokenizer
from syftr.logger import logger
from syftr.studies import get_critique_template, get_react_template

dispatcher = instrument.get_dispatcher()
_event_handler = TokenTrackingEventHandler()
dispatcher.add_event_handler(_event_handler)
dispatcher.add_span_handler(_event_handler._span_handler)

if cfg.instrumentation.tracing_enabled:
    instrument_arize()


@dataclass(kw_only=True)
class Flow:
    response_synthesizer_llm: LLM | FunctionCallingLLM
    template: str | None = None
    get_examples: T.Callable | None = None
    name: str = "Generator Flow"
    params: dict | None = None
    enforce_full_evaluation: bool = False

    _llm_call_data: T.Dict[str, T.List[LLMCallData]] = field(default_factory=dict)

    def __repr__(self):
        return f"{self.name}: {self.params}"

    @property
    def verbose(self):
        log_level = logger.level
        if log_level <= 20:
            return True
        return False

    @property
    def prompt_template(self) -> PromptTemplate:
        if self.template is None:
            raise ValueError("Flow template not set. Cannot create prompt template.")
        prompt_template = None
        function_mappings = None
        if self.get_examples is not None:
            function_mappings = {"few_shot_examples": self.get_examples}

        logger.debug("Creating prompt template from '%s'", self.template)
        prompt_template = PromptTemplate(
            template=self.template,
            prompt_type=PromptType.QUESTION_ANSWER,
            function_mappings=function_mappings,
        )

        return prompt_template

    def get_prompt(self, query) -> str:
        if self.template is None:
            return query

        if self.get_examples is None:
            return self.template.format(query_str=query)

        examples = self.get_examples(query)
        assert examples, "No examples found for few-shot prompting"

        return self.template.format(
            query_str=query,
            few_shot_examples=examples,
        )

    def generate(
        self, query: str
    ) -> T.Tuple[CompletionResponse, float, T.List[LLMCallData]]:
        invocation_id = uuid4().hex
        self._llm_call_data[invocation_id] = []
        response, duration = self._generate(query, invocation_id)
        call_data = self._llm_call_data.pop(invocation_id)
        return response, duration, call_data

    @dispatcher.span
    def _generate(
        self, query: str, invocation_id: str
    ) -> T.Tuple[CompletionResponse, float]:
        assert self.response_synthesizer_llm is not None, (
            "Response synthesizer LLM is not set. Cannot generate."
        )
        start_time = time.perf_counter()
        prompt = self.get_prompt(query)
        response: CompletionResponse = self.response_synthesizer_llm.complete(prompt)
        duration = time.perf_counter() - start_time
        return response, duration

    async def agenerate(
        self, query: str
    ) -> T.Tuple[CompletionResponse, float, T.List[LLMCallData]]:
        invocation_id = uuid4().hex
        self._llm_call_data[invocation_id] = []
        response, duration = await self._agenerate(query, invocation_id)
        call_data = self._llm_call_data.pop(invocation_id)
        return response, duration, call_data

    @dispatcher.span
    async def _agenerate(
        self, query: str, invocation_id: str
    ) -> T.Tuple[CompletionResponse, float]:
        assert self.response_synthesizer_llm is not None, (
            "Response synthesizer LLM is not set. Cannot generate."
        )
        start_time = time.perf_counter()
        prompt = self.get_prompt(query)
        response: CompletionResponse = await self.response_synthesizer_llm.acomplete(
            prompt
        )
        duration = time.perf_counter() - start_time
        return response, duration


@dataclass(kw_only=True)
class RetrieverFlow(Flow):
    """Flow that only retrieves documents."""

    retriever: BaseRetriever
    docstore: BaseDocumentStore | None = None
    hyde_llm: LLM | None = None
    additional_context_num_nodes: int = 0
    name: str = "Retriever Only Flow"

    def __repr__(self):
        return f"{self.name}: {self.params}"

    @cached_property
    def query_engine(self) -> BaseQueryEngine:
        node_postprocessors: T.List[BaseNodePostprocessor] = []
        if self.additional_context_num_nodes > 0:
            assert self.docstore is not None
            node_postprocessors.append(
                PrevNextNodePostprocessor(
                    docstore=self.docstore,
                    num_nodes=int(ceil(self.additional_context_num_nodes / 2)),
                    mode="both",
                )
            )
        response_synthesizer = get_response_synthesizer(
            llm=self.response_synthesizer_llm,
            response_mode=ResponseMode.COMPACT,
        )
        base_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
        )
        if self.hyde_llm is not None:
            hyde = HyDEQueryTransform(llm=self.hyde_llm, include_original=True)
            return TransformQueryEngine(base_engine, query_transform=hyde)
        return base_engine

    @cached_property
    def tokenizer(self) -> T.Callable:
        return get_tokenizer(get_llm_name(self.response_synthesizer_llm))

    def generate(self, query: str, *args, **kwargs):
        raise NotImplementedError("RetrieverFlow does not support generation.")

    async def agenerate(self, query: str, *args, **kwargs):
        raise NotImplementedError("RetrieverFlow does not support generation.")

    @dispatcher.span
    def retrieve(self, query: str) -> T.Tuple[T.List[NodeWithScore], float]:
        start_time = time.perf_counter()
        qb = QueryBundle(query)
        if isinstance(self.query_engine, TransformQueryEngine):
            response = self.query_engine.query(qb)
            assert isinstance(response, Response), (
                f"Expected Response, got {type(response)=}"
            )
            retrieval_result = response.source_nodes
        else:
            retrieval_result = self.query_engine.retrieve(qb)
        duration = time.perf_counter() - start_time
        return retrieval_result, duration

    @dispatcher.span
    async def aretrieve(self, query: str) -> T.Tuple[T.List[NodeWithScore], float]:
        start_time = time.perf_counter()
        qb = QueryBundle(query)
        if isinstance(self.query_engine, TransformQueryEngine):
            response = await self.query_engine.aquery(qb)
            assert isinstance(response, Response), (
                f"Expected Response, got {type(response)=}"
            )
            retrieval_result = response.source_nodes
        else:
            assert hasattr(self.query_engine, "aretrieve"), (
                f"{self.query_engine} does not have 'aretrieve' method"
            )
            retrieval_result = await self.query_engine.aretrieve(qb)
        duration = time.perf_counter() - start_time
        return retrieval_result, duration


@dataclass(kw_only=True)
class RAGFlow(Flow):
    retriever: BaseRetriever
    docstore: BaseDocumentStore | None = None
    hyde_llm: LLM | None = None
    reranker_llm: LLM | None = None
    reranker_top_k: int | None = None
    name: str = "RAG Flow"
    additional_context_num_nodes: int = 0

    def __repr__(self):
        return f"{self.name}: {self.params}"

    @cached_property
    def query_engine(self) -> BaseQueryEngine:
        node_postprocessors: T.List[BaseNodePostprocessor] = []
        if self.reranker_llm is not None:
            assert self.reranker_top_k, (
                "Reranker enabled, need reranker_top_k param set"
            )
            node_postprocessors.append(
                LLMRerank(top_n=self.reranker_top_k, llm=self.reranker_llm)
            )
        if self.additional_context_num_nodes > 0:
            assert self.docstore is not None
            node_postprocessors.append(
                PrevNextNodePostprocessor(
                    docstore=self.docstore,
                    num_nodes=int(ceil(self.additional_context_num_nodes / 2)),
                    mode="both",
                )
            )
        response_synthesizer = get_response_synthesizer(
            llm=self.response_synthesizer_llm,
            response_mode=ResponseMode.COMPACT,
            text_qa_template=self.prompt_template,
        )
        retriever = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
        )
        if self.hyde_llm is not None:
            hyde = HyDEQueryTransform(llm=self.hyde_llm, include_original=True)
            retriever = TransformQueryEngine(retriever, query_transform=hyde)  # type: ignore
        return retriever

    def get_prompt(self, query) -> str:
        if self.template is None:
            return query

        if self.get_examples is None:
            return self.template.format(query_str=query)

        examples = self.get_examples(query)
        assert examples, "No examples found for few-shot prompting"

        return self.template.format(
            query_str=query,
            few_shot_examples=examples,
        )

    def retrieve(self, query: str) -> T.List[NodeWithScore]:
        return self.query_engine.retrieve(QueryBundle(query))

    async def aretrieve(self, query: str) -> T.List[NodeWithScore]:
        assert hasattr(self.query_engine, "aretrieve"), (
            f"{self.query_engine} does not have 'aretrieve' method"
        )
        return await self.query_engine.aretrieve(QueryBundle(query))

    @dispatcher.span
    def _generate(
        self, query: str, invocation_id: str
    ) -> T.Tuple[CompletionResponse, float]:
        start_time = time.perf_counter()
        response = self.query_engine.query(query)
        assert isinstance(response, Response), (
            f"Expected Response, got {type(response)=}"
        )
        completion_response = CompletionResponse(
            text=str(response.response),
            additional_kwargs={
                "source_nodes": response.source_nodes,
                **(response.metadata or {}),  # type: ignore
            },
        )
        duration = time.perf_counter() - start_time
        return completion_response, duration

    @dispatcher.span
    async def _agenerate(
        self, query: str, invocation_id: str
    ) -> T.Tuple[CompletionResponse, float]:
        start_time = time.perf_counter()
        response = await self.query_engine.aquery(query)
        assert isinstance(response, Response), (
            f"Expected Response, got {type(response)=}"
        )
        assert isinstance(response.response, str), (
            f"Expected str, got {type(response.response)=}"
        )
        completion_response = CompletionResponse(
            text=str(response.response),
            additional_kwargs={
                "source_nodes": response.source_nodes,
                **(response.metadata or {}),
            },
        )
        duration = time.perf_counter() - start_time
        return completion_response, duration


@dataclass(kw_only=True)
class SubQuestionRAGFlow(RAGFlow):
    dataset_name: str
    dataset_description: str
    subquestion_engine_llm: LLM
    subquestion_response_synthesizer_llm: LLM
    name: str = "Sub-Question Agentic RAG Flow"

    def __repr__(self):
        return f"{self.name}: {self.params}"

    @cached_property
    def tools(self) -> T.List[QueryEngineTool]:
        # Build query engine like RAGFlow, since this class overrides query_engine
        node_postprocessors: T.List[BaseNodePostprocessor] = []
        if self.reranker_llm is not None:
            assert self.reranker_top_k, (
                "Reranker enabled, need reranker_top_k param set"
            )
            node_postprocessors.append(
                LLMRerank(top_n=self.reranker_top_k, llm=self.reranker_llm)
            )
        if self.additional_context_num_nodes > 0:
            assert self.docstore is not None
            node_postprocessors.append(
                PrevNextNodePostprocessor(
                    docstore=self.docstore,
                    num_nodes=int(ceil(self.additional_context_num_nodes / 2)),
                    mode="both",
                )
            )
        response_synthesizer = get_response_synthesizer(
            llm=self.subquestion_response_synthesizer_llm,
            response_mode=ResponseMode.COMPACT,
            text_qa_template=self.prompt_template,
        )
        retriever = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
        )
        if self.hyde_llm is not None:
            hyde = HyDEQueryTransform(llm=self.hyde_llm, include_original=True)
            retriever = TransformQueryEngine(retriever, query_transform=hyde)  # type: ignore
        return [
            QueryEngineTool(
                query_engine=retriever,
                metadata=ToolMetadata(
                    name=self.dataset_name.replace("/", "_"),
                    description=self.dataset_description,
                ),
            )
        ]

    @cached_property
    def query_engine(self) -> BaseQueryEngine:
        synth = get_response_synthesizer(
            llm=self.response_synthesizer_llm,
            use_async=True,
            response_mode=ResponseMode.TREE_SUMMARIZE,
        )

        return SubQuestionQueryEngine.from_defaults(
            query_engine_tools=self.tools,
            llm=self.subquestion_engine_llm,
            verbose=self.verbose,
            response_synthesizer=synth,
            use_async=False,
        )


@dataclass(kw_only=True)
class AgenticRAGFlow(RAGFlow):
    """Base agentic RAG flow with common fields and methods."""

    dataset_name: str
    dataset_description: str
    name: str = "Agentic RAG Flow"

    def __repr__(self):
        return f"{self.name}: {self.params}"

    @cached_property
    def tools(self) -> T.List[BaseTool]:
        return [
            QueryEngineTool(
                query_engine=self.query_engine,
                metadata=ToolMetadata(
                    name=self.dataset_name.replace("/", "_"),
                    description=self.dataset_description,
                ),
            )
        ]

    @cached_property
    def agent(self) -> AgentRunner:
        raise NotImplementedError()

    @dispatcher.span
    def _generate(
        self,
        query: str,
        invocation_id: str,
    ) -> T.Tuple[CompletionResponse, float]:
        start_time = time.perf_counter()
        response: AgentChatResponse = self.agent.chat(query)
        try:
            completion_response = CompletionResponse(
                text=response.response,
                additional_kwargs=asdict(response),
            )
        except TypeError:
            logger.error("Incorrect response from an agent: %s", response)
            raise
        duration = time.perf_counter() - start_time
        return completion_response, duration

    @dispatcher.span
    async def _agenerate(
        self,
        query: str,
        invocation_id: str,
    ) -> T.Tuple[CompletionResponse, float]:
        start_time = time.perf_counter()
        response: AgentChatResponse = await self.agent.achat(query)
        try:
            completion_response = CompletionResponse(
                text=response.response, additional_kwargs=asdict(response)
            )
        except TypeError:
            logger.error("Incorrect response from an agent: %s", response)
            raise
        duration = time.perf_counter() - start_time
        return completion_response, duration


@dataclass(kw_only=True)
class ReActAgentFlow(AgenticRAGFlow):
    react_prompt: str = get_react_template()
    subquestion_engine_llm: LLM
    max_iterations: int = 10
    name: str = "ReAct Agentic RAG Flow"
    subquestion_response_synthesizer_llm: LLM

    def __repr__(self):
        return f"{self.name}: {self.params}"

    @cached_property
    def agent(self) -> AgentRunner:
        synth = get_response_synthesizer(
            llm=self.subquestion_response_synthesizer_llm,
            use_async=True,
            response_mode=ResponseMode.TREE_SUMMARIZE,
            text_qa_template=self.prompt_template,
        )
        sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=self.tools,  # type: ignore
            llm=self.subquestion_engine_llm,
            verbose=self.verbose,
            response_synthesizer=synth,
            use_async=False,
        )

        tools = [
            QueryEngineTool(
                query_engine=sub_question_engine,
                metadata=ToolMetadata(
                    name=self.dataset_name.replace("/", "_"),
                    description=self.dataset_description,
                ),
            )
        ]

        formatter = ReActChatFormatter.from_defaults(system_header=self.react_prompt)
        return ReActAgent.from_tools(
            tools,  # type: ignore
            llm=self.response_synthesizer_llm,
            max_iterations=self.max_iterations,
            react_chat_formatter=formatter,
            default_tool_choice="any",
            verbose=self.verbose,
        )


@dataclass(kw_only=True)
class CritiqueAgentFlow(AgenticRAGFlow):
    subquestion_engine_llm: LLM
    subquestion_response_synthesizer_llm: LLM
    critique_agent_llm: FunctionCallingLLM
    reflection_agent_llm: FunctionCallingLLM
    max_iterations: int = 10
    critique_template = get_critique_template()
    stopping_condition: StoppingCallable = lambda critique_str: "PASS" in critique_str
    name: str = "Critique Agentic RAG Flow"

    def __repr__(self):
        return f"{self.name}: {self.params}"

    @cached_property
    def agent(self) -> AgentRunner:
        assert isinstance(self.response_synthesizer_llm, FunctionCallingLLM), (
            f"CritiqueAgentFlow requires FunctionCallingLLM. Got {type(self.response_synthesizer_llm)=}"
        )
        main_worker = FunctionCallingAgentWorker.from_tools(
            tools=self.tools, llm=self.response_synthesizer_llm, verbose=self.verbose
        )
        synth = get_response_synthesizer(
            llm=self.subquestion_response_synthesizer_llm,
            use_async=True,
            response_mode=ResponseMode.TREE_SUMMARIZE,
            text_qa_template=self.prompt_template,
        )
        sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=self.tools,  # type: ignore
            llm=self.subquestion_engine_llm,
            verbose=self.verbose,
            response_synthesizer=synth,
            use_async=False,
        )

        tools = [
            QueryEngineTool(
                query_engine=sub_question_engine,
                metadata=ToolMetadata(
                    name=self.dataset_name.replace("/", "_"),
                    description=self.dataset_description,
                ),
            )
        ]

        critique_agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=tools,  # type: ignore
            llm=self.critique_agent_llm,
            verbose=self.verbose,
        )

        critique_worker = ToolInteractiveReflectionAgentWorker.from_defaults(
            max_iterations=self.max_iterations,
            critique_agent_worker=critique_agent_worker,
            critique_template=self.critique_template,
            stopping_callable=self.stopping_condition,
            correction_llm=self.reflection_agent_llm,
            verbose=self.verbose,
        )

        introspective_agent_worker = IntrospectiveAgentWorker.from_defaults(
            reflective_agent_worker=critique_worker,
            main_agent_worker=main_worker,
            verbose=self.verbose,
        )

        chat_history = [
            ChatMessage(
                content="You are an assistant that fixes incorrect user-provided answers.",
                role=MessageRole.SYSTEM,
            )
        ]

        return introspective_agent_worker.as_agent(
            verbose=self.verbose,
            chat_history=chat_history,
            default_tool_choice="any",
        )


@dataclass(kw_only=True)
class LATSAgentFlow(AgenticRAGFlow):
    dataset_name: str
    dataset_description: str
    name: str = "LATS Agent Flow"
    num_expansions: int = 2
    max_rollouts: int = 3

    def __repr__(self):
        return f"{self.name}: {self.params}"

    @cached_property
    def tools(self) -> T.List[BaseTool]:
        return [
            QueryEngineTool(
                query_engine=self.query_engine,
                metadata=ToolMetadata(
                    name=self.dataset_name.replace("/", "_"),
                    description=self.dataset_description,
                ),
            )
        ]

    @cached_property
    def agent(self) -> AgentRunner:
        agent_worker = LATSAgentWorker.from_tools(
            self.tools,
            llm=self.response_synthesizer_llm,
            num_expansions=self.num_expansions,
            max_rollouts=self.max_rollouts,
            verbose=self.verbose,
        )
        return agent_worker.as_agent(
            default_tool_choice="any",
        )


class Flows(Enum):
    GENERATOR_FLOW = Flow
    RAG_FLOW = RAGFlow
    LLAMA_INDEX_REACT_AGENT_FLOW = ReActAgentFlow
    LLAMA_INDEX_CRITIQUE_AGENT_FLOW = CritiqueAgentFlow
    LLAMA_INDEX_SUB_QUESTION_FLOW = SubQuestionRAGFlow
    LLAMA_INDEX_LATS_RAG_AGENT = LATSAgentFlow
    RETRIEVER_FLOW = RetrieverFlow
