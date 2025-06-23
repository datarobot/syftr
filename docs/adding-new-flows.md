# Adding New Flows

This guide documents the process of adding a new Flow type into Syftr, using a real-world example - the Chain-of-Abstraction agent ([paper](https://arxiv.org/abs/2401.17464) and [implementation](https://github.com/run-llama/llama_index/tree/main/llama-index-packs/llama-index-packs-agents-coa)).


## Add new dependencies

In this case, we need to add the LlamaIndex implementation of the CoA agent.
Dependencies should be added to the main `dependencies` section of the `pyproject.toml`.

```diff
diff --git a/pyproject.toml b/pyproject.toml
index 1c8674a..7812ef4 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -30,6 +29,7 @@ dependencies = [
     "langchain>0.3",
     "langchain-community",
     "llama-index",
+    "llama-index-agent-coa>=0.3.2",
     "llama-index-agent-introspective",
     "llama-index-agent-lats",
     "llama-index-agent-openai",
@@ -54,6 +54,7 @@ dependencies = [
     "llama-index-llms-vertex==0.4.3",
     "llama-index-llms-vllm",
     "llama-index-multi-modal-llms-openai",
+    "llama-index-packs-agents-coa>=0.3.2",
     "llama-index-program-openai",
     "llama-index-question-gen-openai",
     "llama-index-readers-file",
```

Then one can run `uv sync --extra dev` to complete the installation process.

## Add the new Flow class

Next we add the new flow to `syftr/flows.py`.
In this case we are adding a new `AgenticRAGFlow`.
Our flow has one unique parameter, `enable_calculator`, to enable custom calculator tools.

```diff
diff --git a/syftr/flows.py b/syftr/flows.py
index c0d22bd..4d86fb8 100644
--- a/syftr/flows.py
+++ b/syftr/flows.py
@@ -46,7 +46,8 @@ from llama_index.core.response_synthesizers.type import ResponseMode
 from llama_index.core.retrievers import BaseRetriever
 from llama_index.core.schema import NodeWithScore
 from llama_index.core.storage.docstore.types import BaseDocumentStore
-from llama_index.core.tools import BaseTool, QueryEngineTool, ToolMetadata
+from llama_index.core.tools import BaseTool, FunctionTool, QueryEngineTool, ToolMetadata
+from llama_index.packs.agents_coa import CoAAgentPack
 from numpy import ceil
 
 from syftr.configuration import cfg
@@ -664,6 +665,57 @@ class LATSAgentFlow(AgenticRAGFlow):
         )
 
 
+@dataclass(kw_only=True)
+class CoAAgentFlow(AgenticRAGFlow):
+    name: str = "CoA Agent Flow"
+    enable_calculator: bool = False
+
+    @cached_property
+    def tools(self) -> T.List[BaseTool]:
+        tools = [
+            QueryEngineTool(
+                query_engine=self.query_engine,
+                metadata=ToolMetadata(
+                    name=self.dataset_name.replace("/", "_"),
+                    description=self.dataset_description,
+                ),
+            ),
+        ]
+
+        if self.enable_calculator:
+
+            def add(a: int, b: int):
+                """Add two numbers together"""
+                return a + b
+
+            def subtract(a: int, b: int):
+                """Subtract b from a"""
+                return a - b
+
+            def multiply(a: int, b: int):
+                """Multiply two numbers together"""
+                return a * b
+
+            def divide(a: int, b: int):
+                """Divide a by b"""
+                return a / b
+
+            code_tools = [
+                FunctionTool.from_defaults(fn=fn)
+                for fn in [add, subtract, multiply, divide]
+            ]
+            tools += code_tools
+        return tools
+
+    @property
+    def agent(self) -> AgentRunner:
+        pack = CoAAgentPack(
+            tools=self.tools,
+            llm=self.response_synthesizer_llm,
+        )
+        return pack.agent
+
+
 class Flows(Enum):
     GENERATOR_FLOW = Flow
     RAG_FLOW = RAGFlow
@@ -671,4 +723,5 @@ class Flows(Enum):
     LLAMA_INDEX_CRITIQUE_AGENT_FLOW = CritiqueAgentFlow
     LLAMA_INDEX_SUB_QUESTION_FLOW = SubQuestionRAGFlow
     LLAMA_INDEX_LATS_RAG_AGENT = LATSAgentFlow
+    LLAMA_INDEX_COA_RAG_AGENT = CoAAgentFlow
     RETRIEVER_FLOW = RetrieverFlow
```

## Test the new Flow class

The first thing to do is add a functional test for your flow to ensure basic functionality.

First we add a new `pytest` fixture which builds a simple instance of the `CoAAgentFlow`, and then we add a new test which executes this flow and validates that it completes successfully and issues several types of LlamaIndex events.
Note that the expected event types may be different for your Flow.

```diff
diff --git a/tests/functional/flows/conftest.py b/tests/functional/flows/conftest.py
index c8cb093..53f06d9 100644
--- a/tests/functional/flows/conftest.py
+++ b/tests/functional/flows/conftest.py
@@ -9,6 +9,7 @@ from llama_index.core.storage.docstore.types import BaseDocumentStore
 
 from syftr.agent_flows import LlamaIndexReactRAGAgentFlow
 from syftr.flows import (
+    CoAAgentFlow,
     CritiqueAgentFlow,
     Flow,
     RAGFlow,
@@ -273,6 +274,23 @@ def react_agent_flow_hybrid_hyde_reranker_few_shot(
     ), study_config
 
 
+@pytest.fixture
+def coa_agent_flow(
+    real_sparse_retriever, gpt_4o_mini, rag_template
+) -> T.Tuple[CoAAgentFlow, StudyConfig]:
+    llm, _ = gpt_4o_mini
+    retriever, docstore, study_config = real_sparse_retriever
+    return CoAAgentFlow(
+        retriever=retriever,
+        docstore=docstore,
+        response_synthesizer_llm=llm,
+        template=rag_template,
+        dataset_name=study_config.dataset.name,
+        dataset_description=study_config.dataset.description,
+        enable_calculator=True,
+    ), study_config
+
+
 @pytest.fixture(scope="session")
 def cot_template():
     return get_template("CoT", with_context=False)
diff --git a/tests/functional/flows/test_agentic_rag.py b/tests/functional/flows/test_agentic_rag.py
index 04ac3bb..37c0d39 100644
--- a/tests/functional/flows/test_agentic_rag.py
+++ b/tests/functional/flows/test_agentic_rag.py
@@ -81,3 +81,19 @@ def test_react_agent_flow_hybrid_hyde_reranker_few_shot(
     for question, _ in QA_PAIRS[study_config.dataset.name]:
         flow.generate(question)
     assert llama_debug.get_event_pairs(CBEventType.AGENT_STEP)
+
+
+def test_coa_agent_flow(coa_agent_flow, llama_debug):
+    flow, study_config = coa_agent_flow
+    for question, _ in QA_PAIRS[study_config.dataset.name]:
+        _, _, call_data = flow.generate(question)
+        assert call_data
+        assert llama_debug.get_event_pairs(CBEventType.LLM)
+        assert llama_debug.get_event_pairs(CBEventType.QUERY)
+        assert llama_debug.get_event_pairs(CBEventType.AGENT_STEP)
+        assert llama_debug.get_event_pairs(CBEventType.SYNTHESIZE)
+
+    # test more complex CoA flow
+    _, _, call_data = flow.generate(
+        "what is 123.123*101.101 and what is its product with 12345. then what is 415.151 - 128.24 and what is its product with the previous product?"
+    )
```
