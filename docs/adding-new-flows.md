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
+        tools: T.List[BaseTool] = [
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
index 04ac3bb..bbaa63f 100644
--- a/tests/functional/flows/test_agentic_rag.py
+++ b/tests/functional/flows/test_agentic_rag.py
@@ -81,3 +81,24 @@ def test_react_agent_flow_hybrid_hyde_reranker_few_shot(
     for question, _ in QA_PAIRS[study_config.dataset.name]:
         flow.generate(question)
     assert llama_debug.get_event_pairs(CBEventType.AGENT_STEP)
+
+
+@pytest.mark.flaky(reruns=4, reruns_delay=2)
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
+
+def test_coa_agent_flow_math(coa_agent_flow, llama_debug):
+    flow, study_config = coa_agent_flow
+    response, _, _ = flow.generate(
+        "what is 123.123*101.101 and what is its product with 12345. "
+        "then what is 415.151 - 128.24 and what is its product with the previous product?"
+    )
+    assert str((123.123 * 101.101) * 12345 * (415.151 - 128.24)) in response.text
diff --git a/tests/unit/test_studies.py b/tests/unit/test_studies.py
index 9ff67ee..2775faa 100644
--- a/tests/unit/test_studies.py
+++ b/tests/unit/test_studies.py
@@ -26,6 +26,7 @@ def test_params():
         [
             "additional_context_enabled",
             "additional_context_num_nodes",
+            "coa_enable_calculator",
             "critique_agent_llm",
             "few_shot_embedding_model",
             "few_shot_enabled",
```

## Add CoAAgentFlow to syftr `build_flow` function

This change allows syftr to construct our new flow class when `build_flow` gets `params['rag_mode'] == 'coa_rag_agent'`.

Also note that the `params["coa_enable_calculator"]` key will not be set yet.
This value is still not added to our search space - this is what we'll tackle next.


```diff
diff --git a/syftr/tuner/qa_tuner.py b/syftr/tuner/qa_tuner.py
index ee99b24..73a2e68 100644
--- a/syftr/tuner/qa_tuner.py
+++ b/syftr/tuner/qa_tuner.py
@@ -21,6 +21,7 @@ from ray.util import state
 from syftr.baselines import set_baselines
 from syftr.evaluation import eval_dataset
 from syftr.flows import (
+    CoAAgentFlow,
     CritiqueAgentFlow,
     Flow,
     LATSAgentFlow,
@@ -292,6 +293,23 @@ def build_flow(params: T.Dict, study_config: StudyConfig) -> Flow:
                     enforce_full_evaluation=enforce_full_evaluation,
                     params=params,
                 )
+            case "coa_rag_agent":
+                flow = CoAAgentFlow(
+                    retriever=rag_retriever,
+                    response_synthesizer_llm=response_synthesizer_llm,
+                    docstore=rag_docstore,
+                    template=template,
+                    get_examples=get_qa_examples,
+                    hyde_llm=hyde_llm,
+                    reranker_llm=reranker_llm,
+                    reranker_top_k=reranker_top_k,
+                    additional_context_num_nodes=additional_context_num_nodes,
+                    dataset_name=study_config.dataset.name,
+                    dataset_description=study_config.dataset.description,
+                    enable_calculator=params["coa_enable_calculator"],
+                    enforce_full_evaluation=enforce_full_evaluation,
+                    params=params,
+                )
             case _:
                 raise ValueError(f"Invalid rag_mode: {params['rag_mode']}")
 ```

## Add `coa_rag_agent` to the search space.

First we add the `coa_rag_agent` option into the list of default `RAG_MODES`.
That's all we need to do to enable this new agent to be selected in the search space.

However, since our agent also has special configuration the `enable_calculator` flag), we create a new subspace for the `CoARagAgent` and incorporate it into the main `SearchSpace` class.

We follow the pattern of the `LATSRagAgent` class here, adding the new agent type to the class defaults, distributions, sampling, and cardinality methods.

```diff
diff --git a/syftr/studies.py b/syftr/studies.py
index 9559567..c2fdd2a 100644
--- a/syftr/studies.py
+++ b/syftr/studies.py
@@ -302,6 +302,7 @@ RAG_MODES: T.List[str] = [
     "critique_rag_agent",
     "sub_question_rag",
     "lats_rag_agent",
+    "coa_rag_agent",
     "no_rag",
 ]
 
@@ -874,6 +875,28 @@ class LATSRagAgent(BaseModel, SearchSpaceMixin):
         return card
 
 
+class CoARagAgent(BaseModel, SearchSpaceMixin):
+    enable_calculator: T.List[bool] = Field(
+        default_factory=lambda: [False, True],
+        description="Enable calcuator tools for CoA agent.",
+    )
+
+    def defaults(self, prefix: str = "") -> T.Dict[str, T.Any]:
+        return {
+            f"{prefix}coa_enable_calculator": False,
+        }
+
+    def build_distributions(self, prefix: str = "") -> T.Dict[str, BaseDistribution]:
+        return {
+            f"{prefix}coa_enable_calculator": CategoricalDistribution(
+                self.enable_calculator,
+            ),
+        }
+
+    def get_cardinality(self) -> int:
+        return len(self.enable_calculator)
+
+
 class SearchSpace(BaseModel):
     model_config = ConfigDict(extra="forbid")  # Forbids unknown fields
     non_search_space_params: T.List[str] = Field(
@@ -945,6 +968,10 @@ class SearchSpace(BaseModel):
         default_factory=LATSRagAgent,
         description="Configuration for the LATS RAG agent.",
     )
+    coa_rag_agent: CoARagAgent = Field(
+        default_factory=CoARagAgent,
+        description="Configuration for the CoA RAG agent.",
+    )
     _custom_defaults: ParamDict = {}
 
     def _defaults(self) -> ParamDict:
@@ -962,6 +989,7 @@ class SearchSpace(BaseModel):
             **self.critique_rag_agent.defaults(),
             **self.sub_question_rag.defaults(),
             **self.lats_rag_agent.defaults(),
+            **self.coa_rag_agent.defaults(),
         }
 
     def update_defaults(self, defaults: ParamDict) -> None:
@@ -1009,6 +1037,7 @@ class SearchSpace(BaseModel):
         distributions.update(self.critique_rag_agent.build_distributions())
         distributions.update(self.sub_question_rag.build_distributions())
         distributions.update(self.lats_rag_agent.build_distributions())
+        distributions.update(self.coa_rag_agent.build_distributions())
 
         if params is not None:
             reduced_distributions = {
@@ -1105,7 +1134,11 @@ class SearchSpace(BaseModel):
                 params.update(**self.lats_rag_agent.sample(trial))
             else:
                 params.update(**self.lats_rag_agent.defaults())
-            params.update(**self.lats_rag_agent.sample(trial))
+        elif params["rag_mode"] == "coa_rag_agent":
+            if "coa_rag_agent" in parameters:
+                params.update(**self.coa_rag_agent.sample(trial))
+            else:
+                params.update(**self.coa_rag_agent.defaults())
 
         if few_shot_enabled := trial.suggest_categorical(
             "few_shot_enabled", self.few_shot_enabled
@@ -1144,6 +1177,8 @@ class SearchSpace(BaseModel):
                 sub_card *= self.sub_question_rag.get_cardinality()
             elif rag_mode == "lats_rag_agent":
                 sub_card *= self.lats_rag_agent.get_cardinality()
+            elif rag_mode == "coa_rag_agent":
+                sub_card *= self.coa_rag_agent.get_cardinality()
 
             if True in self.few_shot_enabled:
                 sub_card *= self.few_shot_retriever.get_cardinality()
```

## Final details

Finally, the flow must be added to the `get_flow_name` function.

And nice-to-have, but not strictly required, we add the flow into the baselines, ensuring that it will run during seeding if the agent is part of the search space.
This is not strictly necessary, but helps ensure Optuna sees an instance of the agent before beginning the optimization phase.

## Testing

Finally, we run a study with the new flow enabled to validate that the flow is functional and gets a reasonable level of performance.

To most efficiently test this, we only enable the `coa_rag_agent` RAG mode.

```yaml
---
name: "coa-rag-agent-test-a100"
dataset:
  xname: "financebench_hf"

reuse_study: true

optimization:
  cpus_per_trial: 1
  gpus_per_trial: 0.2
  max_concurrent_trials: 25
  num_eval_samples: 100
  num_retries_unique_params: 10
  num_trials: 100

search_space:
  rag_modes:
  - "coa_rag_agent"
```

After 100 trials with small LLMs, we achieve a top accuracy of 47% on FinanceBench at a few cents per call.
While this isn't a great score, it's to be expected with such a small study and lower-capability LLMs like Claude Haiku and gpt-4o-mini.

At lower accuracies, this flow is Pareto-competitive with other agents.
