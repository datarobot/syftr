# Debugging and LLM Tracing

syftr supports debugging and LLM tracing of individual flows through the `build_flow.py` script which instantiates and evaluates a single flow of interest, optionally with instrumentation and tracing to gain visibility into the question, prompt, and LLM response.

This tutorial demonstrates how to use `build_flow.py` to inspect a simple flow with instrumentation. First ensure that all the necessary dependencies are installed by running `uv sync --extra dev`.

First in your `config.yaml` uncomment the instrumentation section:
```
instrumentation:
  tracing_enabled: true
  otel_endpoint: http://localhost:6006/v1/traces
```

Next start the Phoenix server by running `phoenix serve`. Then open the Phoenix webpage, typically located at `http://0.0.0.0:6006`.

Finally, in a different terminal, we'll now run `build_flow.py` as follows:
```
python syftr/scripts/build_flow.py --study-config-path studies/example-dr-docs.yaml --interactive --instrumentation --flow-json '{"few_shot_enabled": false, "rag_mode": "no_rag", "template_name": "concise", "response_synthesizer_llm": "gpt-4o-std"}'
```

This will instantiate the flow described in the json and evaluate it against the dataset used in the study-config, in this case `drdocs`. The `--interactive` flag will set a breakpoint at each question and response, and the `--instrumentation` flag will send traces to Phoenix.

Refreshing the Phoenix webpage should now show traces for each QA Pair, response, and any additional internal flow information that is logged.