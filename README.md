# <img src="https://raw.githubusercontent.com/datarobot/syftr/refs/heads/main/docs/syftr-logo.jpeg" alt="syftr Logo" width="200"/> Efficient Search for Pareto-optimal Flows

__syftr__ is an agent optimizer that helps you find the best agentic workflows for a given budget. You bring your own dataset, compose the search space from models and components, and __syftr__ finds the best combination of parameters for your budget. It uses advances in multi-objective Bayesian Optimization and a novel domain-specific "Pareto Pruner" to efficiently sample a search space of agentic and non-agentic flows to estimate a Pareto-frontier (optimal trade-off curve) between accuracy and objectives that compete like cost, latency, throughput.

![syftr](https://raw.githubusercontent.com/datarobot/syftr/refs/heads/main/docs/flowgen_headliner.png)

Please read more details in our [blogpost](https://www.datarobot.com/blog/pareto-optimized-ai-workflows-syftr)
and full [technical paper](https://arxiv.org/abs/2505.20266).

We are excited for what you will discover using __syftr__!

## Libraries and frameworks used

__syftr__ builds on a number of powerful open source projects:

* [Ray](https://www.ray.io/#why-ray) for distributing and scaling search over large clusters of CPUs and GPUs

* [Optuna](https://optuna.org/) for its flexible define-by-run interface (similar to PyTorch’s eager execution) and support for state-of-the-art multi-objective optimization algorithms

* [LlamaIndex](https://www.llamaindex.ai/) for building sophisticated agentic and non-agentic RAG workflows

* [HuggingFace Datasets](https://huggingface.co/docs/datasets/en/index) for fast, collaborative, and uniform dataset interface

* [Trace](https://github.com/microsoft/Trace) for optimizing textual components within workflows, such as prompts

## Installation

Please clone the __syftr__ repo and run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12.7
source .venv/bin/activate
uv sync --extra dev
```

### Required Credentials

__syftr__'s examples require the following credentials:

* Azure OpenAI API key
* Azure OpenAI endpoint URL (`api_url`)
* PostgreSQL server dsn (if no dsn is provided, will use local SQLite)

To enter these credentials, copy [config.yaml.sample](config.yaml.sample) to `config.yaml` and edit the required portions.

## Additional Configuration Options

__syftr__ uses many components including Ray for job scheduling and PostgreSQL for storing results. In this section we describe how to configure them to run __syftr__ successfully.

* The main config file of __syftr__ is `config.yaml`. You can specify paths, logging, database and Ray parameters and many others. For detailed instructions and examples, please refer to [config.yaml.sample](config.yaml.sample).
You can rename this file to `config.yaml` and fill in all necessary details according to your infrastructure.
* You can also configure __syftr__ with environment variables: `export SYFTR_PATHS__ROOT_DIR=/foo/bar`
* When the configuration is correct, you should be able to run [`examples/1-welcome.ipynb`](examples/1-welcome.ipynb) without any problems.
* __syftr__ uses SQLite by default for Optuna storage. The `database.dsn` configuration field can be used to configure any Optuna-supported relational database storage. We recommend Postgres for distributed workloads.

## Quickstart

First, run `make check` to validate your credentials and configuration.
Note that most LLM connections are likely to fail if you have not provided configuration for them.
Next, try the example Jupyter notebooks located in the [`examples`](/examples) directory.
Or directly run a __syftr__ study with user API:

```python
from syftr import api

s = api.Study.from_file("studies/example-dr-docs.yaml")
s.run()
```

Obtaining the results after the study is complete:

```python
s.wait_for_completion()
print(s.pareto_flows)
[{'metrics': {'accuracy': 0.7, 'llm_cost_mean': 0.000258675},
  'params': {'response_synthesizer_llm': 'gpt-4o-mini',
   'rag_mode': 'no_rag',
   'template_name': 'default',
   'enforce_full_evaluation': True}},
   ...
]
```

## Custom LLMs

In addition to the built-in LLMs, you may enable additional OpenAI-API-compatible API endpoints in the ``config.yaml``.

For example:

```yaml
local_models:
  default_api_key: "YOUR_API_KEY_HERE"
  generative:
    - model_name: "microsoft/Phi-4-multimodal-instruct"
      api_base: "http://phi-4-host.com/openai/v1"
      max_tokens: 2000
      context_window: 129072
      is_function_calling_model: true
      additional_kwargs:
        frequency_penalty: 1.0
        temperature: 0.1
    - model_name: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
      api_base: "http://big-vllm-host:8000/v1"
      max_tokens: 2000
      context_window: 129072
      is_function_calling_model: true
      additional_kwargs:
        temperature: 0.6
```

And you may also enable additional embedding model endpoints:

```yaml
local_models:
...
  embedding:
    - model_name: "BAAI/bge-small-en-v1.5"
      api_base: "http://vllmhost:8001/v1"
      api_key: "non-default-value"
      additional_kwargs:
        extra_body:
          truncate_prompt_tokens: 512
    - model_name: "thenlper/gte-large"
      api_base: "http://vllmhost:8001/v1"
      additional_kwargs:
        extra_body:
          truncate_prompt_tokens: 512
```

Models added in the ``config.yaml`` will be automatically added to the default search space, or you can enable them manually for specific flow components.

## Custom Datasets

See detailed instructions [here](docs/datasets.md).

## Citation

If you use this code in your research please cite the following [publication](https://arxiv.org/abs/2505.20266).

```bibtex
@article{syftr2025,
  title={syftr: Pareto-Optimal Generative AI},
  author={Conway, Alexander and Dey, Debadeepta and Hackmann, Stefan and Hausknecht, Matthew and Schmidt, Michael and Steadman, Mark and Volynets, Nick},
  booktitle={Proceedings of the International Conference on Automated Machine Learning (AutoML)},
  year={2025},
}
```

## Contributing

Please read our [contributing guide](/CONTRIBUTING) for details on how to contribute to the project. We welcome contributions in the form of bug reports, feature requests, and pull requests.

Please note we have a [code of conduct](/CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.
