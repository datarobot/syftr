# syftr - Efficient Search for Pareto-optimal Flows
__syftr__ is an agent optimizer that helps you find the best agentic workflows for your budget. You bring your own dataset, compose the search space from models and components you have access to, and syftr finds the best combination of parameters for your budget.

[Paper](https://arxiv.org) | [Blogpost](https://www.datarobot.com)

### Installation
```
pip install dr-syftr
```

### Runtime configuration
__syftr__ uses many components including Ray for job scheduling and PostgreSQL for storing results. In this section we describe how to configure them to run __syftr__ successfully.
* The main config file of syftr is `config.yaml`. You can specify paths, logging, database and Ray parameters and many others. For detailed instructions and examples, please refer to [config.yaml.sample](config.yaml.sample).
You can rename this file to `config.yaml` and fill in all necessary details according to your infrastructure.
* You can specify sensitive credentials including LLM keys in the `runtime-secrets/` directory.
  For example:
  ```bash
  $ cat runtime-secrets/azure_oai__api_key
  asdfasdfasdf12341234
  ```
* You can also configure syftr with environment variables: `export SYFTR_PATHS__ROOT_DIR=/foo/bar`
* If the configuration is correct, you should be able to run [`examples/1-welcome.ipynb`](examples/1-welcome.ipynb) without any problems.

### Quickstart
Running syftr study with user API after the configuration:
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
Please, refer to [`examples`](/examples) directory for more step-by-step examples.

### Custom LLMs
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


### Developer setup
In order to develop syftr, please clone this repo and run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12.7
source .venv/bin/activate
uv sync --extra dev
# Install pre-commit linter and formatter [optional, recommended]
pre-commit install
```

### Citation
If you use this code in your research please cite the following [publication](https://arxiv.org).

```bibtex
@article{syftr2025,
  title={syftr: Pareto-Optimal Generative AI},
  author={Conway, Alexander and Dey, Debadeepta and Hackmann, Stefan and Hausknecht, Matthew and Schmidt, Michael and Steadman, Mark and Volynets, Nick},
  booktitle={Proceedings of the International Conference on Automated Machine Learning (AutoML)},
  year={2025},
}
```

### Contributing
Please read our [contributing guide](/CONTRIBUTING) for details on how to contribute to the project. We welcome contributions in the form of bug reports, feature requests, and pull requests.

Please note we have a [code of conduct](/CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.
