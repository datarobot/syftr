## LLM provider-specific configuration

This page documents the LLM provider-specific configuration options for the ``generative_models`` object in ``config.yaml``.

Individual models are customized in the `generative_models` section of the configuration, for example:

```yaml
azure_oai:
  ...  # Global provider config

  # azure_openai Provider Example
  azure_gpt_4o_mini:
    # Common configuration keys
    provider: azure_openai            # Client type to use (required)
    model_name: gpt-4o-mini           # Name of the model to use (required)
    temperature: 0.0                  # LLM temperature setting (optional)
    max_tokens: 2048                  # Max output tokens (optional)
    system_prompt: null               # Custom system prompt (optional)
  
    # Provider-specific configurations
    deployment_name: "gpt-35-turbo"   # Required for Azure OpenAI models
    api_version: "2024-06-01"         # Optional?
    additional_kwargs:                # Add additional parameters to OpenAI request body
      user: syftr
  
    # Cost example - options are the same for all models (required)
    cost:
      type: tokens                    # tokens, characters, or hourly
      input: 1.00                     # Cost in USD per million
      output: 2.00                    # Cost in USD per million
      # rate: 12.00                   # Average cost per hour of inference server, when type is hourly
```

### Common Configuration Options

Some configuration options are common across all LLM providers.

* **`provider`**: (String, Required) Must be one of the supported provider names (`openai_like`, `azure_openai`, `azure_ai`, `vertex_ai`, `anthropic_vertex`, `cerebras`).
* **`model_name`**: (String, Required) The name of the model to use, which should match the model name required by the provider's API.
* **`temperature`**: (Float, Optional) The temperature setting to use for inference. Defaults to 0.
* **`max_tokens`**: (Integer, Optional) The maximum number of output tokens to produce. Defaults to 2048.
* **`system_prompt`**: (String, Optional) A custom system prompt to use for all completions. Defaults to null

#### Cost

The `cost` dictionary is also common across all LLM providers.

* **`type`**: (String, Required) The cost model type; `tokens`, `characters`, or `hourly`.
* **`input`**: (Float) Required if `type` is `tokens` or `characters`. Cost per million input tokens.
* **`output`**: (Float) Required if `type` is `tokens` or `characters`. Cost per million output tokens.
* **`rate`**: (Float) Required if `type` is `hourly`. Average cost per hour of inference server.


---
### Provider: openai_like
There are no global settings for `openai_like` models, which each have their own endpoints and credentials.

They are configured as follows:

* **`provider`**: (String, Literal) Must be `openai_like` (for OpenAI-compatible APIs, including self-hosted models via vLLM, TGI, etc.).
* **`api_base`**: (String, HttpUrl, Required) The base URL of the OpenAI-compatible API endpoint (e.g., "http://localhost:8000/v1").
* **`api_key`**: (String, SecretStr, Required) The API key for authenticating with the model's endpoint. Can be placed in a file in `runtime-secrets/generative_models__{your_model_key}__api_key`.
* **`api_version`**: (String, Optional) The API version string, if required by the compatible API. Defaults to `None`.
* **`context_window`**: (Integer, Optional) The maximum number of input tokens. Defaults to `3900`.
* **`timeout`**: (Integer, Optional) Timeout in seconds for API requests. Defaults to `120`.
* **`additional_kwargs`**: (Object, Optional) A dictionary of additional keyword arguments to pass to the client. Defaults to an empty dictionary (`{}`).
* **`is_chat_model`**: (Boolean, Optional) Whether the model supports multi-turn chat in addition to single completion requests. Defaults to `True`.
* **`is_function_calling_model`**: (Boolean, Optional) Whether the model supports function calling. Defaults to `False`.

Here is an example using Together.ai

```yaml
generative_models:
  together-r1:
    provider: openai_like
    model_name: "deepseek-ai/DeepSeek-R1"
    max_tokens: 5000
    api_base: "https://api.together.xyz/v1"
    # api_key: <your API key>  # or put a file at runtime-secrets/generative_models__togther_r1__api_key
    context_window: 16384
    cost:
      type: tokens
      input: 7.00
      output: 7.00
```

---
### Provider: azure_openai
The top-level `azure_oai` config object is used to set the `api_url` and `api_key`:

```yaml
azure_oai:
  api_url: "https://my-azure-endpoint.openai.azure.com/"
  api_key: "<your-api-key>"
  api_version: "2024-07-18"  # Default value
```

Individual models are further customized by the deployment name and, optionally, the API version to use:

* **`provider`**: (String, Literal) Must be `azure_openai`.
* **`deployment_name`**: (String, Required) The name of your deployment in Azure OpenAI.
* **`api_url`**: (String, HttpUrl, Optional) Overrides `azure_oai.api_url`.
* **`api_key`**: (String, Optional) Overrides `azure_oai.api_key`.
* **`api_version`**: (String, Optional) Overrides `azure_oai.api_version`.
* **`additional_kwargs`**: (Object, Optional) A dictionary of additional keyword arguments to pass to the API. Defaults to an empty dictionary (`{}`).

---
### Provider: azure_ai
There are no common or global settings for `azure_ai` models, which each have their own endpoints and credentials.

They are configured as follows:

* **`provider`**: (String, Literal) Must be `azure_ai` (for Azure AI Completions, e.g., catalog models).
* **`api_url`**: (String, HttpUrl, Required) The API URL endpoint for this specific model deployment.
* **`api_key`**: (String, SecretStr, Required) The API key for authenticating with the model's endpoint. Can be placed in a file in `runtime-secrets/generative_models__{your_model_key}__api_key`.
* **`api_version`**: (String, Optional) API version string to set in requests.
* **`client_kwargs`**: (Object, Optional) A dictionary of additional keyword arguments to pass to the client.

---
### Provider: vertex_ai
The top-level `gcp_vertex` config object is used to set the default `project_id`, `region`, and `credentials`:

```yaml
gcp_vertex:
  project_id: "<your-project-id>"
  region: "europe-west1"
  credentials: >                      # Can also put GCP credentials in a file named runtime-secrets/gcp_vertex__credentials
    {...}
```

Individual models are further customized by the following:

* **`provider`**: (String, Literal) Must be `vertex_ai`.
* **`context_window`**: (Integer, Optional) The maximum number of input tokens. Defaults to `4096`.
* **`additional_kwargs`**: (Object, Optional) A dictionary of additional keyword arguments to pass to the Vertex API. Defaults to an empty dictionary (`{}`).
* **`safety_settings`**: (Object, Optional) A dictionary defining content safety settings. Defaults to predefined `GCP_SAFETY_SETTINGS` (maximally permissive - see `configuration.py`).
* **`project_id`**: (String, Optional) The GCP Project ID. If not provided (`None`), it will use the global `cfg.gcp_vertex.project_id`. Defaults to `None`.
* **`region`**: (String, Optional) The GCP Region. If not provided (`None`), it will use the global `cfg.gcp_vertex.region`. Defaults to `None`.

---
### Provider: anthropic_vertex
`anthropic_vertex` is used for Anthropic models hosted in Vertex AI. The top-level `gcp_vertex` object is used to provide the default values for `project_id`, `region`, and `credentials`.

Individual models are further customized by the following:

* **`provider`**: (String, Literal) Must be `anthropic_vertex`.
* **`project_id`**: (String, Optional) The GCP Project ID. If not provided (`None`), it will use the global `cfg.gcp_vertex.project_id`. Defaults to `None`.
* **`region`**: (String, Optional) The GCP Region. If not provided (`None`), it will use the global `cfg.gcp_vertex.region`. Defaults to `None`.
* **`thinking_dict`**: (Object, Optional) Configure thinking controls for the LLM. See the Anthropic API docs for more details. For example:
```yaml
    thinking_dict:
      type: enabled
      budget_tokens: 16000
```
* **`additional_kwargs`**: (Object, Optional) A dictionary of additional keyword arguments to pass to the API. Defaults to an empty dictionary (`{}`).

---
### Provider: cerebras
The top-level `cerebras` config object is used to set the `api_url` and `api_key`:

```yaml
cerebras:
  api_url: "https://api.cerebras.ai/v1"  # Default value
  api_key: "<your-api-key>"
```

Individual models are further customized by the following:

* **`provider`**: (String, Literal) Must be `cerebras`.
* **`context_window`**: (Integer, Optional) The maximum number of input tokens. Defaults to `3900`.
* **`is_chat_model`**: (Boolean, Optional) Whether the model supports multi-turn chat in addition to single completion requests.
* **`is_function_calling_model`**: (Boolean, Optional) Whether the model supports function calling.
* **`additional_kwargs`**: (Object, Optional) A dictionary of additional keyword arguments to pass to the Cerebras client. Defaults to an empty dictionary (`{}`).
