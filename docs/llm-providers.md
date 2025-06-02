## LLM provider-specific configuration

This page documents the LLM provider-specific configuration options for the ``generative_models`` object in ``config.yaml``.

Individual models are customized in the `generative_models` section of the configuration, for example:

```yaml
azure_oai:
  ...  # Global provider config

generative_models:
  gpt_4o:
    # Provider-specific configuration:
    provider: azure_oai
    deployment_name: my-gpt-4o

    # Common configuration options:
    cost:
      type: tokens             # tokens, characters, or hourly
        input: 1.00            # Cost in USD per million
        output: 2.00           # Cost in USD per million
        # rate: 12.00          # Average cost per hour of inference server, when type is hourly
    
    metadata:
      model_name: gpt-4o
```


---
### Provider: openai_like
There are no common or global settings for `openai_like` models, which each have their own endpoints and credentials.

They are configured as follows:

* **`provider`**: (String, Literal) Must be `openai_like` (for OpenAI-compatible APIs, including self-hosted models via vLLM, TGI, etc.).
* **`model`**: (String, Required) The name of the model as expected by the OpenAI-compatible API (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", or a local model path if the server is configured that way).
* **`api_base`**: (String, HttpUrl, Required) The base URL of the OpenAI-compatible API endpoint (e.g., "http://localhost:8000/v1").
* **`api_key`**: (String, SecretStr, Required) The API key for authenticating with the model's endpoint. Can be placed in a file in `runtime-secrets/generative_models__{your_model_key}__api_key`.
* **`api_version`**: (String, Optional) The API version string, if required by the compatible API. Defaults to `None`.
* **`timeout`**: (Integer, Optional) Timeout in seconds for API requests. Defaults to `120`.
* **`additional_kwargs`**: (Object, Optional) A dictionary of additional keyword arguments to pass to the client. Defaults to an empty dictionary (`{}`).

Here is an example using Together.ai

```yaml
generative_models:
  together_r1:
    provider: openai_like
    model: "deepseek-ai/DeepSeek-R1"
    api_base: "https://api.together.xyz/v1"
    # api_key: <your API key>  # or put a file at runtime-secrets/generative_models__togther_r1__api_key
    cost:
      type: tokens
      input: 7.00
      output: 7.00
    metadata:
      model_name: "deepseek-ai/DeepSeek-R1"
      context_window: 16384
      num_output: 5000
      is_chat_model: true
      is_function_calling_model: false
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
* **`deployment_name`**: (String, Optional) The name of your deployment in Azure OpenAI. Will default to `metadata.model_name`.
* **`api_version`**: (String, Optional) Override the provider's default configuration value for this model.

---
### Provider: azure_ai
There are no common or global settings for `azure_ai` models, which each have their own endpoints and credentials.

They are configured as follows:

* **`provider`**: (String, Literal) Must be `azure_ai` (for Azure AI Completions, e.g., catalog models).
* **`model_name`**: (String, Required) The model name as recognized by Azure AI Completions (e.g., "Llama-3.3-70B-Instruct").
* **`endpoint`**: (String, HttpUrl, Required) The API URL endpoint for this specific model deployment.
* **`api_key`**: (String, SecretStr, Required) The API key for authenticating with the model's endpoint. Can be placed in a file in `runtime-secrets/generative_models__{your_model_key}__api_key`.

---
### Provider: vertex_ai
The top-level `gcp_vertex` config object is used to set the default `project_id`, `region`, and `credentials`:

```yaml
gcp_vertex:
  project_id: "<your-project-id>"
  region: "europe-west1"
  credentials: >                      # Can also put GCP credentials file in runtime-secrets/gcp_vertex__credentials
    {...}
```

Individual models are further customized by the following:

* **`provider`**: (String, Literal) Must be `vertex_ai`.
* **`model`**: (String, Optional) The name of the model on Google Vertex AI (e.g., "gemini-1.5-pro-001", "text-bison@002"). Defaults to `metadata.model_name`.
* **`project_id`**: (String, Optional) The GCP Project ID. If not provided (`None`), it will use the global `gcp_vertex.project_id`.
* **`region`**: (String, Optional) The GCP Region. If not provided (`None`), it will use the global `gcp_vertex.region`.
* **`safety_settings`**: (Object, Optional) A dictionary defining content safety settings. Defaults to predefined `GCP_SAFETY_SETTINGS` (maximally permissive - see `configuration.py`).

---
### Provider: anthropic_vertex
`anthropic_vertex` is used for Anthropic models hosted in Vertex AI. The top-level `gcp_vertex` object is used to provide the default values for `project_id`, `region`, and `credentials`.

Individual models are further customized by the following:

* **`provider`**: (String, Literal) Must be `anthropic_vertex`.
* **`model`**: (String, Required) The name of the Anthropic model available on Vertex AI (e.g., "claude-3-5-sonnet-v2@20241022").
* **`project_id`**: (String, Optional) The GCP Project ID. If not provided (`None`), it will use the global `cfg.gcp_vertex.project_id`. Defaults to `None`.
* **`region`**: (String, Optional) The GCP Region. If not provided (`None`), it will use the global `cfg.gcp_vertex.region`. Defaults to `None`.

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
* **`model`**: (String, Required) The name of the Cerebras model (e.g., "llama3.1-8b").
* **`additional_kwargs`**: (Object, Optional) A dictionary of additional keyword arguments to pass to the Cerebras client. Defaults to an empty dictionary (`{}`).
