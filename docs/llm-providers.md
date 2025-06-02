## LLM provider-specific configuration

This page documents the LLM provider-specific configuration options for ``config.yaml``.

---
### Provider: azure_openai
* **`provider`**: (String, Literal) Must be `azure_openai`.
* **`deployment_name`**: (String, Optional) The name of your deployment in Azure OpenAI. Will default to `metadata.model_name`.
* **`api_version`**: (String, Optional) The Azure OpenAI API version to use (e.g., "2024-07-18"). If not provided, it may default to a global setting in `cfg.azure_oai.api_version` or a client default. Defaults to `None`.

---
### Provider: vertex_ai
* **`provider`**: (String, Literal) Must be `vertex_ai`.
* **`model`**: (String, Optional) The name of the model on Google Vertex AI (e.g., "gemini-1.5-pro-001", "text-bison@002"). Defaults to `metadata.model_name`.
* **`project_id`**: (String, Optional) The GCP Project ID. If not provided (`None`), it will use the global `cfg.gcp_vertex.project_id`. Defaults to `None`.
* **`region`**: (String, Optional) The GCP Region. If not provided (`None`), it will use the global `cfg.gcp_vertex.region`. Defaults to `None`.
    * *Note*: `project_id` and `region` are typically sourced from the global `gcp_vertex` settings but can be specified in `additional_kwargs` if needed for a specific model client.*
* **`safety_settings`**: (Object, Optional) A dictionary defining content safety settings. Defaults to predefined `GCP_SAFETY_SETTINGS` (maximally permissive - see `configuration.py`).

---
### Provider: anthropic_vertex
* **`provider`**: (String, Literal) Must be `anthropic_vertex`.
* **`model`**: (String, Required) The name of the Anthropic model available on Vertex AI (e.g., "claude-3-5-sonnet-v2@20241022").
* **`project_id`**: (String, Optional) The GCP Project ID. If not provided (`None`), it will use the global `cfg.gcp_vertex.project_id`. Defaults to `None`.
* **`region`**: (String, Optional) The GCP Region. If not provided (`None`), it will use the global `cfg.gcp_vertex.region`. Defaults to `None`.

---
### Provider: azure_ai
* **`provider`**: (String, Literal) Must be `azure_ai` (for Azure AI Completions, e.g., catalog models).
* **`model_name`**: (String, Required) The model name as recognized by Azure AI Completions (e.g., "Llama-3.3-70B-Instruct").
* **`endpoint`**: (String, HttpUrl, Required) The API URL endpoint for this specific model deployment.
* **`api_key`**: (String, SecretStr, Required) The API key for authenticating with the model's endpoint. Can be placed in a file in `runtime-secrets/generative_models__{your_model_key}__api_key`.

---
### Provider: cerebras
* **`provider`**: (String, Literal) Must be `cerebras`.
* **`model`**: (String, Required) The name of the Cerebras model (e.g., "llama3.1-8b").
    * *Note: The API key and API URL are typically derived from the global `cfg.cerebras` settings.*
* **`additional_kwargs`**: (Object, Optional) A dictionary of additional keyword arguments to pass to the Cerebras client. Defaults to an empty dictionary (`{}`).

---
### Provider: openai_like
* **`provider`**: (String, Literal) Must be `openai_like` (for OpenAI-compatible APIs, including self-hosted models via vLLM, TGI, etc.).
* **`model`**: (String, Required) The name of the model as expected by the OpenAI-compatible API (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", or a local model path if the server is configured that way).
* **`api_base`**: (String, HttpUrl, Required) The base URL of the OpenAI-compatible API endpoint (e.g., "http://localhost:8000/v1").
* **`api_key`**: (String, SecretStr, Required) The API key for the endpoint (can be a dummy value like "NA" if the server doesn't require one).
* **`api_version`**: (String, Optional) The API version string, if required by the compatible API. Defaults to `None`.
* **`timeout`**: (Integer, Optional) Timeout in seconds for API requests. Defaults to `120`.
* **`additional_kwargs`**: (Object, Optional) A dictionary of additional keyword arguments to pass to the client. Defaults to an empty dictionary (`{}`).


