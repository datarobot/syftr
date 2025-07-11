Guide for Adding LLMs to syftr
=================================

This guide walks through the steps to add a LLM from a variety of providers to the search space of syftr. Currently syftr is built using Llama Index so any LLM client needs to work with the LLM API.

Note that syftr tends to make a fairly-high number of calls so providing a server to queue, batch and run requests is recommended. syftr includes extensive retry and backoff logic to be robust against over-loading a server if it's queue is full.
It also has it's own retry logic so it is suggested to turn off retries with then LLM client packages, e.g. by setting `max_retires=0`

Configuration
-------------

The various settings are set in `configuration.py`. See the OSS readme more infortmation fo how 


Tool Calling and Agentic Flows
----------------------------

syftr includes multiple agentic flows that leverage tool calling, including forced tool calling via the `any` or `required` tool choice parameter. If the used LLM doesn't support this parameter you may get errors.

For example, with vLLM you will need to pass the `--enable-auto-tool-choice` option along with possibly a tool parser and guided decoding backend such as `outlines`. Often, tool call parsing conflicts with enbaling reasoning so keep that in mind when thinking about which LLMs to use in agentic flows.

LLMMetadata
-----------

Llama index, and hence many agentic flows, rely on having the appropriate `LLMMetadata` object defined on the LLM client. Specifically, it is important to ensure that the following fields are set correctly:

* `context_window`: total number of tokens of supported by the model used to, prune retrieved context if larger
* `num_output`: number of output tokens reserved for the output, used in the calculation of how much context to prune
* `is_chat_model`: whether it supports a chat-api of multiple messages and history
* `is_function_calling_model`: whether it supports function calling

Because this information is model-specific, you will see these fields defined for most open source LLMs, either via the client or wrapping it

OpenAI-like
------------

The easiest integration path is via an OpenAI compatible API. For many LLM-as-a-service providers you can specify your credentials and API endpong, e.g. in the `.env` file. In addition, you can sp
This covers adding an LLM hosted on a vLLM server locally. For a simple search space involving one embedding model and one LLM that can run locally on CPU using docker compose. If running on amd64
with a NVIDIA GPU you can use the following docker compose file using the official docker images.

```yaml
services:
  phi4:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    shm_size: "64gb"
    entrypoint:
      - python3
    command: -m vllm.entrypoints.openai.api_server --download-dir=/root/cache/hf --trust-remote-code --enable-auto-tool-choice --tool-call-parser phi4_mini_json --chat-template examples/tool_chat_template_phi4_mini.jinja  --api-key asdf  --model microsoft/Phi-4-mini-instruct --disable-log-requests --enable-prompt-tokens-details  --guided-decoding-backend outlines --max-model-len 108400
    environment:
      - VLLM_CONFIGURE_LOGGING=0
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 20
  bge-small:
    image: vllm/vllm-openai:latest
    ports:
      - "8001:8000"
    shm_size: "64gb"
    entrypoint:
      - python3
    command: -m vllm.entrypoints.openai.api_server --download-dir=/root/cache/hf --trust-remote-code --api-key asdf --task "embed"  --model BAAI/bge-small-en-v1.5 --disable-log-requests
    environment:
      - VLLM_CONFIGURE_LOGGING=0
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 20
```

For Apple systems or running in environments without CUDA a cpu environment can be used. As official environments are not provided, you need to build a docker image

1. Checkout vLLM via git:
```commandline
git clone https://github.com/vllm-project/vllm.git
```
2. Build the docker image
For amd64 CPUs:
```commandline
docker build -f docker/Dockerfile.cpu  --shm-size=4g .
```
For Apple silicion:
```commandline
docker build -f docker/Dockerfile.arm  --shm-size=4g .
```

It may be the case that the compiliation fails if the parallelism is too high. This is controlled by the environment variable: `MAX_JOBS`

```yaml
services:
  phi4:
    image: vllm-cpu-env
    ports:
      - "8000:8000"
    shm_size: "4gb"
    entrypoint:
      - python3
    command: -m vllm.entrypoints.openai.api_server --download-dir=/root/cache/hf --trust-remote-code --enable-auto-tool-choice --tool-call-parser phi4_mini_json --chat-template examples/tool_chat_template_phi4_mini.jinja  --api-key asdf  --model microsoft/Phi-4-mini-instruct --disable-log-requests --enable-prompt-tokens-details  --guided-decoding-backend outlines --max-model-len 108400
    environment:
      - VLLM_CONFIGURE_LOGGING=0
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 20
  bge-small:
    image: vllm-cpu-env
    ports:
      - "8001:8000"
    shm_size: "4gb"
    entrypoint:
      - python3
    command: -m vllm.entrypoints.openai.api_server --download-dir=/root/cache/hf --trust-remote-code --api-key asdf --task "embed"  --model BAAI/bge-small-en-v1.5 --disable-log-requests
    environment:
      - VLLM_CONFIGURE_LOGGING=0
    healthcheck:
      test: [ "CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 20
```

```commandline
docker compose up -d
```

It may be necesarry to mount in the chat template into the image
Anthropic
---------

For anthropic models we use the `Anthropic` class from `llama-index-llms-anthropic`. For Anthropic on GCP, we provided a helper class to add Google credentials, see `add_scoped_credentials_anthropic` in patches.py

E.g.
```python
   Anthropic(
        model="claude-3-5-haiku@20241022",
        project_id=str(cfg.gcp_vertex.project_id),
        region=str(cfg.gcp_vertex.region),
        temperature=0,
    )
```

Azure AI Foundry
----------------

To add a class from Azure AI Foundry for non-OpenAI models, we currently use `AzureAICompletionsModel` integration in the `llama-index-llms-azure-inference` package.  It is necessary to define the LLMMetadata, see examples in llm.py where we use a subclass and then override the metadata property to return the appropriate metadata.

From the Azure AI foundry screen, grab the `key` and `endpoint` that you have set up for authentication. The model name is up to you

E.g.
```python
class AzureAICompletionsModelLlama(AzureAICompletionsModel):
    def __init__(self, credential, model_name, endpoint, temperature=0):
        super().__init__(
            credential=credential,
            model_name=model_name,
            endpoint=endpoint,
            temperature=temperature,
        )

    @property
    def metadata(self):
        return LLMMetadata(
            context_window=120000,
            num_output=1000,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name="Llama-3.3-70B-Instruct",
        )

 AzureAICompletionsModelPhi4(
    credential=cfg.azure_inference_phi4.api_key.get_secret_value(),  # type: ignore
    model_name=str(cfg.azure_inference_phi4.model_name),  # type: ignore
    endpoint=(  # type: ignore
        "https://"
        + str(cfg.azure_inference_llama33.default_deployment)
        + "."
        + str(cfg.azure_inference_llama33.region_name)
        + ".models.ai.azure.com"
    ),
    temperature=0,  # type: ignore
)
```

Azure OpenAI / OpenAI
---------------------

To work with models on Azure OpenAI we use the `AzureOpenAI` class from the `llama-index-llms-azure-openai` package.

It is important to set the `max_retries` parameter to `0` to work with the existing retry logic in syftr. You may also want to set the `user` option to track syftt usage, currently it is set with a default value of `syftr`.

e.g.
```python
AzureOpenAI(
    model="o1",
    deployment_name="o1",
    api_key=cfg.azure_oai.api_key.get_secret_value(),
    azure_endpoint=str(cfg.azure_oai.api_url),
    api_version="2024-12-01-preview",
    temperature=0,
    max_retries=0,
    additional_kwargs={"user": "syftr"},
)
```

For OpenAI models, use the `OpenAI` client from the `llama-index-llms-openai` package.

AWS Bedrock
-----------
To be added.

DataRobot
---------

To be added.

Google Gemini
-------------

To be added.

Google Vertex
-------------

Use the `Vertex` client from the `llama-index-llms-vertex` package.

For credentials, it sufficient to download the json key from a service account and place the json file in the runtime_secrets folder.

Note we found that even on relatively innocent datasets we encoutered content filtering issues, hance we raised the filters above the default level. 

```python
Vertex(
    model="gemini-2.0-flash-001",
    project=cfg.gcp_vertex.project_id,
    credentials=service_account.Credentials.from_service_account_info(GCP_CREDS)
    if GCP_CREDS
    else {},
    temperature=0,
    max_tokens=8000,
    context_window=_scale(1048000),
    max_retries=0,
    safety_settings=GCP_SAFETY_SETTINGS,
    additional_kwargs={},
)
```

HuggingFace
-----------

For embedding models we provide the option to use HuggingFace inference endpoints by specifying the model and name in the configuration. We provide an inital list of those we have tested.


Together AI
-----------

Use the `OpenAILike` client from the `llama-index-llms-openai-like` package 

```python
OpenAILike(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    api_base="https://api.together.xyz/v1",
    api_key=cfg.togetherai.api_key.get_secret_value(),
    api_version=None,  # type: ignore
    max_tokens=2000,
    context_window=_scale(131072),
    is_chat_model=True,
    is_function_calling_model=True,
    timeout=3600,
    max_retries=0,
)
```
Z