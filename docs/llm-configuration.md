## LLM Configuration

These configuration examples should help you configure supported LLMs

### OpenAI API-Compatible Models

```yaml
openai_api_models:
  generative:
    - model_name: gpt-45
      endpoint: https://my-azureopenai.openai.azure.com/api/v1
      token_costs:
        input: 0.005
        output: 0.015
      hourly_cost_usd: 12
```

## Configuring Azure OpenAI

Azure OpenAI models `o1`, `o3-mini`, `gpt-4o`, `gpt-4o-mini`, and `gpt-35-turbo` are supported.
The LLM deployments must use those names.
Use the following example to configure Azure OpenAI, replacing the values with your account information.

```yaml
# config.yaml
azure_oai:
  # Sensitive strings can also be placed in the appropriate file in runtime-secrets/
  # For example, put this key in a file named runtime-secrets/azure_oai__api_key
  api_key: "fakekey0b92ad66fa6e859a98983e61b8f5dc11"
  api_url: "https://my-endpoint.openai.azure.com/"
```

