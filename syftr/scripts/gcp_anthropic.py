import asyncio
import datetime
import json

import pandas as pd
from anthropic import AnthropicVertex, AsyncAnthropicVertex
from google.oauth2 import service_account
from llama_index.llms.anthropic import Anthropic

from syftr.configuration import cfg


def add_scoped_credentials_anthropic(anthropic_llm: Anthropic) -> Anthropic:
    """Add Google service account credentials to an Anthropic LLM"""
    credentials = service_account.Credentials.from_service_account_info(
        json.loads(cfg.gcp_vertex.credentials.get_secret_value())
    ).with_scopes(["https://www.googleapis.com/auth/cloud-platform"])
    sync_client = anthropic_llm._client
    assert isinstance(sync_client, AnthropicVertex)
    sync_client.credentials = credentials
    anthropic_llm._client = sync_client
    async_client = anthropic_llm._aclient
    assert isinstance(async_client, AsyncAnthropicVertex)
    async_client.credentials = credentials
    anthropic_llm._aclient = async_client
    return anthropic_llm


ANTHROPIC_CLAUDE_SONNET_35 = add_scoped_credentials_anthropic(
    Anthropic(
        model="claude-3-5-sonnet-v2@20241022",
        project_id=str(cfg.gcp_vertex.project_id),
        region="us-east5",
        temperature=0,
    )
)


ANTHROPIC_CLAUDE_HAIKU_35 = add_scoped_credentials_anthropic(
    Anthropic(
        model="claude-3-5-haiku@20241022",
        project_id=str(cfg.gcp_vertex.project_id),
        region="us-east5",
        temperature=0,
    )
)


def estimate_cost_sonnet(call_data: dict) -> dict:
    return {
        "cost_input_tokens": call_data["input_tokens"] * 3.00 / 1e6,
        "cost_output_tokens": call_data["output_tokens"] * 15.00 / 1e6,
        "cost_input_chars": call_data["input_chars"] * 3.00 / 1e6,
        "cost_output_chars": call_data["output_chars"] * 15.00 / 1e6,
    }


def estimate_cost_haiku(call_data: dict) -> dict:
    return {
        "cost_input_tokens": call_data["input_tokens"] * 0.80 / 1e6,
        "cost_output_tokens": call_data["output_tokens"] * 4.00 / 1e6,
        "cost_input_chars": call_data["input_chars"] * 0.80 / 1e6,
        "cost_output_chars": call_data["output_chars"] * 4.00 / 1e6,
    }


async def call_haiku(msg: str) -> dict:
    time = datetime.datetime.now(datetime.UTC)
    response = await ANTHROPIC_CLAUDE_HAIKU_35.acomplete(msg)
    usage = response.raw["usage"]
    data = {
        "input_text": msg,
        "output_text": response.text,
        "input_chars": len(msg),
        "output_chars": len(response.text),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "time": time,
    }
    data.update(**estimate_cost_haiku(data))
    return data


async def call_sonnet(msg: str) -> dict:
    time = datetime.datetime.now(datetime.UTC)
    response = await ANTHROPIC_CLAUDE_SONNET_35.acomplete(msg)
    usage = response.raw["usage"]
    data = {
        "input_text": msg,
        "output_text": response.text,
        "input_chars": len(msg),
        "output_chars": len(response.text),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "time": time,
    }
    data.update(**estimate_cost_sonnet(data))
    return data


async def get_batch(N: int = 10) -> list[dict]:
    tasks = [
        call_sonnet("Write a long essay about the capitol of France?") for _ in range(N)
    ]
    results = await asyncio.gather(*tasks)
    return results


async def run_test(batches: int = 100, batch_size: int = 10):
    try:
        df = pd.read_csv("sonnet-test.csv")
        results = df.to_dict(orient="records")
    except FileNotFoundError:
        results = []
    for _ in range(batches):
        batch_results = await get_batch(N=batch_size)
        results.extend(batch_results)
        df = pd.DataFrame(results)
        df["time"] = pd.to_datetime(df["time"])
        df.to_csv("sonnet-test.csv")
        print(df.describe())
        print(
            df.groupby(pd.Grouper(key="time", freq="T"))[
                [
                    "output_tokens",
                    "output_chars",
                    "cost_output_tokens",
                    "cost_output_chars",
                ]
            ]
            .sum()
            .reset_index()
            .describe()
        )
        print(f"""
Total cost input (tokens): ${df.cost_input_tokens.sum():.4f}
Total cost output (tokens): ${df.cost_output_tokens.sum():.4f}
Total cost input (chars): ${df.cost_input_chars.sum():.4f}
Total cost output (chars): ${df.cost_output_chars.sum():.4f}
""")
        print()


if __name__ == "__main__":
    asyncio.run(run_test())
