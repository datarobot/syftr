from aiolimiter import AsyncLimiter

from syftr.flows import RAGFlow
from syftr.prompt_optimization import optimize_prompt
from syftr.storage import SyftrQADataset


def test_basic_optimization(tiny_flow: RAGFlow, tiny_dataset: SyftrQADataset):
    raw_data = list(tiny_dataset.iter_examples())
    train = raw_data
    test = raw_data
    rate_limiter = AsyncLimiter(10, 10)
    resulting_prompt = optimize_prompt(
        tiny_flow,
        "gpt-4o-mini",
        "gpt-4o-mini",
        train,
        test,
        rate_limiter,
        num_epochs=3,
    )
    assert resulting_prompt
