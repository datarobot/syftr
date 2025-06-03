from syftr.flows import Flow


def test_token_counts_basic_generator(basic_flow_all_llms):
    response, duration, call_data = basic_flow_all_llms.generate(
        "What is the capital of France?"
    )
    assert "paris" in str(response.text).lower()
    assert duration
    assert len(call_data) >= 1, call_data
    assert sum(call.cost for call in call_data) > 0, call_data
    assert sum(call.input_tokens for call in call_data) > 0, call_data
    assert sum(call.output_tokens for call in call_data) > 0, call_data


def test_token_counts_configurable_models(configured_models):
    name, llm = configured_models
    flow = Flow(response_synthesizer_llm=llm)
    response, duration, call_data = flow.generate("What is the capital of France?")
    assert "paris" in str(response.text).lower()
    assert duration
    assert len(call_data) >= 1, call_data
    assert sum(call.cost for call in call_data) > 0, call_data
    assert sum(call.input_tokens for call in call_data) > 0, call_data
    assert sum(call.output_tokens for call in call_data) > 0, call_data
