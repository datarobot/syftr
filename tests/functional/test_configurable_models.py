from syftr.flows import Flow


def test_loaded_models(configured_models):
    name, llm = configured_models
    llm.complete("hello world")


def test_loaded_models_in_flow(configured_models):
    name, llm = configured_models
    flow = Flow(response_synthesizer_llm=llm)
    response, duration, data = flow.generate("hello world")
    assert response.text
    assert duration
    assert data[0].input_tokens
    assert data[0].output_tokens
