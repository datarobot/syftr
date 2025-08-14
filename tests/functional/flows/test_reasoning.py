def test_basic_generator(reasoning_flow):
    response, duration, call_data = reasoning_flow.generate(
        "What is the capital of France?"
    )
    assert "paris" in str(response).lower()
    assert duration
    assert call_data
