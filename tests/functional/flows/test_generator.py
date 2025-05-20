def test_basic_generator(basic_flow):
    response, duration, call_data = basic_flow.generate(
        "What is the capital of France?"
    )
    assert "paris" in str(response).lower()
    assert duration
    assert call_data
