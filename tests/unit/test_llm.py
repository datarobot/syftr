def test_tokenizer(tokenizer):
    result = tokenizer("test string")
    assert result
    assert all(isinstance(res, int) for res in result)
