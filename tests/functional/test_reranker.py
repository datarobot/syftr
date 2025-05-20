def test_reranker_flow_rag(hotpot_reranker_flow, correctness_evaluator):
    question = "Who is Adam Collis?"
    response, duration, call_data = hotpot_reranker_flow.generate(question)

    assert duration > 0
    assert call_data
    eval_result = correctness_evaluator.evaluate(
        query=question,
        response=response.text,
        reference="American Filmmaker and Actor",
    )
    assert eval_result
