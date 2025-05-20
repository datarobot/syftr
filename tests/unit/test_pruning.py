from syftr.pruning import RuntimePruner


def test_pruner_no_timeout():
    timeout = RuntimePruner(
        num_warmup_steps=0,
        num_total_steps=2,
        eval_timeout=10,
    )
    timeout.report(step=1)
    assert not timeout.should_prune()


def test_pruner_during_warmup():
    timeout = RuntimePruner(
        num_warmup_steps=10,
        num_total_steps=20,
        eval_timeout=0,
    )
    timeout.report(step=1)
    assert not timeout.should_prune()


def test_pruner_timeout():
    timeout = RuntimePruner(
        num_warmup_steps=1,
        num_total_steps=3,
        eval_timeout=0,
    )
    timeout.report(step=2)
    assert timeout.should_prune()
