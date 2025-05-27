from syftr.configuration import cfg


def test_model_dump():
    """Test that all fields can be serialized to JSON."""
    cfg.json()
