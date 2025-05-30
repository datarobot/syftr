import io
import json

import yaml

from syftr.configuration import cfg


def test_model_dump_json():
    """Test that all fields can be serialized to JSON."""
    data = cfg.json()
    json.loads(data)


def test_model_dump_yaml():
    """Test that all fields can be serialized to YAML."""
    buf = io.StringIO()
    data = json.loads(cfg.model_dump_json())
    yaml.safe_dump(data, buf)
    buf.seek(0)
    yaml.safe_load(buf)
