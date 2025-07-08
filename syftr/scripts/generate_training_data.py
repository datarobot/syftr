import json
import typing as T
from pathlib import Path

from syftr import storage

TRAINING_DATA_PATH = Path("training_data.jsonl")
# remove the file if it exists
if TRAINING_DATA_PATH.exists():
    TRAINING_DATA_PATH.unlink()

DEFAULT_SYSTEM_CONTENT = "You are a helpful assistant."

DATA_DICT = {
    "type": "chatml",
    "messages": [],
    "source": "unknown",
}


DATASETS: T.List[storage.SyftrQADataset] = [
    storage.CragTask3HF(subset="music"),
    storage.CragTask3HF(subset="sports"),
    storage.CragTask3HF(subset="finance"),
    storage.CragTask3HF(subset="movie"),
    storage.CragTask3HF(subset="open"),
    storage.DRDocsHF(),
    storage.InfiniteBenchHF(),
    storage.MultiHopRAGHF(),
    storage.FinanceBenchHF(),
    storage.BrightHF(subset="biology"),
    storage.BrightHF(subset="stackoverflow"),
    storage.BrightHF(subset="pony"),
    storage.BrightHF(subset="psychology"),
    storage.BrightHF(subset="earth_science"),
    storage.BrightHF(subset="economics"),
    storage.BrightHF(subset="robotics"),
    storage.BrightHF(subset="sustainable_living"),
    # storage.HotPotQAHF(subset="train_hard"),
]


for dataset in DATASETS:
    for example in dataset.iter_examples():
        data = DATA_DICT.copy()
        data["messages"].append(  # type: ignore
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_CONTENT,
            }
        )
        data["messages"].append(  # type: ignore
            {
                "role": "user",
                "content": example.question,
            }
        )
        data["messages"].append(  # type: ignore
            {
                "role": "assistant",
                "content": example.answer,
            }
        )
        data["source"] = dataset.name

        with open("training_data.jsonl", "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    break
