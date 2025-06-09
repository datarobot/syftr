import random
import re
import string
import typing as T

from pydantic import BaseModel, Field


def normalize_text(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


class QAPair(BaseModel):
    question: str
    answer: str
    id: str
    context: T.Union[T.Dict[str, str], T.List[T.Dict]]
    supporting_facts: T.List[T.Any]
    difficulty: str
    qtype: str
    gold_evidence: T.List[str] = Field(
        default_factory=list,
        description="List of gold text snippets that must be retrieved exactly for full recall",
    )


class RandomTrial:
    """A dummy trial class for generating random parameters."""

    def suggest_categorical(self, name, choices):
        return random.choice(choices)

    def suggest_int(self, name, low, high, step=1, log=False):
        return random.randrange(low, high + 1, step)

    def suggest_float(self, name, low, high, step=None, log=False):
        if step:
            num_steps = int((high - low) / step)
            return low + step * random.randint(0, num_steps)
        return random.uniform(low, high)
