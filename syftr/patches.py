from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
)

if TYPE_CHECKING:
    pass


def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
    out = {
        **self._model_kwargs,
        **kwargs,
    }
    if out.get("tools"):
        out["tool_choice"] = {"type": "any"}
    return out
