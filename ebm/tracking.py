import json
from typing import Any, Dict


def to_mlflow_params(params: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in params.items():
        log_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, (str, int, float, bool)):
            out[log_key] = value
        elif value is None:
            out[log_key] = "None"
        else:
            out[log_key] = json.dumps(value)
    return out
