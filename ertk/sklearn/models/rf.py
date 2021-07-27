from typing import Any, Dict, Sequence


def default_rf_param_grid() -> Dict[str, Sequence[Any]]:
    return {"n_estimators": [100, 250, 500], "max_depth": [None, 10, 20, 50]}
