"""Machine-readable JSON export of a Report.

The output is strict JSON: datetimes become ISO-8601 strings, durations become
seconds, dataclasses become dicts (``Insight`` gains its computed ``priority``),
NaN/inf/NaT become null, and numpy / pandas values are converted to plain Python.
"""

from __future__ import annotations

import dataclasses
import json
import math
import numbers
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any

from ..models import Insight, Report

try:  # numpy comes with pandas; it is only needed to recognise its time scalars
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

_MAX_DEPTH = 64


def _np_time(obj: Any) -> Any:
    """numpy timedelta64 -> seconds, datetime64 -> ISO text, NaT -> None (``...`` for anything else).

    timedelta64 subclasses numpy's integer type, so this must run before the Integral branch.
    """
    if _np is None or not isinstance(obj, (_np.timedelta64, _np.datetime64)):
        return ...
    if _np.isnat(obj):
        return None
    if isinstance(obj, _np.timedelta64):
        try:
            return _finite(float(obj / _np.timedelta64(1, "s")))
        except (TypeError, ValueError, OverflowError):  # months / years have no fixed length
            return str(obj)
    try:
        value = obj.astype("datetime64[us]").item()
    except (ValueError, OverflowError, TypeError):
        value = None
    return value.isoformat() if isinstance(value, datetime) else str(_np.datetime_as_string(obj))


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def _key(k: Any) -> str:
    if isinstance(k, str):
        return k
    if isinstance(k, (datetime, date)):
        return k.isoformat()
    if isinstance(k, tuple):
        return "|".join(str(p) for p in k)
    return str(k)


def jsonable(obj: Any, _depth: int = 0) -> Any:
    """Convert ``obj`` into JSON-safe Python (dict/list/str/int/float/bool/None)."""
    if _depth > _MAX_DEPTH:
        return str(obj)
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if type(obj).__name__ in ("NAType", "NaTType"):
        return None
    converted = _np_time(obj)
    if converted is not ...:
        return converted
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return _finite(obj)
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    if isinstance(obj, Decimal):
        return _finite(float(obj)) if obj.is_finite() else None  # float(Decimal("sNaN")) raises
    if isinstance(obj, Enum):
        return jsonable(obj.value, _depth + 1)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {f.name: jsonable(getattr(obj, f.name), _depth + 1) for f in dataclasses.fields(obj)}
        if isinstance(obj, Insight):
            out["priority"] = jsonable(obj.priority, _depth + 1)
        return out
    if isinstance(obj, Mapping):
        return {_key(k): jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).decode("utf-8", errors="replace")
    if hasattr(obj, "columns") and hasattr(obj, "to_dict"):  # pandas DataFrame
        return jsonable(obj.to_dict(orient="records"), _depth + 1)
    if hasattr(obj, "index") and hasattr(obj, "to_dict") and not isinstance(obj, (list, tuple)):  # pandas Series
        return jsonable(obj.to_dict(), _depth + 1)
    if isinstance(obj, numbers.Integral):  # numpy integers
        return int(obj)
    if isinstance(obj, numbers.Real):  # numpy floats
        return _finite(float(obj))
    if hasattr(obj, "tolist"):  # numpy arrays / scalars
        return jsonable(obj.tolist(), _depth + 1)
    if isinstance(obj, (set, frozenset)):
        return [jsonable(v, _depth + 1) for v in sorted(obj, key=repr)]
    if isinstance(obj, (list, tuple)):
        return [jsonable(v, _depth + 1) for v in obj]
    if hasattr(obj, "isoformat"):  # other date-like objects
        return str(obj.isoformat())
    return str(obj)


def to_dict(report: Report) -> dict[str, Any]:
    """The report as JSON-safe nested dicts/lists (same field names as the dataclasses)."""
    return jsonable(report)


def to_json(report: Report, indent: int = 2) -> str:
    """Strict JSON text (no NaN/Infinity); non-ASCII characters are kept as-is."""
    return json.dumps(to_dict(report), indent=indent, ensure_ascii=False, allow_nan=False)
