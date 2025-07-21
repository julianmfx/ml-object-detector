"""
ml_object_detector.services.file_inspection.policy
--------------------------------------------------

Centralised, declarative rules for validating uploaded images.
Every endpoint or background job should import *only* load_policy()
and use the returned object.

Call :func:`load_policy` once (after your YAML/INI has been parsed) and reuse
its return value everywhere – the function is memoised so subsequent calls
are free.

Why a function instead of module-level constants?
------------------------------------------------
Because many projects load `.env` files or config INI/YAML **after**
modules are first imported.  A callable lets you compute the values
exactly once, *after* the app is fully configured, and then cache them.
"""

from __future__ import annotations
from functools import lru_cache
from typing import NamedTuple, Any, Mapping

class FilePolicy(NamedTuple):
    """
    File inspection policy.
    """
    allowed_mime: set[str]
    hard_limit_mb: int          # hard upper bound in bytes
    soft_limit_mb: int | None   # None -> no soft limit / confirmation step


# Public loader
@lru_cache(maxsize=1)
def load_policy(cfg: Mapping[str, Any]) -> FilePolicy:
    """Return a cached :class:`FilePolicy` built from *cfg*.

    Parameters
    ----------
    cfg
        A mapping produced by your global configuration loader.
        Must contain the key ``"file_inspection"`` with the following
        structure::

            {
                "file_inspection": {
                    "allowed_mime": ["image/jpeg", ...],
                    "hard_limit_mb": 10,
                    "soft_limit_mb": 5,  # optional
                }
            }
    Required keys:
        allowed_mime   : list[str]
        hard_limit_mb  : int  (>0)
    Optional:
        soft_limit_mb  : int  (>0)  # confirmation step

    Raises
    ------
    KeyError
        If any required setting is missing.
    ValueError
        If values are of the wrong type or non‑positive.
    """

    section = cfg["file_inspection"]
    try:
        allowed_mime_raw = section["allowed_mime"]
        hard_mb = int(section["hard_limit_mb"])
        soft_mb = section.get("soft_limit_mb")

    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(
            f"Missing required configuration key: file_inspection.{missing}"
        ) from exc

    # Basic validation
    if not isinstance(allowed_mime_raw, (list, set, tuple)):
        raise ValueError("file_inspection.allowed_mime must be a list/tuple/set")

    if hard_mb <= 0:
        raise ValueError("file_inspection.hard_limit_mb must be > 0")

    if soft_mb is not None and int(soft_mb) <= 0:
        raise ValueError("file_inspection.soft_limit_mb must be > 0 if set")

    allowed_mime = {
        m.ower().strip() for m in allowed_mime_raw if m.strip()
    }

    return FilePolicy(
        allowed_mime=allowed_mime,
        max_bytes=hard_mb*1024*1024,
        soft_bytes=int(soft_mb*1024*1024 if soft_mb is not None else None)
    )
