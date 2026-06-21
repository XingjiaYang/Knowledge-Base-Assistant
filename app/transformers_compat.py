from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)
_PATCHED_ALL_TIED_WEIGHTS_KEYS = False


def patch_all_tied_weights_keys() -> None:
    """Keep newer Transformers compatible with older remote model classes."""
    global _PATCHED_ALL_TIED_WEIGHTS_KEYS
    if _PATCHED_ALL_TIED_WEIGHTS_KEYS:
        return

    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        logger.debug("Transformers is unavailable; skipping compatibility patch.")
        _PATCHED_ALL_TIED_WEIGHTS_KEYS = True
        return

    existing = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    if isinstance(existing, property):
        _PATCHED_ALL_TIED_WEIGHTS_KEYS = True
        return

    def get_all_tied_weights_keys(model: Any) -> dict[str, str]:
        value = model.__dict__.get("_all_tied_weights_keys_compat")
        if isinstance(value, dict):
            return value

        legacy_value = getattr(model, "_tied_weights_keys", None) or {}
        if isinstance(legacy_value, dict):
            return {str(key): str(value) for key, value in legacy_value.items()}
        return {str(key): str(key) for key in legacy_value}

    def set_all_tied_weights_keys(model: Any, value: object) -> None:
        if isinstance(value, dict):
            model.__dict__["_all_tied_weights_keys_compat"] = value
            return
        if value is None:
            model.__dict__["_all_tied_weights_keys_compat"] = {}
            return
        model.__dict__["_all_tied_weights_keys_compat"] = {
            str(key): str(key) for key in value
        }

    PreTrainedModel.all_tied_weights_keys = property(
        get_all_tied_weights_keys,
        set_all_tied_weights_keys,
    )
    _PATCHED_ALL_TIED_WEIGHTS_KEYS = True
