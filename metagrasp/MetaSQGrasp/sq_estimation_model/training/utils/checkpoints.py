"""Checkpoint naming helpers for SQ / ExtSQ estimation models."""

from __future__ import annotations

SUPPORTED_MODEL_ARCHS = frozenset({"sq_esti_model", "extsq_esti_model"})


def best_checkpoint_name(model_arch: str) -> str:
    """Return the canonical best-checkpoint filename for a model architecture."""
    if model_arch not in SUPPORTED_MODEL_ARCHS:
        raise ValueError(
            f"Unsupported model architecture {model_arch!r}. "
            f"Expected one of: {sorted(SUPPORTED_MODEL_ARCHS)}"
        )
    return f"{model_arch}_best.pkl"
