"""VLM backend adapters."""

from vlm_pipeline.backends.registry import available_backends, create_backend

__all__ = ["available_backends", "create_backend"]
