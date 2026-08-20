from __future__ import annotations

import importlib
from typing import Type

from vlm_pipeline.backends.base import BackendConfig, VlmBackend


_BACKENDS = {
    "qwen2_vl": "vlm_pipeline.backends.qwen2_vl:Qwen2VlBackend",
    "smolvlm2": "vlm_pipeline.backends.smolvlm2:SmolVlm2Backend",
}


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def create_backend(name: str, config: BackendConfig) -> VlmBackend:
    """Create a registered adapter or a custom ``module:Class`` adapter."""
    normalized = name.strip().lower()
    target = _BACKENDS.get(normalized, name.strip())
    if ":" not in target:
        available = ", ".join(available_backends())
        raise ValueError(
            f"Unknown VLM backend '{name}'. Available backends: {available}. "
            "A custom backend may be supplied as 'python.module:ClassName'."
        )
    module_name, class_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    backend_type: Type[VlmBackend] = getattr(module, class_name)
    backend = backend_type(config)
    if not isinstance(backend, VlmBackend):
        raise TypeError(f"Backend '{target}' does not implement VlmBackend")
    return backend
