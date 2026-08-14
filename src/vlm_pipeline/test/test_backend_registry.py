import pytest

from vlm_pipeline.backends import available_backends, create_backend
from vlm_pipeline.backends.base import BackendConfig


def test_smolvlm_backend_is_created_without_loading_model_libraries() -> None:
    backend = create_backend("smolvlm2", BackendConfig(model_id="test-model"))
    assert backend.name == "smolvlm2"
    assert backend.device_description == "not loaded"
    assert "smolvlm2" in available_backends()


def test_unknown_backend_has_helpful_error() -> None:
    with pytest.raises(ValueError, match="Available backends: smolvlm2"):
        create_backend("qwen-not-added-yet", BackendConfig(model_id="unused"))
