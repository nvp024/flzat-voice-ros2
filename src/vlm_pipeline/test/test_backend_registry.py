import pytest

from vlm_pipeline.backends import available_backends, create_backend
from vlm_pipeline.backends.base import BackendConfig


def test_smolvlm_backend_is_created_without_loading_model_libraries() -> None:
    config = BackendConfig(model_id="test-model")
    backend = create_backend("smolvlm2", config)
    assert backend.name == "smolvlm2"
    assert backend.device_description == "not loaded"
    assert config.do_image_splitting is False
    assert "smolvlm2" in available_backends()


def test_qwen2_vl_backend_is_created_without_loading_model_libraries() -> None:
    config = BackendConfig(model_id="Qwen/Qwen2-VL-2B-Instruct")
    backend = create_backend("qwen2_vl", config)
    assert backend.name == "qwen2_vl"
    assert backend.device_description == "not loaded"
    assert config.do_image_splitting is False
    assert "qwen2_vl" in available_backends()


def test_unknown_backend_has_helpful_error() -> None:
    with pytest.raises(
        ValueError,
        match="Available backends: qwen2_vl, smolvlm2",
    ):
        create_backend("qwen-not-added-yet", BackendConfig(model_id="unused"))
