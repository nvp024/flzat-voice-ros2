import contextlib
import threading

import pytest

from vlm_pipeline.backends.base import (
    BackendConfig,
    GenerationCancelled,
    GenerationRequest,
)
from vlm_pipeline.backends.smolvlm2 import SmolVlm2Backend


class _InputIds:
    shape = (1, 3)


class _Inputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=_InputIds())

    def to(self, _device):
        return self


class _Processor:
    def apply_chat_template(self, *_args, **_kwargs):
        return _Inputs()


class _Torch:
    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


class _CancellingModel:
    def __init__(self, token: threading.Event) -> None:
        self._token = token
        self.observed_stopping_criteria = False

    def generate(self, **kwargs):
        criteria = kwargs["stopping_criteria"]
        self._token.set()
        self.observed_stopping_criteria = criteria[0](None, None)
        return object()


def test_backend_passes_goal_token_to_transformers_generation() -> None:
    token = threading.Event()
    model = _CancellingModel(token)
    backend = SmolVlm2Backend(BackendConfig(model_id="fake"))
    backend._model = model
    backend._processor = _Processor()
    backend._torch = _Torch()
    backend._input_device = "cpu"
    backend._stopping_criteria_list = list
    request = GenerationRequest(
        images=(object(),),
        system_prompt="system",
        user_prompt="user",
        max_new_tokens=48,
        cancel_event=token,
    )

    with pytest.raises(GenerationCancelled):
        backend.generate(request)

    assert model.observed_stopping_criteria is True
