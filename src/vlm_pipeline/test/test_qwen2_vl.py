import contextlib
import threading

import pytest

from vlm_pipeline.backends.base import (
    BackendConfig,
    GenerationCancelled,
    GenerationRequest,
)
from vlm_pipeline.backends.qwen2_vl import Qwen2VlBackend


class _InputIds:
    shape = (1, 3)


class _Inputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=_InputIds())
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _OutputIds:
    def __getitem__(self, key):
        assert key == (slice(None), slice(3, None))
        return [[101, 102]]


class _Processor:
    def __init__(self) -> None:
        self.messages = None
        self.images = None

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "formatted prompt"

    def __call__(self, *, text, images, padding, return_tensors):
        assert text == ["formatted prompt"]
        assert padding is True
        assert return_tensors == "pt"
        self.images = images
        return _Inputs()

    @staticmethod
    def batch_decode(_ids, **_kwargs):
        return ["A red cup is visible."]


class _Torch:
    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


class _Model:
    def __init__(self, cancel_token=None) -> None:
        self.cancel_token = cancel_token
        self.generation_options = None
        self.observed_cancellation = False

    def generate(self, **kwargs):
        self.generation_options = kwargs
        if self.cancel_token is not None:
            self.cancel_token.set()
            criteria = kwargs["stopping_criteria"]
            self.observed_cancellation = criteria[0](None, None)
        return _OutputIds()


def _backend(model: _Model, processor: _Processor) -> Qwen2VlBackend:
    backend = Qwen2VlBackend(BackendConfig(model_id="fake"))
    backend._model = model
    backend._processor = processor
    backend._torch = _Torch()
    backend._input_device = "cpu"
    backend._stopping_criteria_list = list
    return backend


def test_qwen2_vl_formats_system_image_and_user_prompt() -> None:
    image = object()
    processor = _Processor()
    model = _Model()
    backend = _backend(model, processor)

    response = backend.generate(
        GenerationRequest(
            images=(image,),
            system_prompt="Use the image.",
            user_prompt="What is visible?",
            max_new_tokens=24,
        )
    )

    assert response == "A red cup is visible."
    assert processor.images == [image]
    assert processor.messages == [
        {"role": "system", "content": "Use the image."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is visible?"},
            ],
        },
    ]
    assert model.generation_options["max_new_tokens"] == 24
    assert model.generation_options["do_sample"] is False


def test_qwen2_vl_passes_cancellation_token_to_generation() -> None:
    token = threading.Event()
    processor = _Processor()
    model = _Model(token)
    backend = _backend(model, processor)

    with pytest.raises(GenerationCancelled):
        backend.generate(
            GenerationRequest(
                images=(object(),),
                system_prompt="system",
                user_prompt="user",
                max_new_tokens=24,
                cancel_event=token,
            )
        )

    assert model.observed_cancellation is True


def test_qwen2_vl_rejects_image_splitting() -> None:
    backend = Qwen2VlBackend(
        BackendConfig(
            model_id="fake",
            do_image_splitting=True,
        )
    )

    with pytest.raises(ValueError, match="requires do_image_splitting=false"):
        backend.load()
