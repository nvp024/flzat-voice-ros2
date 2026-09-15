"""Hugging Face adapter for the Qwen3-VL model family."""

from __future__ import annotations

from typing import Any

from vlm_pipeline.backends.base import (
    BackendConfig,
    GenerationCancelled,
    GenerationRequest,
    VlmBackend,
)
from vlm_pipeline.cancellation import CancellationStoppingCriteria


class Qwen3VlBackend(VlmBackend):
    """Run Qwen3-VL with the official processor and vision utility."""

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self._torch: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._input_device: Any = None
        self._stopping_criteria_list: Any = None
        self._process_vision_info: Any = None
        self._resolved_revision = config.model_revision

    @property
    def name(self) -> str:
        return "qwen3_vl"

    @property
    def device_description(self) -> str:
        return str(self._input_device or "not loaded")

    @property
    def model_revision(self) -> str:
        return self._resolved_revision

    def load(self) -> None:
        if self._model is not None:
            return
        if self.config.quantization != "none":
            raise ValueError(
                "The qwen3_vl backend currently supports quantization='none' only"
            )
        if self.config.do_image_splitting:
            raise ValueError(
                "The qwen3_vl backend requires do_image_splitting=false"
            )
        if self.config.min_image_pixels < 1:
            raise ValueError("min_image_pixels must be positive")
        if self.config.max_image_pixels < self.config.min_image_pixels:
            raise ValueError(
                "max_image_pixels must be greater than or equal to min_image_pixels"
            )
        try:
            import torch
            import transformers
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, StoppingCriteriaList
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL requires torch, transformers and qwen-vl-utils in "
                "the active Python environment."
            ) from exc

        model_class = getattr(
            transformers,
            "Qwen3VLForConditionalGeneration",
            None,
        )
        if model_class is None:
            model_class = getattr(
                transformers,
                "AutoModelForImageTextToText",
                None,
            )
        if model_class is None:
            raise RuntimeError(
                "The installed transformers version does not provide a "
                "Qwen3-VL-compatible model class."
            )

        self._torch = torch
        self._stopping_criteria_list = StoppingCriteriaList
        self._process_vision_info = process_vision_info
        target_device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch, target_device)
        revision = self.config.model_revision.strip() or "main"
        load_options: dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
            "revision": revision,
        }
        if dtype is not None:
            load_options["dtype"] = dtype
        if self.config.device == "auto" and target_device == "cuda":
            load_options["device_map"] = "auto"

        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
            revision=revision,
        )
        try:
            model = model_class.from_pretrained(
                self.config.model_id,
                **load_options,
            )
        except TypeError as exc:
            if "dtype" not in str(exc) or "dtype" not in load_options:
                raise
            load_options["torch_dtype"] = load_options.pop("dtype")
            model = model_class.from_pretrained(
                self.config.model_id,
                **load_options,
            )
        if "device_map" not in load_options:
            model = model.to(target_device)
        self._model = model.eval()
        self._input_device = self._find_input_device(target_device)
        self._resolved_revision = (
            getattr(getattr(model, "config", None), "_commit_hash", "")
            or revision
        )

    def generate(self, request: GenerationRequest) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("Qwen3-VL backend is not loaded")
        if not request.images:
            raise ValueError("Qwen3-VL requires at least one image")
        system_prompt = request.system_prompt.strip()
        user_prompt = request.user_prompt.strip()
        if not system_prompt or not user_prompt:
            raise ValueError("System and user prompts cannot be empty")
        self._raise_if_cancelled(request)

        image_content = [
            {
                "type": "image",
                "image": image,
                "min_pixels": self.config.min_image_pixels,
                "max_pixels": self.config.max_image_pixels,
            }
            for image in request.images
        ]
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    *image_content,
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        formatted = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        patch_size = getattr(
            getattr(self._processor, "image_processor", None),
            "patch_size",
            None,
        )
        vision_options = {} if patch_size is None else {"image_patch_size": patch_size}
        image_inputs, video_inputs = self._process_vision_info(
            messages,
            **vision_options,
        )
        self._raise_if_cancelled(request)

        output_ids = None
        generated_ids = None
        inputs = None
        try:
            inputs = self._processor(
                text=[formatted],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                do_resize=False,
                return_tensors="pt",
            )
            inputs = inputs.to(self._input_device)
            input_length = inputs["input_ids"].shape[-1]
            generation_options: dict[str, Any] = {
                "max_new_tokens": request.max_new_tokens,
                "do_sample": False,
            }
            if request.cancel_event is not None:
                generation_options["stopping_criteria"] = (
                    self._stopping_criteria_list([
                        CancellationStoppingCriteria(request.cancel_event)
                    ])
                )
            self._raise_if_cancelled(request)
            with self._torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    **generation_options,
                )
            self._raise_if_cancelled(request)
            generated_ids = output_ids[:, input_length:]
            response = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            if not response:
                raise RuntimeError("Qwen3-VL returned an empty response")
            return response
        finally:
            del generated_ids
            del output_ids
            del inputs

    @staticmethod
    def _raise_if_cancelled(request: GenerationRequest) -> None:
        if request.cancel_event is not None and request.cancel_event.is_set():
            raise GenerationCancelled("Qwen3-VL generation was cancelled")

    def _resolve_device(self, torch) -> str:
        requested = self.config.device.strip().lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested not in {"cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        return requested

    def _resolve_dtype(self, torch, device: str):
        requested = self.config.dtype.strip().lower()
        if requested == "auto":
            if device == "cpu":
                return torch.float32
            return (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if requested not in mapping:
            raise ValueError("dtype must be auto, float32, float16, or bfloat16")
        if device == "cpu" and requested == "float16":
            raise ValueError("float16 is not supported for this CPU backend")
        return mapping[requested]

    def _find_input_device(self, fallback: str):
        device = getattr(self._model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        try:
            return next(self._model.parameters()).device
        except (StopIteration, AttributeError):
            return fallback
