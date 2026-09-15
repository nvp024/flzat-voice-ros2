from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Event
from typing import Any


class GenerationCancelled(RuntimeError):
    """Raised when cooperative cancellation stops one backend generation."""


@dataclass(frozen=True)
class BackendConfig:
    """Model-loading options shared by replaceable VLM adapters."""

    model_id: str
    model_revision: str = "main"
    device: str = "auto"
    dtype: str = "auto"
    quantization: str = "none"
    trust_remote_code: bool = False
    local_files_only: bool = False
    do_image_splitting: bool = False
    min_image_pixels: int = 65_536
    max_image_pixels: int = 589_824


@dataclass(frozen=True)
class GenerationRequest:
    """Model-independent request passed to a backend."""

    images: tuple[Any, ...]
    system_prompt: str
    user_prompt: str
    max_new_tokens: int
    cancel_event: Event | None = None


class VlmBackend(ABC):
    """Contract implemented once per model family."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Short backend identifier used in logs and launch parameters."""

    @property
    @abstractmethod
    def device_description(self) -> str:
        """Human-readable device used by the loaded model."""

    @property
    def model_revision(self) -> str:
        """Resolved or requested model revision included in result provenance."""
        return self.config.model_revision

    @abstractmethod
    def load(self) -> None:
        """Load the processor/model exactly once."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> str:
        """Generate a response for already-decoded RGB images."""
