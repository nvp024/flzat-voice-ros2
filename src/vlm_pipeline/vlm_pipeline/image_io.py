from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError
from sensor_msgs.msg import CompressedImage


class ImageDecodeError(ValueError):
    """Raised when a compressed ROS image cannot be safely decoded."""


def decode_compressed_images(
    messages: list[CompressedImage],
    max_total_bytes: int,
    max_image_pixels: int,
) -> tuple[Image.Image, ...]:
    """Decode bounded compressed images to detached RGB PIL images."""
    if not messages:
        raise ImageDecodeError("At least one compressed image is required")
    total_bytes = sum(len(message.data) for message in messages)
    if total_bytes <= 0:
        raise ImageDecodeError("Compressed image data is empty")
    if total_bytes > max_total_bytes:
        raise ImageDecodeError(
            f"Compressed input is {total_bytes} bytes; limit is {max_total_bytes}"
        )

    decoded_images: list[Image.Image] = []
    try:
        for index, message in enumerate(messages):
            try:
                with Image.open(BytesIO(bytes(message.data))) as decoded:
                    decoded.load()
                    if decoded.width * decoded.height > max_image_pixels:
                        raise ImageDecodeError(
                            f"Image {index} has {decoded.width * decoded.height} pixels; "
                            f"limit is {max_image_pixels}"
                        )
                    decoded_images.append(decoded.convert("RGB"))
            except (UnidentifiedImageError, OSError) as exc:
                raise ImageDecodeError(
                    f"Image {index} is not a valid supported compressed image"
                ) from exc
    except Exception:
        for image in decoded_images:
            image.close()
        raise
    return tuple(decoded_images)
