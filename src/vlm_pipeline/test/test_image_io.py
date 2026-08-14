from io import BytesIO

import pytest
from PIL import Image
from sensor_msgs.msg import CompressedImage

from vlm_pipeline.image_io import ImageDecodeError, decode_compressed_images


def _jpeg(width: int = 16, height: int = 12) -> CompressedImage:
    output = BytesIO()
    Image.new("RGB", (width, height), (20, 40, 60)).save(output, format="JPEG")
    message = CompressedImage()
    message.format = "rgb8; jpeg compressed rgb8"
    message.data = output.getvalue()
    return message


def test_decode_returns_detached_rgb_image() -> None:
    images = decode_compressed_images([_jpeg()], 100_000, 1_000)
    assert len(images) == 1
    assert images[0].mode == "RGB"
    assert images[0].size == (16, 12)
    images[0].close()


def test_decode_rejects_invalid_and_oversized_inputs() -> None:
    invalid = CompressedImage()
    invalid.data = b"not an image"
    with pytest.raises(ImageDecodeError, match="not a valid"):
        decode_compressed_images([invalid], 100_000, 1_000)
    with pytest.raises(ImageDecodeError, match="limit"):
        decode_compressed_images([_jpeg(40, 40)], 100_000, 1_000)
    with pytest.raises(ImageDecodeError, match="limit"):
        decode_compressed_images([_jpeg()], 10, 1_000)
