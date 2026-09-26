"""Validate browser image bytes and create request-scoped, private image files."""

import base64
import binascii
import io
import shutil
import tempfile
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_UPLOAD_TOTAL = 32 * 1024 * 1024
MAX_IMAGES = 8
MAX_IMAGE_PIXELS = 16_000_000
FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP", "image/gif": "GIF"}


def materialize_images(request):
    images = request.get("images", [])
    if not isinstance(images, list) or len(images) > MAX_IMAGES:
        raise ValueError("images must be a list of at most 8 uploaded files.")
    if not images:
        return request, None
    # mkdtemp creates a private directory; client filenames never become paths.
    directory = Path(tempfile.mkdtemp(prefix="interndecision-upload-"))
    paths, total, normalized_total = [], 0, 0
    try:
        for index, item in enumerate(images):
            if not isinstance(item, dict) or not isinstance(item.get("data"), str):
                raise ValueError("Upload image bytes; local paths and remote URLs are not accepted.")
            data = item["data"]
            if len(data) > 4 * ((MAX_UPLOAD_BYTES + 2) // 3) + 128:
                raise ValueError("Each image is limited to 12 MB.")
            if data.startswith("data:"):
                header, separator, encoded = data.partition(",")
                if not separator or not header.endswith(";base64"):
                    raise ValueError("Images must use base64 data URLs.")
                content_type = header[5:].split(";")[0].lower()
            else:
                encoded, content_type = data, str(item.get("type", "")).lower()
            if content_type not in FORMATS:
                raise ValueError("Supported image formats: JPEG, PNG, WebP and static GIF.")
            try:
                payload = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Invalid image base64.") from exc
            if not payload or len(payload) > MAX_UPLOAD_BYTES:
                raise ValueError("Each image must be between 1 byte and 12 MB.")
            total += len(payload)
            if total > MAX_UPLOAD_TOTAL:
                raise ValueError("Combined image upload is limited to 32 MB.")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(payload)) as source:
                        if source.format != FORMATS[content_type]:
                            raise ValueError("Image content does not match its declared format.")
                        if source.width * source.height > MAX_IMAGE_PIXELS:
                            raise ValueError("Each image is limited to 16 million pixels.")
                        if getattr(source, "n_frames", 1) != 1:
                            raise ValueError("Animated images are not supported; upload a still image.")
                        source.load()
                        # Apply camera orientation, strip metadata and preserve pixel resolution.
                        image = ImageOps.exif_transpose(source).convert("RGB")
                        image.info.clear()
                        path = directory / f"{index:02d}.png"
                        image.save(path, format="PNG")
                        image.close()
            except (
                UnidentifiedImageError,
                OSError,
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
            ) as exc:
                raise ValueError("Invalid, incomplete or oversized image.") from exc
            normalized_total += path.stat().st_size
            if normalized_total > MAX_UPLOAD_TOTAL:
                raise ValueError("Decoded images exceed 32 MB; upload smaller images.")
            paths.append(str(path))
        return {**request, "images": paths}, directory
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
