"""Validate and normalize uploaded photographs."""

import io

from PIL import Image, ImageOps, UnidentifiedImageError


def product_references(product):
    """Preserve compatibility with single-photo catalogue entries."""
    references = product.get("references") or [{"id": None, "image_path": product["image_path"]}]
    return list(references)


def read_image(upload):
    if upload is None:
        raise ValueError("Please upload a photograph.")
    try:
        with Image.open(io.BytesIO(upload.getvalue())) as original:
            return ImageOps.exif_transpose(original).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("The uploaded file is not a readable image.") from error
