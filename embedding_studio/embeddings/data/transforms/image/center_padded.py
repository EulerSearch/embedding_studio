from PIL import Image
from torchvision.transforms import Compose, Lambda, Pad, Resize, ToTensor

from embedding_studio.embeddings.data.transforms.image.clip_original import (
    NORMALIZE_COLOR,
    convert_image_to_rgb,
)


def _resize_and_pad(n_px: int):
    def resize_and_pad(img: Image) -> Image:
        # Determine the aspect ratio
        aspect: float = img.width / img.height

        # Resize the image so the longest side is n_px
        if aspect > 1:  # width > height
            new_width: int = n_px
            new_height: int = int(n_px / aspect)
        else:
            new_height: int = n_px
            new_width: int = int(n_px * aspect)

        img: Image = Resize((new_height, new_width))(img)

        # Compute padding to make the image square
        pad_width: int = n_px - new_width
        pad_height: int = n_px - new_height

        img: Image = Pad(
            (pad_width // 2, pad_height // 2, pad_width // 2, pad_height // 2),
            fill=0,
        )(img)
        if img.width != n_px or img.height != n_px:
            img: Image = Pad(
                (0, 0, n_px - img.width, n_px - img.height), fill=0
            )(img)

        return img

    return Lambda(resize_and_pad)


def resize_by_longest_and_pad_transform(n_px: int):
    return Compose(
        [
            _resize_and_pad(n_px),
            convert_image_to_rgb,
            ToTensor(),
            NORMALIZE_COLOR,
        ]
    )
