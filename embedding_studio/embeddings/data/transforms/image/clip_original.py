from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

# Using the mean and std of Imagenet is a common practice. They are calculated based on millions of images.
# If you want to train from scratch on your own dataset, you can calculate the new mean and std.
# Otherwise, using the Imagenet pretrianed model with its own mean and std is recommended.
NORMALIZE_COLOR = Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)


def convert_image_to_rgb(image: Image) -> Image:
    return image.convert("RGB")


def center_crop_transform(n_px: int):
    return Compose(
        [
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            convert_image_to_rgb,
            ToTensor(),
            NORMALIZE_COLOR,
        ]
    )
