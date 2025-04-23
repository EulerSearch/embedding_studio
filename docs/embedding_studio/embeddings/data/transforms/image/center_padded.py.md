## Documentation for Image Resizing and Padding

### Overview
The following documentation describes methods for resizing and padding images to create square images while preserving their aspect ratios. These methods are essential for uniform image preprocessing in tasks such as normalization and conversion to tensor formats.

### Method: `_resize_and_pad`

#### Functionality
This method resizes an image so that its longest side matches a specified size (`n_px`), preserving the aspect ratio. It then pads the image with zeros to form a square of dimensions `n_px x n_px`. If rounding causes a size mismatch, additional padding is applied to ensure the image matches the desired dimensions.

#### Parameters
- `n_px`: Integer target size for the longest side of the image.

#### Usage
Purpose: The `_resize_and_pad` method is used to preprocess images uniformly before subsequent operations like normalization or tensor conversion.

#### Example
```python
import PIL.Image as Image
from embedding_studio.embeddings.data.transforms.image.center_padded import _resize_and_pad

# Create a resize and pad transform for a 224x224 output image.
transform = _resize_and_pad(224)
img = Image.open("path/to/image.jpg")
result_img = transform(img)
```

### Method: `resize_and_pad`

#### Functionality
This function performs similar operations as `_resize_and_pad`, resizing an image such that its longest side is set to `n_px`. It preserves the aspect ratio and adds symmetric padding, forming a centered square image. Additional padding may be applied if necessary to reach the specified dimensions.

#### Parameters
- `img`: A PIL Image instance that will be resized and padded.
- `n_px`: An integer representing the target size for the longest side, influencing both the resize and padding operations.

#### Usage
Purpose: The `resize_and_pad` function resizes an image based on its longest side and then appends centered padding to create a square image.

#### Example
```python
from PIL import Image
from torchvision.transforms import ToTensor
from embedding_studio.embeddings.data.transforms.image.center_padded import _resize_and_pad

img = Image.open("input.jpg")
resize_pad = _resize_and_pad(256)  # Obtain the transform
result_img = resize_pad(img)
result_tensor = ToTensor()(result_img)
```

### Method: `resize_by_longest_and_pad_transform`

#### Functionality
This transform resizes an image so that its longest side is equal to `n_px`, pads the image with zeros to form a square, converts the image to RGB, and normalizes its colors.

#### Parameters
- `n_px`: The target dimension for the longest side of the image. The output is a square image of size `n_px x n_px`.

#### Usage
Use this transform for preprocessing images while maintaining their aspect ratio. The padding centers the image within a square canvas, preparing it for model input.

#### Example
```python
from torchvision.transforms import Compose
transform = resize_by_longest_and_pad_transform(224)
output = transform(input_image)
```