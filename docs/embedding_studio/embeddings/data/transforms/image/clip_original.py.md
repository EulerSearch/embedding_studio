## Documentation for Image Transformation Functions

### 1. `convert_image_to_rgb`

#### Functionality
Converts a PIL image to RGB mode. The function takes a PIL Image and returns it in RGB form.

#### Parameters
- `image`: A PIL Image object to be converted.

#### Usage
- **Purpose**: Convert an image to RGB for further processing.

#### Example
```python
from PIL import Image
img = Image.open("path/to/image.jpg")
rgb_img = convert_image_to_rgb(img)
```

---

### 2. `center_crop_transform`

#### Functionality
Transforms an input PIL image by resizing, center cropping, converting to RGB, and normalizing with ImageNet statistics. The transformation then converts the image into a tensor.

#### Parameters
- `n_px`: Size of the target side for resizing and cropping.

#### Usage
Use this function to preprocess images for CLIP models or similar frameworks that require standardized input formats.

#### Example
```python
from PIL import Image
from torchvision.transforms import Compose

# Create a transform with output size 224
transform = center_crop_transform(224)
img = Image.open("my_image.jpg")
normalized_img = transform(img)
```