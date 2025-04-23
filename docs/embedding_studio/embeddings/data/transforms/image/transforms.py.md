# Documentation for `image_transforms`

## Functionality

The `image_transforms` function applies a transformation to images stored in a dataset. It calls the provided transform function, which defaults to `center_crop_transform`, to generate square images of specified size `n_pixels`. If the parameter `del_images` is set to True, the original images will be removed from the dataset.

## Parameters

- **examples**: A dataset containing the images.
- **transform**: A callable transformation function. Default is `center_crop_transform`.
- **n_pixels**: Size (in pixels) of the square image to produce.
- **image_field_name**: The field name containing the original images.
- **pixel_values_name**: The field to store the transformed images.
- **del_images**: A Boolean flag to indicate whether to delete the original images.

## Usage

The purpose of the `image_transforms` function is to process image data in a dataset by applying a customizable transformation.

### Example

```python
result_dataset = image_transforms(
    examples=dataset,
    transform=center_crop_transform,
    n_pixels=224,
    image_field_name="item",
    pixel_values_name="pixel_values",
    del_images=False
)
```