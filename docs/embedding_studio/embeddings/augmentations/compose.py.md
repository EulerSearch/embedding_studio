## Documentation for `AugmentationsComposition`

### Functionality

The `AugmentationsComposition` class composes multiple augmentation strategies, applying each augmentation sequentially to an input object. It generates a list of augmented objects through a systematic approach, allowing for ordered transformations needed on data objects.

### Motivation

`AugmentationsComposition` was created to simplify the combination of various augmentation techniques. It provides a systematic way to chain operations, enabling users to design flexible data augmentation pipelines.

### Inheritance

This class inherits from `AugmentationWithRandomSelection`, which supports random augmentation selection based on a defined selection size. This feature allows for both deterministic and stochastic augmentation workflows.

### Parameters

- `augmentations`: A list of `AugmentationWithRandomSelection` instances to be applied sequentially.
- `selection_size`: A float defining the proportion of augmentations to select.

### Usage

The `AugmentationsComposition` class is utilized to combine multiple augmentation methods in a sequential pipeline. The method `_raw_transform` within this class applies a sequence of augmentations to an input object, transforming it through a pipeline of augmentation steps, thus resulting in multiple altered versions.

#### Example of Usage

To illustrate, suppose you have two augmentations: rotate and flip. You can create a composition as follows:

```python
rotate_aug = RotateAugmentation(angle=45)
flip_aug = FlipAugmentation()
comp = AugmentationsComposition(
    augmentations=[rotate_aug, flip_aug],
    selection_size=1.0
)
result = comp.transform(image)
```

### Method: `AugmentationsComposition._raw_transform`

#### Functionality

The `_raw_transform` method applies a sequence of augmentations sequentially to an input object. It processes the input through each augmentation call and collects the results, enabling exploration of various transformation outcomes.

#### Parameters

- `object`: The input object to be augmented. It can be of any type; however, passing a list will trigger a warning.

### Example of Transforms

Imagine a pipeline that modifies a text or image. Each augmentation may alter the input by applying filters, rotations, or text modifications. The final output is a list of objects produced by the different stages of the augmentation chain.