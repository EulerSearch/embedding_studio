# Documentation for `AwsS3ImageLoader`

## Functionality

`AwsS3ImageLoader` extends `AwsS3DataLoader` to load image files from AWS S3 buckets. It processes downloaded BytesIO objects and converts them into PIL Image objects. This class is designed to handle image files stored in AWS S3, enabling efficient image loading and processing for machine learning or data analysis tasks.

## Inheritance

This class inherits from `AwsS3DataLoader`, reusing the S3 access and retry configuration while adding image-specific processing using PIL.

## Usage

### Purpose

Retrieve image files from S3 and convert them into PIL images.

### Example 1: Loading an Image from S3

```python
from embedding_studio.data_storage.loaders.cloud_storage.s3.s3_image_loader import AwsS3ImageLoader

loader = AwsS3ImageLoader(
    aws_access_key_id='YOUR_KEY',
    aws_secret_access_key='YOUR_SECRET'
)

# Load and process the image
image = loader.load_items([
    S3FileMeta(bucket='my-images', file='photo1.jpg')
])[0].data
resized = image.resize((300, 200))
```

### Example 2: Converting a BytesIO Stream into a PIL Image

```python
from embedding_studio.data_storage.loaders.cloud_storage.s3.s3_image_loader import AwsS3ImageLoader
from io import BytesIO

# Initialize the S3 image loader
loader = AwsS3ImageLoader(
    aws_access_key_id="YOUR_KEY",
    aws_secret_access_key="YOUR_SECRET"
)

# Open image file and create a BytesIO stream
with open("photo.jpg", "rb") as f:
    file_stream = BytesIO(f.read())

# Convert the stream into a PIL Image object
image = loader._get_item(file_stream)
```

## Method: `AwsS3ImageLoader._get_item`

### Functionality

Processes a downloaded BytesIO file into a PIL Image object for further image manipulation and processing.

### Parameters

- `file`: The downloaded BytesIO object containing image data.

### Usage

- **Purpose** - Convert a raw BytesIO stream into a PIL Image for usage.