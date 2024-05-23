import logging
from io import BytesIO

logger = logging.getLogger(__name__)


def download_s3_object_to_memory(
    s3_client, bucket_name: str, object_key: str
) -> BytesIO:
    # Create an S3 client
    try:
        # Download the object into memory
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        data = response["Body"].read()

        # Use BytesIO to create an in-memory file-like object
        in_memory_file = BytesIO(data)

        # Now 'in_memory_file' contains the object data in memory
        return in_memory_file
    except Exception as e:
        logger.exception(f"Error downloading object: {e}")


def list_objects_by_path(s3_client, bucket_name: str, path: str):
    try:
        # List objects with a specific prefix (path)
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=path)

        # Extract object information from the response
        objects = response.get("Contents", [])

        for obj in objects:
            logger.info(f"Object Key: {obj['Key']}")

    except Exception as e:
        logger.exception(f"Error listing objects: {e}")
