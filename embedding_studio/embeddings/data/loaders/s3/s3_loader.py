import io
import logging
import uuid
from typing import Dict, Iterable, List, Optional

import boto3
from datasets import Dataset
from PIL import Image
from pydantic import BaseModel

from embedding_studio.embeddings.data.loaders.data_loader import DataLoader
from embedding_studio.embeddings.data.loaders.s3.item_meta import S3FileMeta

logger = logging.getLogger(__name__)


class AWSS3Credentials(BaseModel):
    role_arn: str
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    external_id: Optional[str] = None


def read_from_s3(client, bucket: str, file: str) -> Image:
    outfile = io.BytesIO()
    try:
        client.download_fileobj(bucket, file, outfile)
        outfile.seek(0)
        return Image.open(outfile)
    except Exception:
        return None


class AWSS3DataLoader(DataLoader):
    def __init__(self, **kwargs):
        super(AWSS3DataLoader, self).__init__(**kwargs)
        self.credentials = AWSS3Credentials(**kwargs)

    def __generate_dataset_from_s3(
        self, files: List[S3FileMeta]
    ) -> Iterable[Dict]:
        logger.info("Connecting to aws s3...")
        task_id: str = str(uuid.uuid4())
        try:
            sts_client = boto3.client(
                "sts",
                aws_access_key_id=self.credentials.aws_access_key_id,
                aws_secret_access_key=self.credentials.aws_secret_access_key,
            )
            if self.credentials.external_id:
                assumed_role_object = sts_client.assume_role(
                    RoleArn=self.credentials.role_arn,
                    RoleSessionName=task_id,
                    ExternalId=self.credentials.external_id,
                )
            else:
                assumed_role_object = sts_client.assume_role(
                    RoleArn=self.credentials.role_arn, RoleSessionName=task_id
                )
            credentials = assumed_role_object["Credentials"]
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=credentials["AccessKeyId"],
                aws_secret_access_key=credentials["SecretAccessKey"],
                aws_session_token=credentials["SessionToken"],
            )
            logger.info("Start downloading data from S3...")
            for val in files:
                image: Image = read_from_s3(s3_client, val.bucket, val.file)
                if image is None:
                    logger.error(
                        f"Unable to download {val.file} from {val.bucket}"
                    )
                    continue
                yield {"item": image, "item_id": val.id}

        except Exception as err:
            logger.error(f"Failed to load dataset from s3: {err}")

    def load(self, items_data: List[S3FileMeta]) -> Dataset:
        return Dataset.from_generator(
            lambda: self.__generate_dataset_from_s3(items_data)
        )
