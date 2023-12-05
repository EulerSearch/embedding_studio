from typing import Dict

from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)

NORMALIZERS: Dict[str, DatasetFieldsNormalizer] = {
    "visual_genome": DatasetFieldsNormalizer("image_id", "image"),
    "visual_genome_objects": DatasetFieldsNormalizer("image_id", "image"),
    "remote_landscapes": DatasetFieldsNormalizer("img_id", "image"),
}
