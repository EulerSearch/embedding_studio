from typing import Optional

from embedding_studio.embeddings.data.preprocessors.image_items_preprocessor import (
    ImageItemsDatasetDictPreprocessor,
)
from embedding_studio.embeddings.data.storages.producer import (
    ItemStorageProducer,
)
from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)


class CLIPItemStorageProducer(ItemStorageProducer):
    def __init__(
        self,
        field_normalizer: DatasetFieldsNormalizer,
        id_field_name: Optional[str] = None,
    ):
        """Producer of ItemsStorage ready to be used for fine-tuning of CLIP model.

        :param field_normalizer: object to unify column names in DatasetDict, so it can be used in fine-tuning script.
        :param id_field_name: name of field with ID (default: None)
        """
        super(CLIPItemStorageProducer, self).__init__(
            ImageItemsDatasetDictPreprocessor(field_normalizer, 224),
            id_field_name,
        )
