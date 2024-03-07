from typing import Optional

from embedding_studio.embeddings.data.preprocessors.text_items_preprocessor import (
    TextItemsDatasetDictPreprocessor,
)
from embedding_studio.embeddings.data.storages.producer import (
    ItemStorageProducer,
)
from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)


class TextItemStorageProducer(ItemStorageProducer):
    def __init__(
        self,
        field_normalizer: DatasetFieldsNormalizer,
        id_field_name: Optional[str] = None,
    ):
        """Producer of ItemsStorage ready to be used for fine-tuning of text-to-text model.

        :param field_normalizer: object to unify column names in DatasetDict, so it can be used in fine-tuning script.
        :param id_field_name: name of field with ID (default: None)
        """
        super(TextItemStorageProducer, self).__init__(
            TextItemsDatasetDictPreprocessor(field_normalizer),
            id_field_name,
        )
