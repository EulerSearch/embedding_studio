from typing import Callable, Optional

from embedding_studio.embeddings.data.preprocessors.dict_items_preprocessor import (
    DictItemsDatasetDictPreprocessor,
)
from embedding_studio.embeddings.data.storages.producer import (
    ItemStorageProducer,
)
from embedding_studio.embeddings.data.transforms.dict.line_from_dict import (
    get_text_line_from_dict,
)
from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)


class DictItemStorageProducer(ItemStorageProducer):
    def __init__(
        self,
        field_normalizer: DatasetFieldsNormalizer,
        id_field_name: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        """Producer of ItemsStorage ready to be used for fine-tuning of text-to-text model.
        Input data is a dataset with not single text containing values, and several columns (not ID) are used.

        :param field_normalizer: object to unify column names in DatasetDict, so it can be used in fine-tuning script.
        :param id_field_name: name of field with ID (default: None)
               None value means that every column except ID will be used to format a solid string,
               fields will be sorted by names in descending order.
        """
        super(DictItemStorageProducer, self).__init__(
            DictItemsDatasetDictPreprocessor(
                field_normalizer,
                lambda v: transform(v)
                if transform
                else get_text_line_from_dict(
                    v, order_fields=True, ascending=False
                ),
            ),
            id_field_name,
        )
