from typing import Callable, Optional, Union

from embedding_studio.embeddings.augmentations.items_storage_augmentation_applier import (
    ItemsStorageAugmentationApplier,
)
from embedding_studio.embeddings.data.preprocessors.dict_items_preprocessor import (
    DictItemsDatasetDictPreprocessor,
)
from embedding_studio.embeddings.data.storages.producer import (
    ItemStorageProducer,
)
from embedding_studio.embeddings.data.storages.storage import ItemsStorage
from embedding_studio.embeddings.data.transforms.dict.line_from_dict import (
    get_text_line_from_dict,
)
from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)
from embedding_studio.embeddings.splitters.dataset_splitter import (
    ItemsStorageSplitter,
)


class DictItemStorageProducer(ItemStorageProducer):
    def __init__(
        self,
        field_normalizer: DatasetFieldsNormalizer,
        id_field_name: Optional[str] = None,
        items_storage_splitter: Optional[ItemsStorageSplitter] = None,
        augmenter: Optional[
            Union[
                ItemsStorageAugmentationApplier,
                Callable[[ItemsStorage], ItemsStorage],
            ]
        ] = None,
        do_augment_test: bool = False,
        do_augmentation_before_preprocess: bool = True,
        transform: Callable[[dict], dict] = None,
    ):
        """Producer of ItemsStorage ready to be used for fine-tuning of text-to-text model.
        Input data is a dataset with not single text containing values, and several columns (not ID) are used.

        :param preprocessor: items dataset dict preprocessing
        :param id_field_name: name of field with ID (default: None)
               None value means that every column except ID will be used to format a solid string,
               fields will be sorted by names in descending order.
        :param items_storage_splitter: class to split the items in subparts(default: None)
        :param augmenter: function that add additional augmented rows to an item storage (default: None)
        :param do_augment_test: do test split augmentation (default: False)
        :param do_augmentation_before_preprocess: do augmentation process before preprocess (default: True)
        :param field_normalizer: object to unify column names in DatasetDict, so it can be used in fine-tuning script.
        :param transform: is a function to get a text line out of dict (default: None)
                          None value means usage of get_text_line_from_dict function
        """
        self.transform = (
            transform
            if transform
            else lambda v: get_text_line_from_dict(
                v, order_fields=True, ascending=False
            )
        )
        super(DictItemStorageProducer, self).__init__(
            DictItemsDatasetDictPreprocessor(
                field_normalizer,
                self.transform,
            ),
            id_field_name,
            items_storage_splitter,
            augmenter,
            do_augment_test,
            do_augmentation_before_preprocess,
        )
