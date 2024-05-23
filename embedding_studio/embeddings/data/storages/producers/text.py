from typing import Callable, Optional, Union

from embedding_studio.embeddings.augmentations.items_storage_augmentation_applier import (
    ItemsStorageAugmentationApplier,
)
from embedding_studio.embeddings.data.preprocessors.text_items_preprocessor import (
    TextItemsDatasetDictPreprocessor,
)
from embedding_studio.embeddings.data.storages.producer import (
    ItemStorageProducer,
)
from embedding_studio.embeddings.data.storages.storage import ItemsStorage
from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)
from embedding_studio.embeddings.splitters.dataset_splitter import (
    ItemsStorageSplitter,
)


class TextItemStorageProducer(ItemStorageProducer):
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
    ):
        """Producer of ItemsStorage ready to be used for fine-tuning of text-to-text model.

        :param preprocessor: items dataset dict preprocessing
        :param id_field_name: specified field name ID (default: None)
        :param items_storage_splitter: class to split the items in subparts(default: None)
        :param augmenter: function that add additional augmented rows to an item storage (default: None)
        :param do_augment_test: do test split augmentation (default: False)
        :param do_augmentation_before_preprocess: do augmentation process before preprocess (default: True)
        """
        super(TextItemStorageProducer, self).__init__(
            TextItemsDatasetDictPreprocessor(field_normalizer),
            id_field_name,
            items_storage_splitter,
            augmenter,
            do_augment_test,
            do_augmentation_before_preprocess,
        )
