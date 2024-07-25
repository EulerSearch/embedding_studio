from typing import Callable, Optional, Union

from embedding_studio.embeddings.augmentations.items_set_augmentation_applier import (
    ItemsSetAugmentationApplier,
)
from embedding_studio.embeddings.data.items.items_set import ItemsSet
from embedding_studio.embeddings.data.items.manager import ItemSetManager
from embedding_studio.embeddings.data.preprocessors.text_items_preprocessor import (
    TextItemsDatasetDictPreprocessor,
)
from embedding_studio.embeddings.data.utils.fields_normalizer import (
    DatasetFieldsNormalizer,
)
from embedding_studio.embeddings.splitters.dataset_splitter import (
    ItemsSetSplitter,
)


class TextItemSetManager(ItemSetManager):
    def __init__(
        self,
        field_normalizer: DatasetFieldsNormalizer,
        id_field_name: Optional[str] = None,
        items_set_splitter: Optional[ItemsSetSplitter] = None,
        augmenter: Optional[
            Union[
                ItemsSetAugmentationApplier,
                Callable[[ItemsSet], ItemsSet],
            ]
        ] = None,
        do_augment_test: bool = False,
        do_augmentation_before_preprocess: bool = True,
    ):
        """Manager of ItemsSet ready to be used for fine-tuning of text-to-text model.

        :param field_normalizer: object of DatasetFieldsNormalizer class (unify column names of training data)
        :param id_field_name: specified field name ID (default: None)
        :param items_set_splitter: class to split the items in subparts(default: None)
        :param augmenter: function that add additional augmented rows to an item items_set (default: None)
        :param do_augment_test: do test split augmentation (default: False)
        :param do_augmentation_before_preprocess: do augmentation process before preprocess (default: True)
        """
        super(TextItemSetManager, self).__init__(
            TextItemsDatasetDictPreprocessor(field_normalizer),
            id_field_name,
            items_set_splitter,
            augmenter,
            do_augment_test,
            do_augmentation_before_preprocess,
        )
