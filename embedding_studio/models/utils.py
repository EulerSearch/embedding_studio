from embedding_studio.models.delete import FailedItemIdWithDetail
from embedding_studio.models.items_handler import (
    DataItem,
    FailedDataItem,
    ItemProcessingFailureStage,
)


def create_failed_deletion_data_item(
    object_id: str, detail: str
) -> FailedItemIdWithDetail:
    return FailedItemIdWithDetail(
        object_id=object_id,
        detail=detail,
    )


def create_failed_data_item(
    item: DataItem, detail: str, failure_stage: ItemProcessingFailureStage
) -> FailedDataItem:
    return FailedDataItem(
        object_id=item.object_id,
        payload=item.payload,
        item_info=item.item_info,
        detail=detail[-1500:],
        failure_stage=failure_stage,
    )
