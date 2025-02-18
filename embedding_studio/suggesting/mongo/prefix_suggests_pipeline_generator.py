from typing import List

from embedding_studio.models.suggesting import SuggestingRequest
from embedding_studio.suggesting.mongo.pipeline_generator import (
    AbstractPipelineGenerator,
)


class PrefixSuggestsPipelineGenerator(AbstractPipelineGenerator):
    """
    A MongoDB-based pipeline generator that focuses on prefix-based matches.
    If only one chunk is found, it searches for that chunk in multiple
    positions. Otherwise, it tries matching the entire sequence from
    position 0 or only the last chunk at position 0.
    """

    def __init__(
        self,
        max_chunks: int = 20,
    ):
        """
        Initialize the Pipeline Generator.

        :param max_chunks: Maximum number of chunks that each document can have.
        """
        super().__init__()
        self.max_chunks = max_chunks

    def generate_pipeline(
        self, request: SuggestingRequest, top_k: int = 10
    ) -> List[dict]:
        """
        Build the MongoDB aggregation pipeline for prefix-based suggestions.

        :param request:
            The request object that contains the found chunks and other context.
        :param top_k:
            The maximum number of suggestions to return. Defaults to 10.
        :return:
            A list of dictionary stages that represent the MongoDB aggregation pipeline.
        """
        chunks_dict = dict()
        chunks_list = list()
        for index in range(self.max_chunks):
            chunks_dict[f"chunk_{index}"] = f"$chunk_{index}"
            chunks_list.append(f"$chunk_{index}")

        found_chunks = request.found_chunks
        if len(found_chunks) > self.max_chunks:
            found_chunks = request.found_chunks[-self.max_chunks :]

        n_chunks = len(found_chunks)
        if n_chunks == 0:
            return []

        # For multiple chunks, match either sequence or last chunk at start
        matching_shift_exprs = []
        for index in range(len(found_chunks)):
            matching_shifts = list()
            for pos in range(len(found_chunks) - index):
                current_matching_shifts = []
                current_matching_shifts.append(
                    {
                        f"chunk_{pos}": found_chunks[index + pos],
                    }
                )
                current_matching_shifts.append(
                    {
                        f"chunk_{pos}": found_chunks[index + pos].lower(),
                    }
                )
                current_matching_shifts.append(
                    {
                        f"chunk_{pos}": found_chunks[index + pos].upper(),
                    }
                )
                current_matching_shifts.append(
                    {
                        f"chunk_{pos}": found_chunks[index + pos]
                        .lower()
                        .capitalize(),
                    }
                )

                matching_shifts.append(
                    {
                        "$or": current_matching_shifts,
                    }
                )

            matching_shift_exprs.append(
                {
                    "$and": [
                        {"n_chunks": {"$gte": len(matching_shifts)}},
                        *matching_shifts,
                    ]
                }
            )

        case_branches = []
        for index in range(len(found_chunks)):
            case_branches.append(
                {
                    "case": {
                        "$or": [
                            {"$eq": [f"$chunk_0", found_chunks[index]]},
                            {
                                "$eq": [
                                    f"$chunk_0",
                                    found_chunks[index].lower(),
                                ]
                            },
                            {
                                "$eq": [
                                    f"$chunk_0",
                                    found_chunks[index].upper(),
                                ]
                            },
                            {
                                "$eq": [
                                    f"$chunk_0",
                                    found_chunks[index].lower().capitalize(),
                                ]
                            },
                        ]
                    },
                    "then": len(found_chunks) - index - 1,
                },
            )

        pipeline = [
            {"$match": {"$or": matching_shift_exprs}},
            {
                "$addFields": {
                    "match_length": {
                        "$switch": {"branches": case_branches, "default": -1},
                    }
                }
            },
            {
                "$group": {
                    "_id": {"chunk_0": "$chunk_0", "labels": "$labels"},
                    "original_id": {"$first": "$_id"},
                    "prob": {"$first": "$prob"},
                    "match_length": {"$first": "$match_length"},
                    "chunks": {"$first": chunks_dict},
                }
            },
            # Sort by match type and probability
            {"$sort": {"prob": -1, "match_length": -1}},
            # Project results
            {
                "$project": {
                    "_id": "$original_id",
                    "prob": "$prob",
                    "labels": "$_id.labels",
                    "match_info": {
                        "type": {"$literal": "exact"},
                        "length": {"$literal": n_chunks},
                        "position": "$match_length",
                    },
                    "chunks": 1,
                }
            },
            {"$limit": top_k},
        ]

        return pipeline
