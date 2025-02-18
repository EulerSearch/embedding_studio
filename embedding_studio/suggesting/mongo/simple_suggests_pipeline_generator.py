from typing import List

from embedding_studio.models.suggesting import SuggestingRequest
from embedding_studio.suggesting.mongo.pipeline_generator import (
    AbstractPipelineGenerator,
)
from embedding_studio.utils.string_utils import generate_fuzzy_regex


class SimpleSuggestsPipelineGenerator(AbstractPipelineGenerator):
    """
    A simplified MongoDB pipeline generator that primarily matches based on the first chunk (chunk_0).
    It supports exact, case-insensitive, prefix, and fuzzy matches, and sorts results by match type
    and probability.
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
        Build the MongoDB aggregation pipeline to fetch the top matching documents.

        :param request:
            The SuggestingRequest containing the 'next_chunk' text and any relevant context.
        :param top_k:
            The maximum number of documents to return. Defaults to 10.
        :return:
            A list of dictionaries, each representing a stage in the MongoDB aggregation pipeline.
        """
        fuzzy_pattern = generate_fuzzy_regex(request.next_chunk)
        chunks_dict = dict()
        for index in range(self.max_chunks):
            chunks_dict[f"chunk_{index}"] = f"$chunk_{index}"

        match_prefix_part_or_conditions = []
        for index in range(0, len(request.next_chunk)):
            match_prefix_part_or_conditions.append(
                {"search_0": request.next_chunk[: index + 1]}
            )
            match_prefix_part_or_conditions.append(
                {"search_0": request.next_chunk[: index + 1].lower()}
            )
            match_prefix_part_or_conditions.append(
                {"search_0": request.next_chunk[: index + 1].upper()}
            )
            match_prefix_part_or_conditions.append(
                {
                    "search_0": request.next_chunk[: index + 1]
                    .lower()
                    .capitalize()
                }
            )

        pipeline = [
            {"$match": {"n_chunks": {"$gte": 1}}},
            {"$match": {"$or": match_prefix_part_or_conditions}},
            {
                "$addFields": {
                    "match_info": {
                        "type": {
                            "$switch": {
                                "branches": [
                                    {
                                        "case": {
                                            "$eq": [
                                                "$chunk_0",
                                                request.next_chunk,
                                            ]
                                        },
                                        "then": "exact",
                                    },
                                    {
                                        "case": {
                                            "$regexMatch": {
                                                "input": "$chunk_0",
                                                "regex": f"^{request.next_chunk}",
                                                "options": "i",
                                            }
                                        },
                                        "then": "prefix",
                                    },
                                    {
                                        "case": {
                                            "$regexMatch": {
                                                "input": "$chunk_0",
                                                "regex": fuzzy_pattern,
                                                "options": "i",
                                            }
                                        },
                                        "then": "fuzzy",
                                    },
                                ],
                                "default": "none",
                            }
                        },
                        "position": {"$literal": 0},
                        "matched_text": "$chunk_0",
                    }
                }
            },
            {"$match": {"match_info.type": {"$ne": "none"}}},
            {
                "$addFields": {
                    "match_info.rank": {
                        "$switch": {
                            "branches": [
                                {
                                    "case": {
                                        "$eq": ["$match_info.type", "exact"]
                                    },
                                    "then": 1,
                                },
                                {
                                    "case": {
                                        "$eq": ["$match_info.type", "prefix"]
                                    },
                                    "then": 2,
                                },
                                {
                                    "case": {
                                        "$eq": ["$match_info.type", "fuzzy"]
                                    },
                                    "then": 3,
                                },
                            ],
                            "default": 4,
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": {"chunk_0": "$chunk_0", "labels": "$labels"},
                    "original_id": {"$first": "$_id"},
                    "prob": {"$first": "$prob"},
                    "match_info": {"$first": "$match_info"},
                    "chunks": {
                        "$first": {
                            f"chunk_{i}": f"$chunk_{i}"
                            for i in range(self.max_chunks)
                        }
                    },
                }
            },
            {
                "$addFields": {
                    "match_info.length": {
                        "$size": {
                            "$filter": {
                                "input": {"$objectToArray": "$chunks"},
                                "as": "chunk",
                                "cond": {"$ne": ["$$chunk.v", None]},
                            }
                        }
                    },
                }
            },
            {
                "$sort": {
                    "match_info.rank": 1,
                    "prob": -1,
                    "match_info.length": -1,
                }
            },
            {
                "$project": {
                    "_id": "$original_id",
                    "prob": 1,
                    "labels": "$_id.labels",
                    "match_info": 1,
                    "chunks": 1,
                }
            },
            {"$limit": top_k},
        ]

        return pipeline
