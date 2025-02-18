from typing import List

from embedding_studio.models.suggesting import SuggestingRequest
from embedding_studio.suggesting.mongo.pipeline_generator import (
    AbstractPipelineGenerator,
)
from embedding_studio.utils.string_utils import generate_fuzzy_regex


class FullSuggestsPipelineGenerator(AbstractPipelineGenerator):
    """
    A comprehensive MongoDB pipeline generator that looks for multi-chunk sequences or
    last-chunk matches, then incorporates exact, prefix, or fuzzy matching
    for the upcoming chunk.
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
        Build an advanced MongoDB aggregation pipeline that can handle both
        full-sequence and last-chunk matches, incorporating exact, prefix,
        and fuzzy matching for the next chunk.

        :param request:
            The SuggestingRequest containing the found chunks and the next chunk to match.
        :param top_k:
            The maximum number of results to return.
        :return:
            A list of dictionaries representing each stage in the MongoDB aggregation pipeline.
        """
        fuzzy_pattern = generate_fuzzy_regex(request.next_chunk)
        n_chunks = len(request.found_chunks)
        if n_chunks == 0:
            return []

        # Build the $or conditions for all possible starting positions
        found_chunks = request.found_chunks
        if len(found_chunks) > self.max_chunks:
            found_chunks = request.found_chunks[-self.max_chunks :]

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

        case_branches.append(
            {
                "case": {
                    "$or": [
                        {"$eq": [f"$search_0", request.next_chunk[:1]]},
                        {
                            "$eq": [
                                f"$search_0",
                                request.next_chunk[:1].lower(),
                            ]
                        },
                        {
                            "$eq": [
                                f"$search_0",
                                request.next_chunk[:1].upper(),
                            ]
                        },
                        {
                            "$regexMatch": {
                                "input": "$chunk_0",
                                "regex": fuzzy_pattern,
                                "options": "i",
                            }
                        },
                    ]
                },
                "then": 0,
            },
        )

        # Create an array of chunks for $indexOfArray usage
        [f"$chunk_{i}" for i in range(self.max_chunks)]

        pipeline = [
            # ------------------------------
            # 1) MATCH STAGE
            # ------------------------------
            {
                "$match": {
                    "$or": matching_shift_exprs
                    + match_prefix_part_or_conditions
                }
            },
            # ------------------------------
            # 2) ADD MATCH TYPE INFO (full vs. last + type of match)
            # ------------------------------
            {
                "$addFields": {
                    "match_info": {
                        "position": {
                            "$switch": {
                                "branches": case_branches,
                                "default": -1,
                            },
                        },
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
                    }
                },
            },
            {"$match": {"match_info.type": {"$ne": "none"}}},
            # ------------------------------
            # 3) ADD THE POSITION FIELD
            #    We'll determine the position by checking whether we're "full" or "last"
            #    and using $indexOfArray on either the first found_chunks[0] or the last_chunk.
            # ------------------------------
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
