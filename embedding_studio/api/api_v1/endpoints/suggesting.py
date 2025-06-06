from fastapi import APIRouter, HTTPException, Query, status

from embedding_studio.api.api_v1.schemas.suggesting import (
    GetSuggestionsRequest,
    GetSuggestionsResponse,
    ListPhrasesRequest,
    ListPhrasesResponse,
    Span,
    SuggestingPhrase,
    SuggestingPhrasesAddDomainsRequest,
    SuggestingPhrasesAddingRequest,
    SuggestingPhrasesAddingResponse,
    SuggestingPhrasesAddLabelsRequest,
    SuggestingPhrasesAdjustProbabilityRequest,
    SuggestingPhrasesRemoveAllDomainsRequest,
    SuggestingPhrasesRemoveAllLabelsRequest,
    SuggestingPhrasesRemoveDomainsRequest,
    SuggestingPhrasesRemoveLabelsRequest,
    Suggestion,
)
from embedding_studio.context.app_context import context
from embedding_studio.utils.redis_utils import ft_unescape_punctuation
from embedding_studio.utils.string_utils import combine_chunks

router = APIRouter()


# TODO: add LRU-cache


@router.post(
    "/get-top-k",
    response_model=GetSuggestionsResponse,
    status_code=status.HTTP_200_OK,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_suggestions(body: GetSuggestionsRequest):
    suggestion_request = (
        context.suggester.phrases_manager.convert_phrase_to_request(
            phrase=body.phrase, domain=body.domain
        )
    )
    raw_suggestions = context.suggester.get_topk_suggestions(
        request=suggestion_request, top_k=body.top_k
    )

    suggestions = []
    for raw_suggestion in raw_suggestions:
        # Build the suggestion body from the suggestion’s chunks.
        suggestion_body = combine_chunks(
            [ft_unescape_punctuation(chunk) for chunk in raw_suggestion.chunks]
        )

        # Get the list of prefix chunks.
        prefix_chunks = raw_suggestion.prefix_chunks

        # Calculate which found chunk we should use based on the number of prefix chunks.
        matched_index = len(suggestion_request.chunks.found_chunks) - len(
            prefix_chunks
        )
        # Clamp matched_index to ensure it's within the bounds of found_chunk_spans.
        matched_index = max(
            0,
            min(
                matched_index,
                len(suggestion_request.spans.found_chunk_spans) - 1,
            ),
        )

        # Determine the span that indicates where the suggestion should split.
        if suggestion_request.chunks.next_chunk:
            span = suggestion_request.spans.next_chunk_span

            if len(suggestion_request.spans.found_chunk_spans) == 0:
                matching_span_start = (
                    suggestion_request.spans.next_chunk_span.start
                )
            else:
                # Full matching span: from the start of the matched found chunk to the end of the next chunk.
                matching_span_start = (
                    suggestion_request.spans.found_chunk_spans[
                        matched_index
                    ].start
                )

            matching_span_end = suggestion_request.spans.next_chunk_span.end

        elif matched_index < len(suggestion_request.spans.found_chunk_spans):
            span = suggestion_request.spans.found_chunk_spans[matched_index]
            matching_span_start = span.start
            matching_span_end = span.end

        else:
            span = Span(start=0, end=0)
            matching_span_start = 0
            matching_span_end = 0

        if matching_span_start == 0 and matching_span_end == 0:
            combined_prefix = combine_chunks(
                [ft_unescape_punctuation(chunk) for chunk in prefix_chunks]
            )
            suggestions.append(
                Suggestion(
                    prefix="",
                    postfix=combined_prefix,
                    matching_span=Span(
                        start=matching_span_start,
                        end=matching_span_end,
                    ),
                    match_type=raw_suggestion.match_type,
                    prob=raw_suggestion.prob,
                    labels=raw_suggestion.labels,
                )
            )

        else:

            # Combine all prefix chunks except the last one.
            combined_prefix = combine_chunks(
                [
                    ft_unescape_punctuation(chunk)
                    for chunk in prefix_chunks[:-1]
                ]
            )
            last_chunk = ft_unescape_punctuation(prefix_chunks[-1])

            # Calculate the split index in the last prefix chunk.
            split_index = span.end - len(combined_prefix)
            split_index = max(0, min(len(last_chunk), split_index))

            # Split the last prefix chunk.
            prefix_part = last_chunk[:split_index]
            postfix_prefix_part = last_chunk[split_index:]

            # Build the aligned prefix and postfix.
            suggestion_prefix = combined_prefix + " " + prefix_part
            suggestion_postfix = postfix_prefix_part + " " + suggestion_body

            suggestions.append(
                Suggestion(
                    prefix=suggestion_prefix,
                    postfix=suggestion_postfix,
                    matching_span=Span(
                        start=matching_span_start,
                        end=matching_span_end,
                    ),
                    match_type=raw_suggestion.match_type,
                    prob=raw_suggestion.prob,
                    labels=raw_suggestion.labels,
                )
            )

    return GetSuggestionsResponse(suggestions=suggestions)


@router.post(
    "/phrases/add",
    response_model=SuggestingPhrasesAddingResponse,
    status_code=status.HTTP_200_OK,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def add_phrase(body: SuggestingPhrasesAddingRequest):
    """
    Insert multiple SuggestingPhrase documents into the system.
    Return IDs for successfully inserted phrases and any errors that occur.
    """
    phrase_ids = context.suggester.phrases_manager.add(body.phrases)

    # Placeholder logic
    return SuggestingPhrasesAddingResponse(phrase_ids=phrase_ids)


@router.delete(
    "/phrases/delete/{phrase_id}",
    status_code=status.HTTP_200_OK,
)
def delete_phrase(phrase_id: str):
    """
    Delete a phrase by its ID.
    """
    context.suggester.phrases_manager.delete([phrase_id])


@router.post(
    "/phrases/add-labels",
    status_code=status.HTTP_200_OK,
)
def add_labels(body: SuggestingPhrasesAddLabelsRequest):
    """
    Add labels to a phrase.
    """
    try:
        context.suggester.phrases_manager.add_labels(
            phrase_id=body.phrase_id, labels=body.labels
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Phrase {body.phrase_id} not found",
        )


@router.post(
    "/phrases/remove-labels",
    status_code=status.HTTP_200_OK,
)
def remove_labels(body: SuggestingPhrasesRemoveLabelsRequest):
    """
    Remove labels from a phrase.
    """
    try:
        context.suggester.phrases_manager.remove_labels(
            phrase_id=body.phrase_id, labels=body.labels
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Phrase {body.phrase_id} not found",
        )


@router.post(
    "/phrases/add-domains",
    status_code=status.HTTP_200_OK,
)
def add_domains(body: SuggestingPhrasesAddDomainsRequest):
    """
    Add domains to a phrase.
    """
    try:
        context.suggester.phrases_manager.add_domains(
            phrase_id=body.phrase_id, labels=body.domains
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Phrase {body.phrase_id} not found",
        )


@router.post(
    "/phrases/remove-domains",
    status_code=status.HTTP_200_OK,
)
def remove_domains(body: SuggestingPhrasesRemoveDomainsRequest):
    """
    Remove domains from a phrase.
    """
    try:
        context.suggester.phrases_manager.remove_domains(
            phrase_id=body.phrase_id, labels=body.domains
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Phrase {body.phrase_id} not found",
        )


@router.post(
    "/phrases/update-probability",
    status_code=status.HTTP_200_OK,
)
def update_probability(body: SuggestingPhrasesAdjustProbabilityRequest):
    """
    Adjust (increment/decrement) a phrase's probability by `delta`.
    """
    try:
        context.suggester.phrases_manager.update_probability(
            phrase_id=body.phrase_id, new_probability=body.new_prob
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Phrase {body.phrase_id} not found",
        )


@router.get(
    "/phrases/get-info",
    response_model=SuggestingPhrase,
    status_code=status.HTTP_200_OK,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def get_phrase_info(phrase_id: str = Query(..., description="Phrase ID")):
    try:
        search_document = context.suggester.phrases_manager.get_info_by_id(
            phrase_id=phrase_id
        )
        return SuggestingPhrase(
            phrase=search_document.phrase,
            prob=search_document.prob,
            labels=search_document.labels,
        )

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Phrase ID {phrase_id} not found",
        )


@router.post(
    "/phrases/remove-all-labels",
    status_code=status.HTTP_200_OK,
)
def remove_labels(body: SuggestingPhrasesRemoveAllLabelsRequest):
    context.suggester.phrases_manager.remove_all_label_values(body.labels)


@router.post(
    "/phrases/remove-all-domains",
    status_code=status.HTTP_200_OK,
)
def remove_domains(body: SuggestingPhrasesRemoveAllDomainsRequest):
    context.suggester.phrases_manager.remove_all_domain_values(body.domains)


@router.get(
    "/phrases/list",
    response_model=ListPhrasesResponse,
    response_model_by_alias=False,
    response_model_exclude_none=True,
)
def list_phrases(body: ListPhrasesRequest):
    """
    Return a paginated list of phrase IDs.
    """
    search_documents = context.suggester.phrases_manager.list_phrases(
        offset=body.offset, limit=body.limit
    )
    return ListPhrasesResponse(
        items=[
            SuggestingPhrase(
                phrase=doc.phrase,
                prob=doc.prob,
                labels=doc.labels,
            )
            for doc in search_documents
        ]
    )
