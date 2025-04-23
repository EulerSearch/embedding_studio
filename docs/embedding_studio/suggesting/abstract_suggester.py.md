# AbstractSuggester Documentation

## Overview

`AbstractSuggester` is an abstract base class that defines the interface for all suggester implementations in the embedding studio. It establishes a contract by requiring the implementation of the `get_topk_suggestions` method and the `phrases_manager` property. The class standardizes the suggestion workflow by outlining how to generate suggestions based on input requests, allowing for flexibility and customization by subclasses.

## Method: get_topk_suggestions

### Functionality

This method returns the top-k suggestion candidates using the provided request. It retrieves candidate phrases from the `phrases_manager`, computes scores, and returns the highest ranking suggestions as a list of `Suggest` objects.

### Parameters

- `request`: A `SuggestingRequest` instance containing details such as the text context for suggestions.
- `top_k`: An integer specifying the number of top suggestions to return. Defaults to 10.

### Return Value

A list of `Suggest` objects, each encapsulating a suggestion text and its associated score.

### Implementation Example

Subclasses must override this abstract method. An example implementation is shown below:

```python
class MySuggester(AbstractSuggester):
    def get_topk_suggestions(self, request: SuggestingRequest, top_k: int = 10) -> List[Suggest]:
        candidate_phrases = self.phrases_manager.get_candidate_phrases(request.text)
        scored_phrases = [(self._score_phrase(phrase, request), phrase) for phrase in candidate_phrases]
        scored_phrases.sort(reverse=True)
        results = [Suggest(text=phrase.text, score=score) for score, phrase in scored_phrases[:top_k]]
        return results
```

### Purpose

This method defines a contract for obtaining text suggestions based on the context provided in a `SuggestingRequest`. It allows flexibility in how suggestions are scored and returned, enabling custom implementation by subclasses without altering the interface.

---

## Property: phrases_manager

### Functionality

Returns the phrase manager instance used by the suggester. The phrase manager is responsible for managing candidate phrases for suggestions.

### Parameters

None.

### Usage

- **Purpose**: Retrieve the phrases manager in a subclass of `AbstractSuggester`.

#### Example

```python
class MySuggester(AbstractSuggester):
    @property
    def phrases_manager(self) -> AbstractSuggestionPhraseManager:
        return self._phrase_manager
```

---

## Inheritance

`AbstractSuggester` inherits from Python's ABC module and enforces that all extending classes implement its abstract methods.

### Usage

Subclasses must override:
- `get_topk_suggestions`: Returns a list of suggestions based on a request.
- `phrases_manager`: Manages candidate phrases and scoring.

#### Example

Below is an example subclass implementation:

```python
class CustomSuggester(AbstractSuggester):
    @property
    def phrases_manager(self):
        return custom_phrase_manager

    def get_topk_suggestions(self, request, top_k=10):
        candidates = self.phrases_manager.get_candidate_phrases(request.text)
        # Logic for scoring and ranking candidates.
        return candidates[:top_k]
```