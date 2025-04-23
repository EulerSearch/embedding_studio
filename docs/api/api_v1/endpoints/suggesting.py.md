# Documentation for Suggesting Usage and Management API

---

## `/get-top-k`

### Functionality
Returns top-k phrase suggestions based on a partial phrase and optional domain.

### Request Parameters
- `phrase` *(str)*: The partial input text to complete.
- `domain` *(str, optional)*: Contextual domain to narrow down suggestions.
- `top_k` *(int, default=3)*: Number of top suggestions to return (max: 100).

### Request JSON Example
```json
{
  "phrase": "deep l",
  "domain": "ml",
  "top_k": 5
}
```
- `phrase`: The text input for which to generate suggestions.
- `domain`: A string specifying the contextual domain for the input.
- `top_k`: An integer indicating the number of top suggestions to return.

### Response JSON Example
```json
{
  "suggestions": [
    {
      "prefix": "deep ",
      "postfix": "learning",
      "matching_span": { "start": 0, "end": 5 },
      "match_type": "prefix",
      "prob": 0.92,
      "labels": ["neural-nets", "ml"],
      "domains": ["ml"]
    }
  ]
}
```
- `prefix`: The portion of the phrase matched by user input.
- `postfix`: The suggested text completion.
- `matching_span`: Object indicating which part of the input matched.
- `match_type`: Matching strategy type (`exact`, `prefix`, `fuzzy`, etc.).
- `prob`: Probability/confidence score for suggestion relevance.
- `labels`: Related labels or tags for the suggestion.
- `domains`: Associated application domains.

---

## `/phrases/add`

### Functionality
Adds multiple phrases to the suggestion database.

### Request JSON Example
```json
{
  "phrases": [
    {
      "phrase": "deep learning",
      "labels": ["ai", "ml"],
      "domains": ["ml"],
      "prob": 0.95
    }
  ]
}
```
- `phrase`: Text of the suggesting phrase.
- `labels`: Tags or categories assigned to the phrase.
- `domains`: Contexts where the phrase applies.
- `prob`: Numeric weighting for phrase ranking (higher = more likely to appear).

### Response JSON Example
```json
{
  "phrase_ids": ["a1b2c3d4"]
}
```
- `phrase_ids`: List of internal system-assigned IDs for each added phrase.

---

## `/phrases/add-labels`

### Functionality
Adds new labels to a specific phrase.

### Request JSON Example
```json
{
  "phrase_id": "abc123",
  "labels": ["new-tag", "trending"]
}
```
- `phrase_id`: Unique identifier of the phrase to modify.
- `labels`: New labels to associate with the phrase.

---

## `/phrases/remove-labels`

### Functionality
Removes specific labels from a phrase.

### Request JSON Example
```json
{
  "phrase_id": "abc123",
  "labels": ["old-tag"]
}
```
- `phrase_id`: Unique phrase identifier.
- `labels`: List of labels to be removed.

---

## `/phrases/add-domains`

### Functionality
Associates domains with a phrase.

### Request JSON Example
```json
{
  "phrase_id": "abc123",
  "domains": ["science", "edu"]
}
```
- `phrase_id`: ID of the phrase to update.
- `domains`: New domains to associate with the phrase.

---

## `/phrases/remove-domains`

### Functionality
Removes domains from a phrase.

### Request JSON Example
```json
{
  "phrase_id": "abc123",
  "domains": ["edu"]
}
```
- `phrase_id`: Phrase identifier.
- `domains`: Domains to disassociate.

---

## `/phrases/remove-all-domains`

### Functionality
Globally removes specified domains from all phrases.

### Request JSON Example
```json
{
  "domains": ["deprecated"]
}
```
- `domains`: List of domain names to remove globally.

---

## `/phrases/update-probability`

### Functionality
Updates the probability score of a phrase.

### Request JSON Example
```json
{
  "phrase_id": "abc123",
  "new_prob": 0.85
}
```
- `phrase_id`: ID of the phrase to modify.
- `new_prob`: New probability value (0–1) used for ranking.

---

## `/phrases/get-info`

### Functionality
Returns full details for a single phrase.

### Response JSON Example
```json
{
  "phrase": "deep learning",
  "labels": ["ai", "ml"],
  "domains": ["ml"],
  "prob": 0.92
}
```
- `phrase`: Suggesting phrase text.
- `labels`: Phrase categories or tags.
- `domains`: Relevant contextual scopes.
- `prob`: Phrase ranking score.

---

## `/phrases/list`

### Functionality
Returns paginated list of all phrases.

### Request JSON Example
```json
{
  "limit": 20,
  "offset": 0
}
```
- `limit`: Number of items to return.
- `offset`: Pagination start index.

### Response JSON Example
```json
{
  "items": [
    {
      "phrase": "machine learning",
      "labels": ["ml"],
      "domains": ["ml"],
      "prob": 1.0
    }
  ]
}
```
- `items`: List of full suggesting phrase records.