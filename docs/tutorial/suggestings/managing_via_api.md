# Managing Suggestions with the Embedding Studio API

This tutorial explains how to use the Embedding Studio API to manage suggestions - from retrieving suggestions as users type to adding, updating, and maintaining your suggestion database.

## Getting Started

Embedding Studio exposes a set of REST API endpoints for managing suggestions under the `/suggesting` path. These endpoints allow you to:

1. Get suggestions for user input
2. Add new suggestion phrases
3. Manage phrase metadata (labels, domains, probability)
4. List and search existing phrases

## Retrieving Suggestions

The most common operation is retrieving suggestions as a user types:

### Get Suggestions

```http
POST /suggesting/get-top-k
```

**Request Body:**
```json
{
  "phrase": "how to make pas",
  "domain": "cooking",
  "top_k": 5
}
```

**Response:**
```json
{
  "suggestions": [
    {
      "prefix": "how to make ",
      "postfix": "pasta from scratch",
      "matching_span": {"start": 0, "end": 12},
      "match_type": "prefix",
      "prob": 0.85,
      "labels": ["recipe", "italian"]
    },
    {
      "prefix": "how to make ",
      "postfix": "pastry dough",
      "matching_span": {"start": 0, "end": 12},
      "match_type": "prefix",
      "prob": 0.72,
      "labels": ["baking", "dessert"]
    }
    // ...more suggestions
  ]
}
```

This endpoint:
1. Takes the user's input phrase
2. Optionally filters by domain
3. Returns top_k suggestions ranked by probability
4. Formats suggestions with prefix/postfix for easy display

### How the Response is Structured

For each suggestion:
- `prefix`: The part that matches what the user has already typed
- `postfix`: The suggested completion
- `matching_span`: The start and end positions in the original phrase
- `match_type`: How the match was found ("exact", "prefix", or "fuzzy")
- `prob`: The probability score (higher is more relevant)
- `labels`: Category tags associated with the suggestion

## Managing Suggestion Phrases

Embedding Studio provides a complete set of endpoints to manage your suggestion database:

### Add New Phrases

```http
POST /suggesting/phrases/add
```

**Request Body:**
```json
{
  "phrases": [
    {
      "phrase": "how to make pasta from scratch",
      "prob": 0.85,
      "labels": ["recipe", "italian"],
      "domains": ["cooking", "food"]
    },
    {
      "phrase": "how to make pastry dough",
      "prob": 0.72,
      "labels": ["baking", "dessert"],
      "domains": ["cooking", "food"]
    }
  ]
}
```

**Response:**
```json
{
  "phrase_ids": ["5f9a7b2c3d4e5f6a7b8c9d0e", "1a2b3c4d5e6f7g8h9i0j1k2l"]
}
```

This endpoint:
1. Accepts a list of phrases with metadata
2. Tokenizes and processes each phrase
3. Stores them in the suggestion database
4. Returns generated IDs for each phrase

### Delete a Phrase

```http
DELETE /suggesting/phrases/delete/{phrase_id}
```

This endpoint removes a specific phrase from the suggestion database.

## Managing Phrase Metadata

You can update various aspects of suggestion phrases:

### Add Labels to a Phrase

```http
POST /suggesting/phrases/add-labels
```

**Request Body:**
```json
{
  "phrase_id": "5f9a7b2c3d4e5f6a7b8c9d0e",
  "labels": ["beginner", "quick-meal"]
}
```

### Remove Labels from a Phrase

```http
POST /suggesting/phrases/remove-labels
```

**Request Body:**
```json
{
  "phrase_id": "5f9a7b2c3d4e5f6a7b8c9d0e",
  "labels": ["quick-meal"]
}
```

### Add Domains to a Phrase

```http
POST /suggesting/phrases/add-domains
```

**Request Body:**
```json
{
  "phrase_id": "5f9a7b2c3d4e5f6a7b8c9d0e",
  "domains": ["recipes", "beginners"]
}
```

### Remove Domains from a Phrase

```http
POST /suggesting/phrases/remove-domains
```

**Request Body:**
```json
{
  "phrase_id": "5f9a7b2c3d4e5f6a7b8c9d0e",
  "domains": ["beginners"]
}
```

## Updating Suggestion Probabilities

Adjust how suggestions are ranked:

```http
POST /suggesting/phrases/update-probability
```

**Request Body:**
```json
{
  "phrase_id": "5f9a7b2c3d4e5f6a7b8c9d0e",
  "new_prob": 0.95
}
```

Higher probability values (0.0-1.0) cause suggestions to appear higher in results.

## Retrieving Phrase Information

### Get Phrase Details

```http
GET /suggesting/phrases/get-info?phrase_id=5f9a7b2c3d4e5f6a7b8c9d0e
```

**Response:**
```json
{
  "phrase": "how to make pasta from scratch",
  "prob": 0.85,
  "labels": ["recipe", "italian", "beginner"]
}
```

### List Phrases

```http
GET /suggesting/phrases/list
```

**Request Body:**
```json
{
  "offset": 0,
  "limit": 100
}
```

**Response:**
```json
{
  "items": [
    {
      "phrase": "how to make pasta from scratch",
      "prob": 0.85,
      "labels": ["recipe", "italian", "beginner"]
    },
    {
      "phrase": "how to make pastry dough",
      "prob": 0.72,
      "labels": ["baking", "dessert"]
    }
    // ...more phrases
  ]
}
```

## Bulk Operations

For efficient management of large suggestion datasets:

### Remove Labels from All Phrases

```http
POST /suggesting/phrases/remove-all-labels
```

**Request Body:**
```json
{
  "labels": ["outdated", "deprecated"]
}
```

This removes specified labels from all phrases that have them.

### Remove Domains from All Phrases

```http
POST /suggesting/phrases/remove-all-domains
```

**Request Body:**
```json
{
  "domains": ["old-category", "test"]
}
```

This removes specified domains from all phrases that have them.

## Building a Suggestion Pipeline

Here's a typical workflow for managing suggestions:

1. **Initial Setup**: Add a base set of suggestion phrases with appropriate domains and labels
2. **User Interaction**: As users interact with your system, collect data on which suggestions are used
3. **Feedback Loop**: Periodically update suggestion probabilities based on usage patterns
4. **Maintenance**: Add new phrases, remove outdated ones, and adjust metadata

## Real-World Example: E-commerce Search

For an e-commerce site:

1. **Add domain-specific suggestions**:
   ```json
   {
     "phrases": [
       {"phrase": "men's running shoes", "prob": 0.9, "domains": ["footwear", "men"]},
       {"phrase": "women's running shoes", "prob": 0.9, "domains": ["footwear", "women"]},
       {"phrase": "bluetooth headphones", "prob": 0.85, "domains": ["electronics", "audio"]}
     ]
   }
   ```

2. **Get user suggestions**: As a user types "run" in the women's section:
   ```json
   {"phrase": "run", "domain": "women", "top_k": 5}
   ```

3. **Update probabilities**: After noticing "women's running shoes" is selected more often:
   ```json
   {"phrase_id": "abc123", "new_prob": 0.95}
   ```

## Performance Considerations

- **Batch additions**: When adding many phrases, use the bulk add endpoint
- **Pagination**: Use offset and limit when listing phrases
- **Domain filtering**: Always specify a domain when getting suggestions for better performance
- **Probabilities**: Keep probability scores updated to maintain relevant results