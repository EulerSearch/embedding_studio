# Embedding Studio Improvement Workflow

## Understanding Vector Improvement

Embedding Studio's improvement system is a powerful feature that leverages user interaction data to continuously enhance search relevance. This guide explains how the improvement workflow operates and the underlying mechanics that make it effective.

## Core Concept: How It Works in Simple Terms

Vector improvement in Embedding Studio operates on an intuitive principle that mimics human learning: **When a user clicks on a search result, the system learns that this result was relevant to their query.** 

Imagine each piece of content and each search query as points in space. When a user searches for something, we find content that's close to their query in this space. When they click on some results but not others, the system:

1. Gently moves the clicked items closer to the query point
2. Pushes non-clicked items slightly further away

Over time, as many users perform searches, the system continuously refines these positions, creating a more accurate map of relevance that reflects real human preferences. This creates a natural feedback loop that automatically improves search quality based on actual usage.

This works because vector embeddings capture semantic meaning - when two vectors are close together, it means their content is conceptually similar. By adjusting these vectors based on user behavior, we're essentially encoding human judgment directly into the search system.

## Detailed Workflow

### 1. Clickstream Collection

The improvement process begins with collecting user interactions:

```
User Search → Results Displayed → User Clicks (or doesn't) → Interaction Captured
```

The system captures:
- User queries (converted to vector embeddings)
- Search results presented to the user
- Which results the user clicked on
- Session metadata (time, user ID, etc.)

Code responsible for this:
- `clickstream_client.py`: Contains endpoints for registering sessions and recording events
- `sessions.py`: Defines the data structures for storing session information

### 2. Session Processing

Periodically, the improvement worker processes sessions marked for improvement:

```python
@dramatiq.actor(
    queue_name="improvement_worker",
    max_retries=settings.IMPROVEMENT_WORKER_MAX_RETRIES,
    time_limit=settings.IMPROVEMENT_WORKER_TIME_LIMIT,
)
def improvement_worker():
    # Fetch pending sessions and process them
    sessions_for_improvement = []
    # ...
    handle_improvement(sessions_for_improvement)
```

Key processing steps:
1. Sessions are filtered to include only those with meaningful user interactions
2. Irrelevant sessions (explicitly marked or with no clicks) are excluded
3. Valid sessions are batched for efficient processing

### 3. Vector Preparation

For each valid session, the system prepares the vectors for adjustment:

1. **Query Vector Retrieval**: Gets the embedding of what the user searched for
2. **Results Classification**:
   - Clicked items → positive examples
   - Non-clicked items → negative examples
3. **Vector Normalization**: Ensures all vectors have consistent dimensions

```python
# Convert tensors for processing
queries = torch.stack([inp.query.vector for inp in data_for_improvement])
clicked_vectors = torch.stack([
    torch.stack([ce.vector for ce in inp.clicked_elements], dim=1)
    for inp in data_for_improvement
]).transpose(1, 2)
non_clicked_vectors = torch.stack([
    torch.stack([nce.vector for nce in inp.non_clicked_elements], dim=1)
    for inp in data_for_improvement
]).transpose(1, 2)
```

### 4. Vector Adjustment: The Science Behind It

The heart of the improvement system is the `TorchBasedAdjuster`, which uses machine learning techniques to optimize vectors. Let's break down how it works:

```python
# Enable gradient tracking for optimization
clicked_vectors.requires_grad_(True)
non_clicked_vectors.requires_grad_(True)

# Set up optimizer
optimizer = torch.optim.AdamW(
    [clicked_vectors, non_clicked_vectors],
    lr=self.adjustment_rate
)

# Run optimization iterations
for _ in range(self.num_iterations):
    optimizer.zero_grad()
    
    # Compute similarities
    clicked_similarity = self.compute_similarity(
        queries, clicked_vectors, self.softmin_temperature
    )
    non_clicked_similarity = self.compute_similarity(
        queries, non_clicked_vectors, self.softmin_temperature
    )
    
    # Define loss function:
    # - Maximize clicked similarities (negative term)
    # - Minimize non-clicked similarities (positive term)
    loss = -torch.mean(clicked_similarity**3) + torch.mean(
        non_clicked_similarity**3
    )
    
    # Compute gradients and update vectors
    loss.backward()
    optimizer.step()
```

#### What's really happening here?

Imagine we're teaching the system to understand user preferences through a game of "hotter/colder":

1. **Setting up the game**: We start with the current positions of query vectors, clicked items, and non-clicked items in the embedding space.

2. **Learning mechanism**: The system can determine which direction to move each vector to increase or decrease similarity.

3. **The goal**: Make clicked items "hotter" (closer to the query) and non-clicked items "colder" (further from the query).

4. **The cubic function (x³)**: This is a clever mathematical trick. Using the cube instead of just the raw similarity means:
   - Small differences are barely changed (if something is just slightly wrong, we don't overreact)
   - Large differences get much stronger corrections (if something is very wrong, we make bigger adjustments)

5. **Multiple iterations**: Instead of making one big adjustment (which might overcompensate), we make several small adjustments, checking each time if we're improving.

The system supports different ways of measuring "closeness" between vectors:

- **COSINE**: Only cares about direction, not magnitude - like measuring the angle between two sticks regardless of their length.
- **DOT**: Cares about both direction and magnitude - longer sticks pointed in the right direction rank higher.
- **EUCLID**: Measures the straight-line distance between points - truly spatial similarity.

This implementation directly translates the intuitive idea of "make good results more findable" into a mathematical optimization problem that PyTorch can solve efficiently.

### 5. Personalization

A key aspect of the improvement system is how it handles personalization:

```python
# Create personalized ID by appending user_id if not already personalized
new_object_id = (
    object_id
    if object_id in not_originals
    else f"{object_id}_{user_id}"
)
```

Rather than modifying original vectors (which would affect all users), the system:
1. Creates personalized copies of vectors for specific users
2. Leaves original vectors intact for users without interaction history
3. Associates adjusted vectors with particular user IDs

This approach enables:
- User-specific search experiences
- Preservation of a baseline for new users
- Isolation between different users' feedback

### 6. Deployment

Once vectors are adjusted, they're upserted into the vector database:

```python
# Insert or update the personalized vectors
if len(objects_to_upsert) > 0:
    blue_collection.upsert(objects_to_upsert)
```

These personalized vectors immediately influence future search results, creating a continuous feedback loop:

```
User Search → Clickstream Collection → Vector Adjustment → 
Personalized Vectors → Enhanced Search Results → More User Feedback
```

## Internal Data Structures

Understanding the key data structures helps comprehend the improvement flow:

- **ImprovementInput**: Container for a session's query and result vectors
  ```python
  ImprovementInput(
      session_id=session.session_id,
      query=ImprovementElement(...),  # Query vector
      clicked_elements=[...],         # Vectors user clicked on
      non_clicked_elements=[...],     # Vectors user didn't click on
  )
  ```

- **ImprovementElement**: Represents a vector with metadata
  ```python
  ImprovementElement(
      id=res_obj.object_id,
      vector=res_vector,       # Tensor of shape [M, D]
      is_average=[...],        # Flags for avg vs specific vectors
      user_id=session.user_id,
  )
  ```

- **Object**: Database representation of an embedding
  ```python
  Object(
      object_id=new_object_id,  # Personalized ID
      original_id=object_id,    # Reference to original
      user_id=user_id,
      parts=object_parts,       # Vector components
      # Additional metadata
  )
  ```

This complete pipeline transforms user interactions into optimized search experiences, continuously learning from real-world usage patterns.