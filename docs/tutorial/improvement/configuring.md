# Tuning and Using Embedding Studio's Improvement System

## Current Implementation Details

Embedding Studio's improvement system is implemented as a combination of several interacting components, working together to create a feedback loop between user behavior and search relevance. Let's examine how the current implementation works in practice.

### The `TorchBasedAdjuster` Implementation

The core algorithm in `torch_based_adjuster.py` uses PyTorch to perform gradient descent optimization on embedding vectors. Here's how the current implementation works:

```python
class TorchBasedAdjuster(VectorsAdjuster):
    def __init__(
        self,
        search_index_info: SearchIndexInfo,
        adjustment_rate: float = 0.1,
        num_iterations: int = 10,
        softmin_temperature: float = 1.0,
    ):
        # Initialize parameters
        self.search_index_info = search_index_info
        self.adjustment_rate = adjustment_rate
        self.num_iterations = num_iterations
        self.softmin_temperature = softmin_temperature
```

This implementation is based on solid machine learning principles:

1. It uses Adam optimization (AdamW), an advanced gradient descent algorithm that adapts learning rates
2. It computes similarity based on your chosen metric (cosine, dot product, or Euclidean)
3. It runs multiple optimization iterations to refine vectors gradually
4. It applies a cubic function to the similarity scores to emphasize significant differences

### The Improvement Pipeline

The worker in `worker.py` coordinates the improvement process:

1. It periodically checks for pending sessions marked for improvement
2. It processes them in batches to efficiently utilize resources
3. It handles errors gracefully, ensuring the system remains stable

### Vector Personalization

The current implementation creates personalized vectors for each user:

```python
# Create personalized ID by appending user_id
new_object_id = (
    object_id
    if object_id in not_originals
    else f"{object_id}_{user_id}"
)
```

This means:
- Each user gets their own optimized vectors
- New users see the original vectors
- User preferences don't interfere with each other

## How to Configure and Tune the System

The improvement system can be tuned through several key parameters, each affecting different aspects of the adjustment process:

### 1. Adjustment Rate

The adjustment rate (learning rate) controls how aggressively vectors are modified:

```python
adjuster = TorchBasedAdjuster(
    search_index_info=search_index_info,
    adjustment_rate=0.1,  # Configurable
    # ...
)
```

**Recommendation ranges**:
- Conservative: 0.01-0.05
- Balanced: 0.05-0.2
- Aggressive: 0.2-0.5

**Effects**:
- Higher values create more dramatic changes but may cause overshooting
- Lower values make more conservative adjustments requiring more iterations
- For new deployments, start conservative and increase gradually

### 2. Number of Iterations

This parameter determines how many optimization steps are performed for each batch:

```python
adjuster = TorchBasedAdjuster(
    search_index_info=search_index_info,
    num_iterations=10,  # Configurable
    # ...
)
```

**Recommendation ranges**:
- Quick adjustment: 5-10
- Standard processing: 10-25
- Deep optimization: 25-50

**Effects**:
- More iterations allow for finer adjustments but increase processing time
- Fewer iterations are faster but may not fully optimize the vectors
- Complex embedding spaces typically benefit from more iterations

### 3. Softmin Temperature

This parameter affects how the system handles multi-part vectors:

```python
adjuster = TorchBasedAdjuster(
    search_index_info=search_index_info,
    softmin_temperature=1.0,  # Configurable
    # ...
)
```

**Recommendation ranges**:
- Sharp focus: 0.1-0.5
- Balanced: 0.5-2.0
- Smooth distribution: 2.0-5.0

**Effects**:
- Lower values make the minimum distance more influential
- Higher values smooth out the influence across multiple parts
- Especially important for document chunking or multi-modal embeddings

### 4. Scheduler Timing

Controls how frequently the improvement worker runs:

```python
# In settings.py or environment variables
IMPROVEMENT_SECONDS_INTERVAL=3600  # Run hourly
```

**Recommendation ranges**:
- Real-time systems: 300-900 seconds (5-15 minutes)
- Balanced systems: 1800-3600 seconds (30-60 minutes)
- Batch-oriented: 7200-86400 seconds (2-24 hours)

**Effects**:
- More frequent runs provide faster feedback but use more resources
- Less frequent runs are more efficient but delay improvement effects
- Match with your typical user engagement patterns

## Implementation Example

Here's a complete example of how to configure a custom vectors adjuster in your plugin:

```python
from embedding_studio.embeddings.improvement.torch_based_adjuster import TorchBasedAdjuster
from embedding_studio.models.embeddings.models import SearchIndexInfo

class MyCustomPlugin:
    # ...
    
    def get_vectors_adjuster(self):
        """
        Returns a configured vectors adjuster instance.
        """
        # Get search index information with distance metrics
        search_index_info = self.get_search_index_info()
        
        # For e-commerce product search
        if self.is_product_search:
            return TorchBasedAdjuster(
                search_index_info=search_index_info,
                adjustment_rate=0.15,     # Moderately aggressive
                num_iterations=15,        # Standard processing
                softmin_temperature=0.8,  # Slightly focused
            )
        
        # For document search
        elif self.is_document_search:
            return TorchBasedAdjuster(
                search_index_info=search_index_info,
                adjustment_rate=0.05,      # Conservative
                num_iterations=25,         # More iterations for complex docs
                softmin_temperature=2.0,   # Smoother for chunked documents
            )
        
        # Default configuration
        else:
            return TorchBasedAdjuster(
                search_index_info=search_index_info,
                adjustment_rate=0.1,
                num_iterations=10,
                softmin_temperature=1.0,
            )
```

## Improvement Foundations

### The Science Behind Vector Adjustment: How and Why It Works

To understand why the current implementation works so well, let's explore the science behind it:

#### Vector Space Models: The Foundation

Embedding Studio is built on the concept of vector space models, where:

- Words, phrases, and documents are represented as points in a high-dimensional space
- Similarity between items is measured by their proximity in this space
- The dimensions of this space capture semantic meaning

This mathematical representation allows us to:
1. Convert language into numbers (embeddings)
2. Measure how similar concepts are
3. Find related content efficiently

#### The Learning Process: Gradient Descent

The adjustment algorithm uses gradient descent, a fundamental machine learning technique:

1. We define a goal: make clicked items more similar to queries and non-clicked items less similar
2. We measure how far we are from that goal using a loss function
3. We compute the gradient (direction of steepest improvement)
4. We take small steps in that direction
5. We repeat until we're satisfied with the results

This is similar to finding your way down a mountain in fog - you feel which way is steepest and take small steps downward until you reach the valley.

#### Why the Cubic Function Matters

The current implementation uses a cubic function in the loss calculation:

```python
loss = -torch.mean(clicked_similarity**3) + torch.mean(
    non_clicked_similarity**3
)
```

This is a critical design choice because:

1. It emphasizes large differences more than small ones
2. It creates stronger gradients for items that are very wrong
3. It helps the system focus on fixing the most egregious ranking errors first

In simpler terms, it's like telling the system: "Don't worry too much about items that are almost right, but really fix the ones that are way off."

#### Multi-Part Document Handling

The code handles multi-part documents (like chunked text) through sophisticated aggregation:

```python
# Differentiable soft minimum using log-sum-exp
softmin_weights = torch.exp(
    -similarities / softmin_temperature
)
softmin_weights = softmin_weights / softmin_weights.sum(
    dim=2, keepdim=True
)
similarities = torch.sum(
    softmin_weights * similarities, dim=2
)
```

This creates a differentiable version of the minimum function, allowing the system to focus on the most similar part of a document while still being able to compute gradients for learning.

#### Why Personalization Works

The personalization approach in the current implementation is particularly effective because:

1. It creates separate vectors for each user's preferences
2. It preserves the original vectors for new users
3. It allows contradictory preferences among different users

This is like having a library where the books stay in the same places for new visitors, but regular visitors get their own customized maps showing shortcuts to their favorite sections.

### Scientific Benefits of the Current Approach

The Embedding Studio implementation has several scientific advantages:

1. **It's computationally efficient**: By using PyTorch and batched processing, it can handle large datasets
2. **It's mathematically sound**: Using gradient descent with proper loss functions ensures convergence
3. **It respects different similarity metrics**: Working with cosine, dot product, or Euclidean distance
4. **It handles uncertainty gracefully**: Using softmin for multi-part documents
5. **It avoids catastrophic forgetting**: By creating personalized copies rather than modifying originals

## Advanced Tuning Considerations

### 1. Session Selection Criteria

Not all sessions provide equally valuable signal. Consider implementing custom filters:

```python
def should_use_session_for_improvement(session):
    # Skip sessions with no clicks
    if len(session.events) == 0:
        return False
        
    # Skip very short sessions (possibly bounces)
    if session.duration_seconds < 10:
        return False
        
    # Skip ambiguous sessions (too many clicks without clear preference)
    if len(session.events) > 10:
        return False
        
    # Prioritize sessions with clicks on lower-ranked results
    # (These indicate potential ranking issues)
    for event in session.events:
        result_position = get_result_position(session, event.object_id)
        if result_position > 3:  # Clicked something not in top 3
            return True
            
    return True  # Default: use session
```

### 2. Vector Normalization

Different embedding models may require different normalization:

```python
# Normalize vectors before adjustment
def prepare_vectors(vectors):
    # L2 normalization for models using cosine similarity
    if self.search_index_info.metric_type == MetricType.COSINE:
        return F.normalize(vectors, p=2, dim=-1)
    
    # Min-max scaling for dot product models
    elif self.search_index_info.metric_type == MetricType.DOT:
        min_val = torch.min(vectors)
        max_val = torch.max(vectors)
        return (vectors - min_val) / (max_val - min_val)
    
    # No normalization for Euclidean distance
    else:
        return vectors
```

### 3. Multi-part Vector Handling

For models that produce multiple vectors per item (like chunked documents), the aggregation strategy matters:

```python
# In your plugin configuration
search_index_info = SearchIndexInfo(
    metric_type=MetricType.COSINE,
    # Choose aggregation strategy based on content type
    metric_aggregation_type=MetricAggregationType.MIN  # or AVG
)
```

**Guidance for choosing aggregation**:
- `MIN`: Better for finding exact matches within documents
- `AVG`: Better for overall document relevance
- Document search typically benefits from MIN
- Product/image search typically benefits from AVG

### 4. Feedback Loop Management

Be cautious of reinforcement bias where the system becomes too focused on existing patterns:

```python
# Example: Mix in exploration results
def get_search_results(query, user_id, exploration_ratio=0.2):
    # Get personalized results
    personalized_results = get_personalized_results(query, user_id)
    
    # Get non-personalized results
    standard_results = get_standard_results(query)
    
    # Mix results to maintain exploration
    final_results = []
    for i in range(len(personalized_results)):
        if random.random() < exploration_ratio:
            # Include some non-personalized results
            final_results.append(standard_results[i])
        else:
            # Use personalized results
            final_results.append(personalized_results[i])
            
    return final_results
```

## Practical Implementation Steps

To implement the improvement system effectively:

1. **Enable clickstream collection**:
   ```python
   # Register search session
   session = Session(
       session_id=str(uuid.uuid4()),
       search_query=query,
       user_id=user_id,
       created_at=datetime_utils.utc_timestamp()
   )
   context.clickstream_dao.register_session(session)
   
   # Record click events
   event = SessionEvent(
       session_id=session_id,
       object_id=clicked_item_id,
       created_at=datetime_utils.utc_timestamp()
   )
   context.clickstream_dao.push_events([event])
   ```

2. **Schedule improvement processing**:
   ```python
   # Mark session for improvement
   context.sessions_for_improvement.create(
       schema=SessionForImprovementCreateSchema(
           session_id=session_id,
       )
   )
   ```

3. **Configure improvement parameters**:
   ```python
   # In your settings.py or environment variables
   ADJUSTMENT_RATE=0.1
   NUM_ITERATIONS=15
   SOFTMIN_TEMPERATURE=1.0
   IMPROVEMENT_SECONDS_INTERVAL=1800
   ```
By carefully tuning these parameters and monitoring the results, you can optimize the improvement system for your specific use case, embedding models, and user behavior patterns.