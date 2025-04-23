# Documentation for `average_pool`

## Functionality

The `average_pool` function applies average pooling on neural network hidden states. It uses an attention mask to ignore tokens that are not of interest, ensuring that only the specified tokens contribute to the pooling operation.

## Parameters

- `last_hidden_states`: A tensor of shape [batch_size, sequence_length, hidden_size] containing the last hidden states of the model.
- `attention_mask`: A tensor of the same sequence dimension marking tokens to attend (1) and ignore (0).

## Usage

- **Purpose**: To aggregate token-level representations into a single vector by averaging over tokens specified by the attention mask.

### Example

Suppose you have model outputs stored in `hidden_states` and an `attention_mask`. You can obtain the pooled output via:

    pooled = average_pool(hidden_states, attention_mask)