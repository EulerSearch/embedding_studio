from torch import Tensor


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Performs average pooling on the last hidden states using an attention mask.

    This function processes the last hidden states of a neural network by applying an attention mask.
    The attention mask is a binary matrix where a zero indicates that the corresponding position
    in the hidden states should not be attended to. During the operation,
    positions in the last hidden states that are aligned with zeros in the attention mask are set to zero.
    This effectively disregards those positions in subsequent processing steps,
    ensuring that the network only focuses on the relevant parts of the input as defined by the mask.
    It then computes the sum of these adjusted values along the sequence dimension
    and divides by the sum of the attention mask to obtain the average. This is
    useful for aggregating token-level representations into a single vector,
    considering only the tokens of interest as specified by the attention mask.

    :param last_hidden_states: A tensor of shape [batch_size, sequence_length, hidden_size]
                               containing the last hidden states of a model.
    :param attention_mask: A tensor of the same height as last_hidden_states indicating
                           which tokens to attend to (1) and which to ignore (0).
    :return: A tensor of shape [batch_size, hidden_size] representing the averaged
             hidden states.
    """
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
