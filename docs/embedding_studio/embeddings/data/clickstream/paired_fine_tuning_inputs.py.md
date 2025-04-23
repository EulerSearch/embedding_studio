# Documentation for `_make_lists_equal_size`

## Functionality

The `_make_lists_equal_size` function adjusts two lists so they have equal sizes. It cycles through the elements of the shorter list until both lists reach the length of the longer list, then returns the two lists with equal sizes.

## Parameters

- `list1`: First list of elements.
- `list2`: Second list of elements.

## Usage

- **Purpose**: Equalizes the sizes of two lists for paired data processing.

### Example

```python
list1, list2 = _make_lists_equal_size([1, 2], [3, 4, 5, 6])
# Returns: ([1, 2, 1, 2], [3, 4, 5, 6])
```