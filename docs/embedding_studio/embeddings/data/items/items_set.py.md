# Documentation for ItemsSet Class

## Functionality
The ItemsSet class is a wrapper around a Huggingface Dataset. It is designed to represent and provide utilities for a set of search result items. It extends the Dataset to allow retrieval by item IDs and indices.

## Parameters
- dataset: A Huggingface Dataset containing items.
- item_field_name: The field representing the item used for embedding.
- id_field_name: The field representing the unique item identifier.

## Motivation
ItemsSet was created to facilitate search, indexing, and retrieval within a dataset. It integrates with the Huggingface Dataset, providing extra functionality specific to item management.

## Inheritance
ItemsSet inherits from the Huggingface Dataset class, augmenting its capabilities to support item-specific operations.

## Example
```python
dataset = load_dataset(...)
items_set = ItemsSet(dataset, "text", "id")
results = items_set.items_by_ids(["id1", "id2"])
```
This example shows how to instantiate and retrieve items using their IDs.

---

## Documentation for ItemsSet.item_field_name

### Functionality
This property returns the name of the dataset field that contains the items for processing by the embedding model. It retrieves the underlying `_item_field_name` attribute from an ItemsSet instance.

### Parameters
- *Getter*: No parameters are required when retrieving the item field name.
- *Setter*: 
  - value: A non-empty string representing the new name for the item field.

### Usage
- **Purpose**: Retrieve or update the data field name representing items in the dataset.

#### Example (Getter)
Assuming an ItemsSet instance is defined as `items_set`:

```python
item_name = items_set.item_field_name
```
This returns the current item field name.

#### Example (Setter)
To change the field name, ensuring the value is a non-empty string, you can do:

```python
items_set.item_field_name = "new_field_name"
```

---

## Documentation for ItemsSet.id_field_name

### Functionality
This property returns the name of the field used for item IDs in the dataset. It retrieves the current string value that specifies the field holding the unique identifier of an item.

### Parameters
- None for retrieving the value.
- When setting, use:
  - value: A non-empty string representing the ID field name.

### Usage
- **Purpose**: Retrieve or update the identifier field name for the ItemsSet. This property is used to access and manipulate the unique ID of each data item.

#### Example
To get the identifier field:

```python
id_field = items_set.id_field_name
```

To set a new identifier field:

```python
items_set.id_field_name = "new_id"
```

---

## Documentation for `ItemsSet.id_to_index`

### Functionality
This property returns a dictionary that maps each ID from the dataset to a list of indices where the ID occurs. If the mapping is uninitialized, it is rebuilt by iterating through the dataset and appending each occurrence's index.

### Parameters
None.

### Usage
- **Purpose**: Quickly retrieve the positions of items sharing the same ID in the dataset.

#### Example
```python
items_set = ItemsSet(dataset, "text", "id")
mapping = items_set.id_to_index
print(mapping["example_id"])
```

---

## Documentation for ItemsSet.rows_by_ids

This method retrieves rows from the original dataset based on a provided list of IDs. It uses an internal mapping from ID values to dataset indices. If `ignore_missed` is False, the method raises an `IndexError` for any missing ID.

### Parameters
- `ids`: A list of identifiers used to look up rows in the dataset.
- `ignore_missed`: Boolean flag. If False, missing IDs trigger an error; if True, missing IDs are skipped.

### Usage
- **Purpose**: Fetch rows from the dataset corresponding to given IDs.

#### Example
Suppose you have an ItemsSet instance named `set_obj`:

```python
rows = set_obj.rows_by_ids(["id1", "id2"], ignore_missed=True)
```

---

## Documentation for ItemsSet.items_by_indices

### Functionality
This method returns a slice of items from the dataset based on a given list of indices. It retrieves items corresponding to the specified index positions.

### Parameters
- `indices` (List[int]): A list of indices to retrieve items from the dataset.

### Usage
Use this method to access specific items in the dataset by passing a list of indices that indicate the desired item positions.

#### Example
```python
items_set = ItemsSet(dataset, 'id', 'item')
selected_items = items_set.items_by_indices([0, 2, 5])
```

---

## Documentation for ItemsSet.items_by_ids

### Functionality
Retrieves a slice of items from the dataset using a list of IDs. It uses the `rows_by_ids` method internally to get the rows and then extracts the item and ID fields.

### Parameters
- `ids`: List[Any] - A list of IDs for which items will be retrieved.

### Return Value
Returns a tuple containing two lists:
- A list of items corresponding to the given IDs.
- A list of IDs related to the retrieved items.

### Usage
Use this method to quickly obtain items and their IDs by passing a list of ID values.

#### Example
```python
items, ids = items_set.items_by_ids(['id1', 'id2'])
```

---

## Documentation for `ItemsSet.items_slice`

### Functionality
This method returns a subset of dataset items by slicing the dataset using the provided start and end indices.

### Parameters
- `start_idx`: An integer indicating the start index of the slice.
- `end_idx`: An optional integer specifying the end index (exclusive).

### Usage
- **Purpose**: To extract a list of items within a specified range from the dataset.

#### Example
```python
items = dataset.items_slice(10, 20)
```