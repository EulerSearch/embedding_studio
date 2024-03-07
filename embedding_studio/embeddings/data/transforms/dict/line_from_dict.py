from typing import Dict, List, Optional, Union


def get_field_name(
    name: str,
    field_names: Optional[Dict[str, str]] = None,
    ignore_missed: bool = True,
) -> str:
    """Get mapped (or not) name of a field.

    :param name: name of a field to be mapped.
    :param field_names: dictionary of name mappings (default: None)
    :param ignore_missed: to throw an error if there is no such keys in field_names (default: True)
    :return: mapped name of a field.
    """
    if ignore_missed:
        return field_names.get(name, name)
    else:
        if field_names is not None:
            return field_names[name]
        else:
            return name


def get_text_line_from_dict(
    object: dict,
    separator: str = " ",
    order_fields: Optional[Union[bool, List[str]]] = True,
    ascending: bool = True,
    field_names: Optional[Dict[str, str]] = None,
    ignore_missed: bool = True,
) -> str:
    """Convert a dict into a string, that can be used for embedding models fine-tuning.

    :param object: dict to be converted into a string.
    :param separator: symbol to be used to separation of different values (default: ' ')
    :param order_fields: the way to order fields (default: True)
                         If set as False, field will bnot be sorted at all.
                         If set as True, fields will be sorted accordingly to ascending params.
                         If set as list of object keys, so fields will be sorted accordingly to the provided order.
    :param ascending: (default: True)
    :param field_names: dictionary of name mappings (default: None)
    :param ignore_missed: to throw an error if there is no such keys in field_names (default: True)
    :return:
    """
    fields = list(object.keys())
    if isinstance(order_fields, list):
        fields = order_fields
    elif order_fields == True:
        fields = sorted(fields, reverse=not ascending)

    return separator.join(
        [
            f"{get_field_name(k, field_names, ignore_missed)}: {order_fields[k]}"
            for k in fields
        ]
    )
