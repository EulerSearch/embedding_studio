class DotDict:
    """
    A dictionary-like class that allows attribute-style access to dictionary items.

    This class converts a dictionary into an object where keys become attributes,
    allowing for dot notation access to dictionary values.

    :param dictionary: The dictionary to convert to a DotDict object
    :return: A new DotDict instance with the dictionary's keys as attributes
    """

    def __init__(self, dictionary):
        """
        Initialize a DotDict with values from the given dictionary.

        Each key-value pair from the dictionary becomes an attribute of the DotDict.

        :param dictionary: The dictionary to convert to a DotDict object
        """
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __repr__(self):
        """
        Return a string representation of the DotDict.

        Creates a representation showing the class name and all attributes with their values.

        :return: String representation of the DotDict instance
        """
        attrs = ", ".join(
            f"{key}={repr(value)}" for key, value in self.__dict__.items()
        )
        return f"{self.__class__.__name__}({attrs})"
