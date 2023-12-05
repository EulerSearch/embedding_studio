from typing import Annotated

from pydantic import BeforeValidator

# PyObjectId - a custom data type for MongoDB object identifiers,
# using the annotation with BeforeValidator for data type validation
# before serialization (for the situation where the ObjectId is represented
# as a string).
PyObjectId = Annotated[str, BeforeValidator(str)]
