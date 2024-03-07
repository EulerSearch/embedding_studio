from pydantic import BaseModel


class QueryItem(BaseModel):
    class Config:
        arbitrary_types_allowed = True
