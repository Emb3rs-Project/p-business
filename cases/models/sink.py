
from pydantic import BaseModel, validator


class SinkModel(BaseModel):
    id : int
    name : str
    consumer_type : str


    @validator("consumer_type")
    def values_consumer_type(cls, v):
        possible_values = ["non-household","household"]
        if not v in possible_values:
            raise ValueError(f"consumer_type must be one of {possible_values}")