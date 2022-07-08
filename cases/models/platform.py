

from pydantic import BaseModel, validator

from cases.models.sink import SinkModel


class PlatformModel(BaseModel):
    group_of_sinks : list[SinkModel]


    @validator("group_of_sinks")
    def must_have_one(cls, v):
        if len(v) == 0:
            raise ValueError("Group of Sinks must not be Empty")
        return v
