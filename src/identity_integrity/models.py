"""Validated data models shared by extraction and graph construction."""

from pydantic import BaseModel, ConfigDict


class Triple(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    relation: str
    object: str


class TripleList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    triples: list[Triple]
