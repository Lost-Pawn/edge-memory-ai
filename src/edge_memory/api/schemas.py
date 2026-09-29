"""Request/response schemas for the edge API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class UpdateRequest(BaseModel):
    text: str = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    hybrid: bool = True


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class OfflineRequest(BaseModel):
    offline: bool
