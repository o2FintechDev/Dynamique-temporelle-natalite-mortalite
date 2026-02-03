from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Literal

Intent = Literal["explore", "compare", "quality", "synthesize"]

class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    intent: Intent
    tool_calls: list[ToolCall]
    notes: str = ""

class Artifact(BaseModel):
    artifact_id: str
    kind: Literal["table", "figure", "metrics", "json"]
    title: str
    payload: Any
    meta: dict[str, Any] = Field(default_factory=dict)

class EvidenceSentence(BaseModel):
    sentence: str
    evidence_ids: list[str]

class Narrative(BaseModel):
    title: str
    bullets: list[EvidenceSentence]
