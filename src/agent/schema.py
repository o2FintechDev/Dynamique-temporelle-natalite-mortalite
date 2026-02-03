from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

Intent = Literal["explore", "compare", "summarize"]

class ToolCall(BaseModel):
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    intent: Intent
    rationale: str
    tool_calls: list[ToolCall]

class EvidenceItem(BaseModel):
    artefact_id: str
    type: str
    title: str

class NarrativePacket(BaseModel):
    intent: Intent
    evidence: list[EvidenceItem]
    constraints: dict[str, Any] = Field(default_factory=dict)
