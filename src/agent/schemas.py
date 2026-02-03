# src/agent/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Literal

Intent = Literal["explore", "diagnose", "model", "summarize", "export", "unknown"]

class ToolCall(BaseModel):
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    intent: Intent
    calls: list[ToolCall]

class Artifact(BaseModel):
    artifact_id: str
    kind: Literal["figure", "table", "metric", "text", "file"]
    name: str
    path: str
    meta: dict[str, Any] = Field(default_factory=dict)
