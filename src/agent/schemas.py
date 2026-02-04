from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

Intent = Literal["exploration", "methodologie", "modelisation", "resultats", "anthropologie", "export"]

class ToolCall(BaseModel):
    tool_name: str = Field(..., description="Nom logique de l'outil agent (clé tools.py)")
    variables: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    intent: Intent
    page_targets: list[str] = Field(default_factory=list, description="Pages Streamlit à alimenter")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    notes: str | None = None

class ArtefactRef(BaseModel):
    kind: Literal["figure", "table", "metric", "model", "manifest"]
    path: str
    label: str | None = None

class ExecutionResult(BaseModel):
    run_id: str
    intent: Intent
    variables: list[str] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list)
    artefacts: list[ArtefactRef] = Field(default_factory=list)

class Manifest(BaseModel):
    run_id: str
    created_at_utc: str
    user_query: str
    intent: Intent
    tools_called: list[str]
    variables: list[str]
    artefacts: list[dict[str, Any]] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)
