# src/agent/schemas.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

Intent = Literal["exploration", "methodologie", "modelisation", "resultats", "anthropologie", "export"]


class ToolCall(BaseModel):
    tool_name: str = Field(..., description="Nom logique de l'outil agent (clé tools.py)")
    variables: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    intent: Intent
    page_targets: List[str] = Field(default_factory=list, description="Pages Streamlit à alimenter")
    tool_calls: List[ToolCall] = Field(default_factory=list)
    notes: Optional[str] = None


class ArtefactRef(BaseModel):
    kind: Literal["figure", "table", "metric", "model", "manifest"]
    path: str
    label: Optional[str] = None


class ExecutionResult(BaseModel):
    run_id: str
    intent: Intent
    variables: List[str] = Field(default_factory=list)
    tools_called: List[str] = Field(default_factory=list)
    artefacts: List[ArtefactRef] = Field(default_factory=list)


class Manifest(BaseModel):
    run_id: str
    created_at_utc: str
    user_query: str
    intent: Intent
    tools_called: List[str]
    variables: List[str]
    artefacts: List[Dict[str, Any]] = Field(default_factory=list)
    versions: Dict[str, str] = Field(default_factory=dict)

    # accès direct artefact par label (ex: manifest.lookup["acf_plot"] -> path)
    lookup: Dict[str, str] = Field(default_factory=dict)
