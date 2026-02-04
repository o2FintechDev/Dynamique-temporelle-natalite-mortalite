# src/agent/schemas.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class ToolCall:
    tool_name: str
    variables: list[str]
    params: dict[str, Any]

@dataclass
class Plan:
    intent: str
    tool_calls: list[ToolCall]

@dataclass
class ArtefactRef:
    kind: str
    path: str
    label: str | None = None

    def dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": self.path, "label": self.label}

@dataclass
class ExecutionResult:
    run_id: str
    intent: str
    variables: list[str]
    tools_called: list[str]
    artefacts: list[ArtefactRef]

@dataclass
class Manifest:
    run_id: str
    created_at_utc: str
    user_query: str
    intent: str
    tools_called: list[str]
    variables: list[str]
    artefacts: list[dict[str, Any]]
    versions: dict[str, Any]
    lookup: dict[str, str]

    def dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "user_query": self.user_query,
            "intent": self.intent,
            "tools_called": self.tools_called,
            "variables": self.variables,
            "artefacts": self.artefacts,
            "versions": self.versions,
            "lookup": self.lookup,
        }
