from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

EvidenceKind = Literal["figure", "table", "metric", "model", "narrative"]

class EvidenceRef(BaseModel):
    kind: EvidenceKind
    path: str
    label: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

class Paragraph(BaseModel):
    text_md: str
    evidence: List[EvidenceRef] = Field(default_factory=list)
    importance: Literal["core", "support"] = "core"

class Section(BaseModel):
    title: str
    paragraphs: List[Paragraph] = Field(default_factory=list)

class Chapter(BaseModel):
    title: str
    sections: List[Section] = Field(default_factory=list)

class NarrativePacket(BaseModel):
    run_id: str
    chapters: List[Chapter] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
