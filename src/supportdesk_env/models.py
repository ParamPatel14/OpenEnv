from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Priority = Literal["low", "medium", "high"]


class Labels(BaseModel):
    category: str = Field(min_length=1)
    priority: Priority


class Ticket(BaseModel):
    ticket_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    from_email: str = Field(min_length=3)
    body: str = Field(min_length=1)
    received_at: datetime


class Workspace(BaseModel):
    labels: Optional[Labels] = None
    fields: Dict[str, str] = Field(default_factory=dict)
    reply_draft: str = ""


class Progress(BaseModel):
    score: float = 0.0
    label_score: float = 0.0
    extraction_score: float = 0.0
    reply_score: float = 0.0
    satisfied: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)


class TaskInfo(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    max_steps: int = 8


class SupportDeskObservation(BaseModel):
    task: TaskInfo
    ticket: Ticket
    workspace: Workspace
    progress: Progress
    instructions: str


class SupportDeskAction(BaseModel):
    labels: Optional[Labels] = None
    fields: Optional[Dict[str, str]] = None
    reply_append: Optional[str] = None
    final: bool = False


class SupportDeskState(BaseModel):
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    step_count: int = 0
    done: bool = False
    score: float = 0.0
    max_steps: int = 0


class SupportDeskReward(BaseModel):
    value: float
    delta_score: float = 0.0
    step_cost: float = 0.0
    safety_penalty: float = 0.0


class StepResult(BaseModel):
    observation: SupportDeskObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
