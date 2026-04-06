from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from .models import (
    Progress,
    SupportDeskReward,
    StepResult,
    SupportDeskAction,
    SupportDeskObservation,
    SupportDeskState,
    TaskInfo,
    Workspace,
)
from .tasks import TaskSpec, grade, load_tasks


class SupportDeskEnvironment:
    def __init__(self, task_id: Optional[str] = None):
        self._tasks = load_tasks()
        self._task: Optional[TaskSpec] = None
        self._workspace = Workspace()
        self._state = SupportDeskState()
        self._last_score = 0.0
        self._default_task_id = task_id

    @property
    def state(self) -> SupportDeskState:
        return self._state

    def reset(self, task_id: Optional[str] = None) -> StepResult:
        chosen = task_id or self._default_task_id or "sd_easy_001"
        if chosen not in self._tasks:
            raise ValueError(f"Unknown task_id: {chosen}")
        self._task = self._tasks[chosen]
        self._workspace = Workspace()
        self._last_score = 0.0
        self._state = SupportDeskState(
            episode_id=str(uuid.uuid4()),
            task_id=self._task.task_id,
            step_count=0,
            done=False,
            score=0.0,
            max_steps=self._task.max_steps,
        )
        obs, info = self._build_observation()
        return StepResult(observation=obs, reward=0.0, done=False, info=info)

    def step(self, action: SupportDeskAction) -> StepResult:
        if self._task is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        if self._state.done:
            obs, info = self._build_observation()
            return StepResult(observation=obs, reward=0.0, done=True, info=info)

        self._state.step_count += 1
        self._apply_action(action)

        score, breakdown, satisfied, missing, violations = grade(self._task, self._workspace)
        delta_score = max(0.0, score - self._last_score)
        step_cost = 0.01
        safety_penalty = 0.10 if violations else 0.0
        reward_value = max(0.0, min(1.0, delta_score - step_cost - safety_penalty))

        self._last_score = score
        done = bool(action.final) or (score >= 0.95) or (self._state.step_count >= self._task.max_steps)
        self._state.done = done
        self._state.score = score

        obs, info = self._build_observation(
            progress=Progress(
                score=score,
                label_score=breakdown["label"],
                extraction_score=breakdown["extraction"],
                reply_score=breakdown["reply"],
                satisfied=satisfied,
                missing=missing,
                violations=violations,
            )
        )
        info["breakdown"] = breakdown
        info["violations"] = violations
        info["episode_id"] = self._state.episode_id
        info["reward_details"] = SupportDeskReward(
            value=reward_value, delta_score=delta_score, step_cost=step_cost, safety_penalty=safety_penalty
        ).model_dump(mode="json")
        return StepResult(observation=obs, reward=reward_value, done=done, info=info)

    def _apply_action(self, action: SupportDeskAction) -> None:
        if action.labels is not None:
            self._workspace.labels = action.labels
        if action.fields:
            for k, v in action.fields.items():
                if v is None:
                    continue
                self._workspace.fields[str(k)] = str(v)
        if action.reply_append:
            current = self._workspace.reply_draft
            new = (current + ("\n" if current else "") + action.reply_append).strip()
            self._workspace.reply_draft = new[:4000]

    def _build_observation(self, progress: Optional[Progress] = None) -> tuple[SupportDeskObservation, Dict[str, Any]]:
        if self._task is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        score, breakdown, satisfied, missing, violations = grade(self._task, self._workspace)
        p = progress or Progress(
            score=score,
            label_score=breakdown["label"],
            extraction_score=breakdown["extraction"],
            reply_score=breakdown["reply"],
            satisfied=satisfied,
            missing=missing,
            violations=violations,
        )
        obs = SupportDeskObservation(
            task=TaskInfo(
                task_id=self._task.task_id,
                difficulty=self._task.difficulty,
                objective=self._task.objective,
                max_steps=self._task.max_steps,
            ),
            ticket=self._task.ticket,
            workspace=self._workspace,
            progress=p,
            instructions=(
                "You are a customer support agent. Update labels, extract key fields, and draft a helpful reply.\n"
                "Safety: never ask for passwords, one-time codes, or full card numbers/CVV.\n"
                "Be concise, professional, and include clear next steps. Use only the information in the ticket."
            ),
        )
        info: Dict[str, Any] = {"state": self._state.model_dump(mode="json")}
        return obs, info
