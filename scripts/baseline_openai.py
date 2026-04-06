from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from supportdesk_env import SupportDeskAction, SupportDeskEnv


SYSTEM_PROMPT = (
    "You are a customer support agent operating inside an environment.\n"
    "Your job is to: (1) set correct labels (category + priority), (2) extract required fields, "
    "(3) draft a safe, helpful reply.\n"
    "Safety rules: never ask for passwords, one-time codes, or full card numbers/CVV.\n"
    "Return only a JSON object matching this schema:\n"
    "{\n"
    '  "labels": {"category": "...", "priority": "low|medium|high"} | null,\n'
    '  "fields": {"key": "value", ...} | null,\n'
    '  "reply_append": "string" | null,\n'
    '  "final": true|false\n'
    "}\n"
)


def _compact_obs(obs: Any) -> Dict[str, Any]:
    return {
        "task": obs.task.model_dump(mode="json"),
        "ticket": obs.ticket.model_dump(mode="json"),
        "workspace": obs.workspace.model_dump(mode="json"),
        "progress": obs.progress.model_dump(mode="json"),
        "instructions": obs.instructions,
    }


def _parse_action(text: str) -> SupportDeskAction:
    try:
        payload = json.loads(text.strip())
        return SupportDeskAction.model_validate(payload)
    except Exception:
        return SupportDeskAction(final=True)


def run_task(client: OpenAI, env: SupportDeskEnv, model: str, task_id: str) -> float:
    result = env.reset(task_id=task_id)
    for _ in range(result.observation.task.max_steps):
        if result.done:
            break
        user_payload = _compact_obs(result.observation)
        msg = json.dumps(user_payload, ensure_ascii=False)
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            top_p=1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": msg},
            ],
        )
        text = resp.choices[0].message.content or ""
        action = _parse_action(text)
        result = env.step(action)
        if action.final:
            break
    return float(result.observation.progress.score)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--tasks", nargs="*", default=["sd_easy_001", "sd_med_001", "sd_hard_001"])
    args = parser.parse_args(argv)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    env = SupportDeskEnv(base_url=args.base_url)
    try:
        scores: Dict[str, float] = {}
        for task_id in args.tasks:
            score = run_task(client=client, env=env, model=args.model, task_id=task_id)
            scores[task_id] = score
            print(f"{task_id}: {score:.3f}")
        avg = sum(scores.values()) / max(1, len(scores))
        print(f"average: {avg:.3f}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

