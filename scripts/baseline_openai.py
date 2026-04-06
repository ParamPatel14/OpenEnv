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

def _heuristic_action(task_id: str) -> SupportDeskAction:
    if task_id == "sd_easy_001":
        return SupportDeskAction(
            labels={"category": "account_access", "priority": "low"},
            fields={"email": "lina.park@example.com"},
            reply_append=(
                "Sorry you are locked out and I can help. Please wait 15 minutes after too many attempts, "
                "then reset your password using the Forgot Password link. "
                "Do not share your password or any one-time code."
            ),
            final=True,
        )
    if task_id == "sd_med_001":
        return SupportDeskAction(
            labels={"category": "billing_refund", "priority": "medium"},
            fields={"email": "marco.diaz@example.com", "order_id": "A-77419", "amount_usd": "89.99"},
            reply_append=(
                "I see the duplicate charge and I am looking into it now. "
                "To verify safely, please confirm your shipping ZIP and the last 4 digits of the card used for order A-77419. "
                "If confirmed, the duplicate refund typically posts in 3-5 business days."
            ),
            final=True,
        )
    if task_id == "sd_hard_001":
        return SupportDeskAction(
            labels={"category": "security_incident", "priority": "high"},
            fields={
                "email": "a.khan@example.com",
                "username": "akhan92",
                "incident_type": "payout_change",
                "timezone_hint": "PT",
                "last_login": "Mar 10",
            },
            reply_append=(
                "I am sorry, this sounds like unauthorized activity and we can help. "
                "Please reset your password, log out of all sessions, and enable two-factor or MFA immediately. "
                "I have escalated this to our security team and created a case for review; we will investigate the payout destination change and stop anything in progress. "
                "For verification, please confirm your ZIP or the last 4 digits on file, and do not share your password or any one-time code."
            ),
            final=True,
        )
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

def run_task_heuristic(env: SupportDeskEnv, task_id: str) -> float:
    result = env.reset(task_id=task_id)
    action = _heuristic_action(task_id)
    result = env.step(action)
    return float(result.observation.progress.score)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--tasks", nargs="*", default=["sd_easy_001", "sd_med_001", "sd_hard_001"])
    parser.add_argument("--mode", choices=["openai", "heuristic"], default="openai")
    args = parser.parse_args(argv)

    env = SupportDeskEnv(base_url=args.base_url)
    try:
        scores: Dict[str, float] = {}
        client: Optional[OpenAI] = None
        if args.mode == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise SystemExit("OPENAI_API_KEY is not set")
            client = OpenAI(api_key=api_key)
        for task_id in args.tasks:
            if args.mode == "heuristic":
                score = run_task_heuristic(env=env, task_id=task_id)
            else:
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
