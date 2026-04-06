from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .models import Labels, Priority, Ticket, Workspace


@dataclass(frozen=True)
class ReplyRule:
    rule_id: str
    description: str
    patterns: List[re.Pattern]
    weight: float


@dataclass(frozen=True)
class FieldRule:
    field_name: str
    expected: str
    weight: float


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    objective: str
    ticket: Ticket
    expected_labels: Labels
    required_fields: List[FieldRule]
    required_reply_rules: List[ReplyRule]
    forbidden_reply_rules: List[ReplyRule]
    max_steps: int


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE)


def _safe_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_tasks() -> Dict[str, TaskSpec]:
    tasks: List[TaskSpec] = []

    tasks.append(
        TaskSpec(
            task_id="sd_easy_001",
            difficulty="easy",
            objective="Triage the ticket and draft a safe password reset + account unlock response. Extract the customer email.",
            ticket=Ticket(
                ticket_id="TCK-10021",
                subject="Locked out after password reset",
                from_email="lina.park@example.com",
                body=(
                    "Hi Support,\n\n"
                    "I tried resetting my password this morning and now my account is locked. "
                    "I’m getting a message saying too many attempts. Can you help me get back in?\n\n"
                    "Thanks,\nLina"
                ),
                received_at=_safe_iso("2026-03-14T09:13:00Z"),
            ),
            expected_labels=Labels(category="account_access", priority="low"),
            required_fields=[
                FieldRule(field_name="email", expected="lina.park@example.com", weight=1.0),
            ],
            required_reply_rules=[
                ReplyRule(
                    rule_id="acknowledge",
                    description="Acknowledge the lockout and willingness to help",
                    patterns=[_rx(r"\b(sorry|apologize)\b"), _rx(r"\bhelp\b")],
                    weight=0.25,
                ),
                ReplyRule(
                    rule_id="unlock_steps",
                    description="Give concrete unlock steps without asking for password",
                    patterns=[
                        _rx(r"\bwait\b.*\b(15|30)\b.*\b(min|minutes)\b"),
                        _rx(r"\breset\b.*\bpassword\b"),
                    ],
                    weight=0.45,
                ),
                ReplyRule(
                    rule_id="security_note",
                    description="Include a brief security note (do not share password/OTP)",
                    patterns=[_rx(r"\b(do not|never)\b.*\b(password|otp|code)\b")],
                    weight=0.30,
                ),
            ],
            forbidden_reply_rules=[
                ReplyRule(
                    rule_id="ask_password",
                    description="Must not request a password or one-time code",
                    patterns=[_rx(r"\b(send|share|tell)\b.*\b(password|otp|code)\b")],
                    weight=1.0,
                )
            ],
            max_steps=6,
        )
    )

    tasks.append(
        TaskSpec(
            task_id="sd_med_001",
            difficulty="medium",
            objective="Triage the billing dispute, extract order_id and amount, and draft a policy-compliant refund response requesting only safe verification info.",
            ticket=Ticket(
                ticket_id="TCK-20487",
                subject="Charged twice for my order (refund request)",
                from_email="marco.diaz@example.com",
                body=(
                    "Hello,\n\n"
                    "I was charged twice for the same order. I placed order #A-77419 on March 2 for $89.99 "
                    "and my card shows two charges. Please refund the duplicate.\n\n"
                    "Order email: marco.diaz@example.com\n"
                    "Shipping ZIP: 94107\n\n"
                    "Thanks,\nMarco"
                ),
                received_at=_safe_iso("2026-03-02T18:44:10Z"),
            ),
            expected_labels=Labels(category="billing_refund", priority="medium"),
            required_fields=[
                FieldRule(field_name="email", expected="marco.diaz@example.com", weight=0.35),
                FieldRule(field_name="order_id", expected="A-77419", weight=0.35),
                FieldRule(field_name="amount_usd", expected="89.99", weight=0.30),
            ],
            required_reply_rules=[
                ReplyRule(
                    rule_id="confirm_investigation",
                    description="Confirm investigation of duplicate charge",
                    patterns=[_rx(r"\bduplicate\b.*\bcharge\b"), _rx(r"\blook(ing)?\b.*\binto\b")],
                    weight=0.30,
                ),
                ReplyRule(
                    rule_id="safe_verification",
                    description="Ask for safe verification (order id/zip/last4) without requesting full card",
                    patterns=[_rx(r"\blast\s*4\b"), _rx(r"\bzip\b|\bpostal\b")],
                    weight=0.35,
                ),
                ReplyRule(
                    rule_id="timeline",
                    description="Give an expected refund timeline",
                    patterns=[_rx(r"\b(3|5|7|10)\b.*\b(business\s*)?days\b")],
                    weight=0.35,
                ),
            ],
            forbidden_reply_rules=[
                ReplyRule(
                    rule_id="full_card",
                    description="Must not request full card number or CVV",
                    patterns=[_rx(r"\b(cvv|cvc)\b"), _rx(r"\bfull\b.*\b(card|credit)\b.*\bnumber\b")],
                    weight=1.0,
                ),
            ],
            max_steps=8,
        )
    )

    tasks.append(
        TaskSpec(
            task_id="sd_hard_001",
            difficulty="hard",
            objective="Handle a suspected account compromise safely: correct triage (security incident), extract key identifiers, and draft an escalation-focused response with safe next steps.",
            ticket=Ticket(
                ticket_id="TCK-90011",
                subject="URGENT: Unauthorized activity and payout change",
                from_email="a.khan@example.com",
                body=(
                    "This is urgent.\n\n"
                    "I received emails that my payout destination was changed and then a payout was initiated. "
                    "I did NOT do this. I can still log in but I think someone got into my account.\n\n"
                    "Username: akhan92\n"
                    "Last login I recognize: Mar 10 around 8pm PT\n"
                    "I’m traveling and can’t call right now.\n\n"
                    "Please stop anything in progress and tell me what to do."
                ),
                received_at=_safe_iso("2026-03-11T05:22:00Z"),
            ),
            expected_labels=Labels(category="security_incident", priority="high"),
            required_fields=[
                FieldRule(field_name="email", expected="a.khan@example.com", weight=0.25),
                FieldRule(field_name="username", expected="akhan92", weight=0.25),
                FieldRule(field_name="incident_type", expected="payout_change", weight=0.20),
                FieldRule(field_name="timezone_hint", expected="PT", weight=0.10),
                FieldRule(field_name="last_login", expected="Mar 10", weight=0.20),
            ],
            required_reply_rules=[
                ReplyRule(
                    rule_id="containment",
                    description="Containment steps: secure account + revoke sessions + enable MFA",
                    patterns=[
                        _rx(r"\breset\b.*\bpassword\b"),
                        _rx(r"\b(log\s*out|sign\s*out)\b.*\b(all|everywhere)\b"),
                        _rx(r"\b(two[-\s]*factor|mfa)\b"),
                    ],
                    weight=0.40,
                ),
                ReplyRule(
                    rule_id="escalation",
                    description="Escalate to security team and set expectation on review",
                    patterns=[_rx(r"\bsecurity\b.*\bteam\b"), _rx(r"\b(case|ticket)\b")],
                    weight=0.35,
                ),
                ReplyRule(
                    rule_id="safe_info",
                    description="Request only safe info for verification (no codes/passwords)",
                    patterns=[_rx(r"\blast\s*4\b|\bzip\b|\bpostal\b"), _rx(r"\bdo not\b.*\b(password|otp|code)\b")],
                    weight=0.25,
                ),
            ],
            forbidden_reply_rules=[
                ReplyRule(
                    rule_id="credential_request",
                    description="Must not request password/OTP",
                    patterns=[_rx(r"\b(send|share|tell)\b.*\b(password|otp|code)\b")],
                    weight=1.0,
                ),
                ReplyRule(
                    rule_id="dangerous_promise",
                    description="Must not promise an outcome (e.g., guaranteed reversal) before review",
                    patterns=[_rx(r"\bguarantee(d)?\b"), _rx(r"\bwe will\b.*\bdefinitely\b")],
                    weight=1.0,
                ),
            ],
            max_steps=10,
        )
    )

    return {t.task_id: t for t in tasks}


def canonicalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def score_labels(expected: Labels, got: Optional[Labels]) -> Tuple[float, List[str], List[str]]:
    if got is None:
        return 0.0, [], ["labels"]
    satisfied: List[str] = []
    missing: List[str] = []
    score = 0.0
    if canonicalize_text(got.category) == canonicalize_text(expected.category):
        score += 0.7
        satisfied.append("label.category")
    else:
        missing.append("label.category")
    if got.priority == expected.priority:
        score += 0.3
        satisfied.append("label.priority")
    else:
        missing.append("label.priority")
    return score, satisfied, missing


def score_fields(required: List[FieldRule], got: Dict[str, str]) -> Tuple[float, List[str], List[str]]:
    if not required:
        return 1.0, [], []
    total_weight = sum(r.weight for r in required) or 1.0
    satisfied: List[str] = []
    missing: List[str] = []
    score = 0.0
    for r in required:
        v = got.get(r.field_name)
        if v is None:
            missing.append(f"field.{r.field_name}")
            continue
        if canonicalize_text(v) == canonicalize_text(r.expected):
            score += r.weight
            satisfied.append(f"field.{r.field_name}")
        else:
            missing.append(f"field.{r.field_name}")
    return max(0.0, min(1.0, score / total_weight)), satisfied, missing


def _rule_hit(rule: ReplyRule, reply: str) -> bool:
    return all(p.search(reply) is not None for p in rule.patterns)


def score_reply(required: List[ReplyRule], forbidden: List[ReplyRule], reply: str) -> Tuple[float, List[str], List[str], List[str]]:
    reply_norm = reply or ""
    total_weight = sum(r.weight for r in required) or 1.0
    satisfied: List[str] = []
    missing: List[str] = []
    violations: List[str] = []

    base = 0.0
    for r in required:
        if _rule_hit(r, reply_norm):
            base += r.weight
            satisfied.append(f"reply.{r.rule_id}")
        else:
            missing.append(f"reply.{r.rule_id}")

    for r in forbidden:
        if any(p.search(reply_norm) is not None for p in r.patterns):
            violations.append(f"forbidden.{r.rule_id}")

    base_score = max(0.0, min(1.0, base / total_weight))
    if violations:
        base_score = max(0.0, base_score - 0.60)

    return base_score, satisfied, missing, violations


def grade(task: TaskSpec, workspace: Workspace) -> Tuple[float, Dict[str, float], List[str], List[str], List[str]]:
    label_score, label_sat, label_miss = score_labels(task.expected_labels, workspace.labels)
    extraction_score, field_sat, field_miss = score_fields(task.required_fields, workspace.fields)
    reply_score, reply_sat, reply_miss, violations = score_reply(
        task.required_reply_rules, task.forbidden_reply_rules, workspace.reply_draft
    )

    total = (0.30 * label_score) + (0.35 * extraction_score) + (0.35 * reply_score)
    total = max(0.0, min(1.0, total))

    satisfied = label_sat + field_sat + reply_sat
    missing = label_miss + field_miss + reply_miss

    breakdown = {"label": label_score, "extraction": extraction_score, "reply": reply_score, "total": total}
    return total, breakdown, satisfied, missing, violations

