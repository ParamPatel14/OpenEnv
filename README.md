---
title: SupportDesk Env (OpenEnv)
emoji: ""
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - agents
  - evaluation
---

# SupportDesk Env (OpenEnv)

SupportDesk Env is a real-world, production-inspired environment for training and evaluating AI agents on customer support work:

- Ticket triage (category + priority)
- Structured information extraction
- Safe, policy-compliant reply drafting

The environment follows the OpenEnv interface (`reset()` / `step()` / `state()`), exposes typed models, and ships as a containerized Hugging Face Space.

## What makes it “real-world”

The tasks model common support workflows:

- Routing decisions (category/priority) that affect incident response
- Extracting key identifiers (email, order ID, amount, device, timestamps)
- Writing a reply that is correct, safe, and follows policy constraints (no sensitive data requests, no dangerous instructions)

The grader is deterministic and returns a score in `[0.0, 1.0]` with partial credit for progress.

## Action Space

The agent interacts by updating a “workspace” over multiple steps and optionally submitting.

`SupportDeskAction`

- `labels`: optional `{ "category": str, "priority": "low"|"medium"|"high" }`
- `fields`: optional dict of extracted fields (e.g., `email`, `order_id`, `amount_usd`)
- `reply_append`: optional string appended to the current draft reply
- `final`: optional bool, when true ends the episode and triggers final grading

## Observation Space

`SupportDeskObservation`

- `task`: task metadata (task id, difficulty, objective)
- `ticket`: the current support ticket (subject, from, body, received_at)
- `workspace`: current agent progress (labels, extracted fields, reply draft)
- `progress`: current scores and which rubric items are satisfied
- `instructions`: concise policy + formatting requirements

## Tasks (Easy → Medium → Hard)

Each task has a deterministic grader (score 0.0–1.0).

1. Easy: account access / password reset
2. Medium: billing refund with required identifiers
3. Hard: suspected account compromise requiring secure handling and escalation

## Reward Function

At each step the environment computes the score of the current workspace and provides reward as:

`reward = max(0, new_score - previous_score) - step_cost - safety_penalties`

This produces dense reward for partial progress (correct label, correct fields, required reply elements) and penalizes unsafe behavior (asking for full card numbers, credentials, or other disallowed content).

## Local Setup

```bash
python -m pip install -e .
python -m uvicorn supportdesk_env.server.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Docker

```bash
docker build -t supportdesk-env -f server/Dockerfile .
docker run -p 8000:8000 supportdesk-env
```

## Baseline (OpenAI)

The baseline runs a fixed policy prompt, uses deterministic generation settings, and reports reproducible grader scores for all tasks.

Set:

```bash
setx OPENAI_API_KEY "..."
```

Run:

```bash
python scripts/baseline_openai.py --base-url http://localhost:8000 --model gpt-4o-mini
```

## Expected Baseline Scores

Baseline scores depend on the model. With deterministic settings (`temperature=0`) you should see stable results run-to-run on the same model.

