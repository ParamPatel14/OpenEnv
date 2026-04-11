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

## Security Measures

This environment implements several production-grade security measures out-of-the-box:

1. **API Key Authentication**: You can lock down the environment endpoints by setting the `SUPPORTDESK_API_KEY` environment variable. When set, all requests must include this key in the `X-API-Key` header. The Python client will automatically pick it up and pass it if set in the environment.
2. **CORS Controls**: CORS is configured by default. You can restrict the allowed origins using the `ALLOWED_ORIGINS` environment variable (e.g., `ALLOWED_ORIGINS="https://my-domain.com"`).
3. **Security Headers**: All HTTP responses include strict security headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Strict-Transport-Security`, `Content-Security-Policy`).

## Local Setup

```bash
python -m pip install -e .
python -m uvicorn supportdesk_env.server.app:app --host 0.0.0.0 --port 8000
```

Validate:

```bash
openenv validate
```

Health check:

```bash
curl http://localhost:8000/health
```

## Docker

```bash
docker build -t supportdesk-env .
docker run -p 8000:8000 supportdesk-env
```

## Deploy to Hugging Face Spaces

```bash
openenv push --repo-id <your-username>/supportdesk-env
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

Heuristic (no API key):

```bash
python scripts/baseline_openai.py --base-url http://localhost:8000 --mode heuristic
```

## Expected Baseline Scores

Baseline scores depend on the model. With deterministic settings (`temperature=0`) you should see stable results run-to-run on the same model.

Heuristic baseline (`--mode heuristic`, fully deterministic):

- `sd_easy_001`: 0.790
- `sd_med_001`: 1.000
- `sd_hard_001`: 0.790
- average: 0.860
