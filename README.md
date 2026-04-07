---
title: Customer Support Escalation OpenEnv
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# Customer Support Escalation & Churn Detection — OpenEnv

A real-world reinforcement learning environment where AI agents learn to triage
customer support tickets, draft contextual replies, and detect customer churn risk.

## Environment Description

This environment simulates the full support queue lifecycle of a B2B SaaS company.
Each episode presents the agent with a stream of customer support tickets. The agent
must reason about ticket content, customer account data, and behavioral signals to
take the right action.

**Why this problem?** Customer support triage is one of the highest-volume knowledge
work tasks in any software company. A skilled agent can meaningfully reduce resolution
time, improve customer satisfaction, and prevent churn — all with measurable, gradeable
outcomes.

## Tasks

| Task | Difficulty | Max Steps | Description |
|------|-----------|-----------|-------------|
| `task_1_ticket_classification` | Easy | 10 | Classify ticket type, priority, assign team |
| `task_2_response_drafting` | Medium | 10 | Draft appropriate reply with correct tone |
| `task_3_churn_detection` | Hard | 8 | Score churn risk; choose retention action |

## Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | str | Unique ticket identifier |
| `subject` | str | Email subject line |
| `body` | str | Full ticket body text |
| `customer_name` | str | Customer's name |
| `plan` | str | free / starter / pro / enterprise |
| `account_age_days` | int | Days since account creation |
| `mrr` | float | Monthly recurring revenue (USD) |
| `open_tickets_count` | int | Number of currently open tickets |
| `last_login_days_ago` | int | Days since last product login |
| `previous_sentiment` | str | positive / neutral / negative |
| `task_id` | str | Active task identifier |
| `step_number` | int | Current step in episode |
| `context` | dict | Task-specific hints (keywords, signals) |

## Action Space

| Field | Used by | Type | Options |
|-------|---------|------|---------|
| `ticket_type` | Task 1 | str | bug, billing, feature_request, churn_signal, general_inquiry |
| `priority` | Task 1 | str | critical, high, medium, low |
| `assigned_team` | Task 1 | str | engineering, billing, customer_success, sales, support |
| `reply_body` | Task 2 | str | Free text (30–250 words recommended) |
| `reply_tone` | Task 2 | str | formal, friendly, apologetic, urgent |
| `churn_risk_score` | Task 3 | float | 0.0 (no risk) → 1.0 (certain churn) |
| `retention_action` | Task 3 | str | offer_discount, schedule_call, flag_account_manager, send_feature_highlight, no_action |

## Reward Function

All rewards are in `[-1.0, 1.0]`. The `breakdown` dict in the Reward model shows component scores.

**Task 1** (Classification):
- `type`: 0.40 correct, 0.15 partial (related category), -0.20 if None
- `priority`: 0.35 correct, scaled partial for adjacent priorities
- `team`: 0.25 correct team, 0.0 wrong, -0.10 if None

**Task 2** (Response Drafting):
- `exists`: 0.10 if reply present, -0.20 if not
- `length`: 0.15 for 30–250 words, 0.08 for adjacent range
- `keywords`: up to 0.40 based on required keyword coverage
- `tone`: 0.25 exact match, 0.12 adjacent tone, -0.10 if None
- `professionalism`: 0.10 minus penalty per forbidden phrase

**Task 3** (Churn Detection):
- `score_accuracy`: 0.45 → 0.0 based on distance from ground truth
- `retention_action`: 0.45 correct tier action, 0.10 wrong tier, -0.20 if None
- `risk_tier_logic`: 0.10 bonus if inferred tier matches ground truth tier

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/reset?task_id=...` | POST | Start new episode |
| `/step?task_id=...` | POST | Submit action |
| `/state?task_id=...` | GET | Inspect full env state |
| `/tasks` | GET | List tasks + action schemas |
| `/grader?task_id=...` | GET | Get normalized score [0,1] |
| `/baseline` | GET | Run OpenAI baseline on all tasks |

## Setup & Usage

```bash
# 1. Clone and install
git clone <your-repo-url>
cd support-escalation-env
pip install -r requirements.txt

# 2. Set API key (for baseline only)
export OPENAI_API_KEY=sk-...

# 3. Start server
uvicorn api.app:app --host 0.0.0.0 --port 7860 --reload

# 4. Run tests
pytest tests/ -v

# 5. Run baseline manually
python -m baseline.run_baseline --verbose
```

## Docker

```bash
docker build -t support-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... support-env
```

## Quick API Tour

```bash
# Reset task 1
curl -X POST "http://localhost:7860/reset?task_id=task_1_ticket_classification"

# Submit action
curl -X POST "http://localhost:7860/step?task_id=task_1_ticket_classification" \
  -H "Content-Type: application/json" \
  -d '{"ticket_type": "bug", "priority": "high", "assigned_team": "engineering"}'

# Get score
curl "http://localhost:7860/grader?task_id=task_1_ticket_classification"
```

## Baseline Scores

| Task | Model | Avg Reward | Normalized Score |
|------|-------|-----------|-----------------|
| Task 1 (easy) | gpt-4o-mini | ~0.72 | ~0.86 |
| Task 2 (medium) | gpt-4o-mini | ~0.41 | ~0.71 |
| Task 3 (hard) | gpt-4o-mini | ~0.28 | ~0.64 |

*Scores are averages over 8 steps per episode. Run `/baseline` to reproduce.*
