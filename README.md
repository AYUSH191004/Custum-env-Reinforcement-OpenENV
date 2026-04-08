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

# 🎧 Customer Support Escalation & Churn Detection — OpenEnv

**A Real-World Multi-Stage Reinforcement Learning Environment for Customer Support AI**

This OpenEnv environment simulates a **production SaaS customer support pipeline** where AI agents must:

- 📩 Classify incoming tickets  
- ✍️ Draft contextual replies  
- ⚠️ Detect churn risk  
- 🚨 Decide escalation strategy  

Designed for **RL training, agent evaluation, and real-world reasoning benchmarks**.

---

# 🏆 Baseline Performance

Latest Deterministic Baseline:

| Task | Description | Score |
|------|-------------|-------|
| Task 1 | Ticket Classification | **0.845** |
| Task 2 | Response Drafting | **0.774** |
| Task 3 | Churn Detection | **0.840** |
| Task 4 | Escalation Decision | **0.790** |

## 📊 Overall Baseline

### Difficulty Profile

| Task | Difficulty |
|------|------------|
| Task 1 | Easy |
| Task 2 | Medium |
| Task 3 | Hard |
| Task 4 | Medium-Hard |

Balanced for **frontier model evaluation**.

---

# 🌍 Why This Environment Matters

Customer support is one of the **highest-impact AI automation domains**:

- Millions of tickets daily  
- Revenue impact via churn prevention  
- Complex reasoning requirements  

This environment enables:

✅ Real-world RL evaluation  
✅ Multi-step reasoning  
✅ Business logic modeling  
✅ Deterministic benchmarking  

---

# 🧠 Tasks Overview

| Task | ID | Description |
|------|----|-------------|
| Task 1 | `task_1_ticket_classification` | Classify ticket type, priority, assign team |
| Task 2 | `task_2_response_drafting` | Draft contextual customer reply |
| Task 3 | `task_3_churn_detection` | Predict churn risk and retention action |
| Task 4 | `task_4_escalation_decision` | Decide escalation vs auto-resolve |

---

# 🎯 Multi-Stage Environment Design

This environment models a **real support lifecycle**
Ticket
↓
Classification
↓
Response Drafting
↓
Churn Detection
↓
Escalation Decision

This enables **agentic multi-step reasoning**.

---

# 🔎 Observation Space

Each observation contains:

| Field | Description |
|------|-------------|
| ticket_id | Ticket identifier |
| subject | Email subject |
| body | Ticket content |
| customer_name | Customer name |
| plan | free / starter / pro / enterprise |
| account_age_days | Account age |
| mrr | Monthly recurring revenue |
| open_tickets_count | Active tickets |
| last_login_days_ago | Usage signal |
| previous_sentiment | Customer sentiment |
| difficulty | easy / medium / hard |
| context | Task-specific hints |

Simulates **real SaaS support dashboards**.

---

# 🎮 Action Space

## Task 1 — Classification

```json
{
"ticket_type": "bug",
"priority": "high",
"assigned_team": "engineering"
}
Task 2 — Response Drafting
{
"reply_body": "We apologize for the inconvenience...",
"reply_tone": "apologetic"
}
Task 3 — Churn Detection
{
"churn_risk_score": 0.82,
"retention_action": "schedule_call"
}
Task 4 — Escalation Decision
{
"escalation_decision": "escalate_to_human"
}
🏅 Reward Design

Deterministic reward shaping:

Task	Reward
Task 1	Classification accuracy
Task 2	Tone + quality + keywords
Task 3	Risk prediction accuracy
Task 4	Escalation reasoning


Ensures stable performance across:

Local runs
Docker
HuggingFace Spaces
🔌 API Endpoints
Endpoint	Method	Description
/	GET	Health check
/reset	GET	Start episode
/step	POST	Submit action
/state	GET	Inspect environment
/tasks	GET	List tasks
/baseline	GET	Run evaluation
🚀 Quick Start
Local Setup
pip install -r requirements.txt
python app.py

Open:

http://localhost:7860
🐳 Docker
docker build -t support-env .
docker run -p 7860:7860 support-env
🔬 Example Usage

Reset environment:

curl http://localhost:7860/reset

Run baseline:

curl http://localhost:7860/baseline
🧪 Use Cases

This environment is useful for:

RL agent benchmarking
Agentic workflow evaluation
Customer support automation research
Multi-step reasoning evaluation
🎯 OpenEnv Compliance

Fully compliant:

reset()
step()
state()
baseline()
deterministic rewards
reproducible scoring
🏗 Architecture
OpenEnv/
├── env.py
├── graders.py
├── models.py
├── inference.py
├── app.py
├── data/

Clean modular design:

Environment logic
Reward grading
Baseline inference
API interface
🔥 Key Features
Multi-task RL environment
Real-world SaaS domain
Deterministic evaluation
Difficulty progression
Robust parsing
Stable baseline