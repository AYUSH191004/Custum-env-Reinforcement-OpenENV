"""
FastAPI server exposing CustomerSupportEnv via HTTP.
All 6 OpenEnv-required endpoints implemented.

Start: uvicorn api.app:app --host 0.0.0.0 --port 7860 --reload
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OpenEnv.env import CustomerSupportEnv, Action

_envs: dict = {}

TASK_IDS = [
    "task_1_ticket_classification",
    "task_2_response_drafting",
    "task_3_churn_detection",
    "task_4_escalation_decision",       
]


def get_env(task_id: str) -> CustomerSupportEnv:
    if task_id not in TASK_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: {TASK_IDS}",
        )
    if task_id not in _envs:
        _envs[task_id] = CustomerSupportEnv(task_id=task_id)
    return _envs[task_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    for tid in TASK_IDS:
        env = CustomerSupportEnv(task_id=tid)
        env.reset()
        _envs[tid] = env
    yield


app = FastAPI(
    title="Customer Support Escalation — OpenEnv",
    description=(
        "Real-world RL environment: classify tickets, draft replies, "
        "detect churn, decide escalation path."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_score(avg_reward: float) -> float:
    """Map avg reward [-1, 1] → [0, 1]."""
    return round(max(0.0, min(1.0, (avg_reward + 1.0) / 2.0)), 3)


# ── Endpoint 1: Health ──────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    return {
        "status":  "ok",
        "env":     "customer-support-escalation-env",
        "version": "1.1.0",
        "tasks":   TASK_IDS,
    }


# ── Endpoint 2: GET /reset ─────────────────────────────────────────────────
@app.post("/reset", tags=["openenv"])
def reset(task_id: str = Query(default="task_1_ticket_classification")):
    """Start a new episode. Returns the first Observation."""
    env = get_env(task_id)
    obs = env.reset()
    return obs.model_dump()


# ── Endpoint 3: POST /step ──────────────────────────────────────────────────
@app.post("/step", tags=["openenv"])
def step(
    action: Action,
    task_id: str = Query(default="task_1_ticket_classification"),
):
    """Submit one action. Returns {observation, reward, done, info}."""
    env = get_env(task_id)
    if env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode finished. Call POST /reset first.",
        )
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


# ── Endpoint 4: GET /state ──────────────────────────────────────────────────
@app.get("/state", tags=["openenv"])
def state(task_id: str = Query(default="task_1_ticket_classification")):
    """Full environment state including difficulty and history."""
    env = get_env(task_id)
    return env.state()


# ── Endpoint 5: GET /tasks ──────────────────────────────────────────────────
@app.get("/tasks", tags=["openenv"])
def list_tasks():
    """All 4 tasks with action schemas for the validator."""
    return {
        "tasks": [
            {
                "id":          "task_1_ticket_classification",
                "difficulty":  "easy",
                "description": "Classify ticket type, priority, assign team.",
                "max_steps":   10,
                "action_schema": {
                    "ticket_type":   "str: bug|billing|feature_request|churn_signal|general_inquiry",
                    "priority":      "str: critical|high|medium|low",
                    "assigned_team": "str: engineering|billing|customer_success|sales|support",
                },
            },
            {
                "id":          "task_2_response_drafting",
                "difficulty":  "medium",
                "description": "Draft a contextually appropriate reply.",
                "max_steps":   10,
                "action_schema": {
                    "reply_body": "str: full reply text (30–250 words)",
                    "reply_tone": "str: formal|friendly|apologetic|urgent",
                },
            },
            {
                "id":          "task_3_churn_detection",
                "difficulty":  "hard",
                "description": "Score churn risk and select retention action.",
                "max_steps":   8,
                "action_schema": {
                    "churn_risk_score":  "float: 0.0–1.0",
                    "retention_action":  "str: offer_discount|schedule_call|flag_account_manager|send_feature_highlight|no_action",
                },
            },
            {
                "id":          "task_4_escalation_decision",
                "difficulty":  "medium",
                "description": "Decide whether to auto-resolve, escalate, or request more info.",
                "max_steps":   4,
                "action_schema": {
                    "escalation_decision": "str: auto_resolve|escalate_to_human|request_more_info",
                },
            },
        ]
    }


# ── Endpoint 6: GET /grader ─────────────────────────────────────────────────
@app.get("/grader", tags=["openenv"])
def grader(task_id: str = Query(default="task_1_ticket_classification")):
    """Normalized [0,1] score for completed or in-progress episode."""
    env = get_env(task_id)
    s   = env.state()
    return {
        "task_id":       task_id,
        "score":         normalize_score(s["avg_reward"]),
        "accuracy":      s["accuracy"],
        "total_reward":  s["total_reward"],
        "avg_reward":    s["avg_reward"],
        "steps_taken":   s["step_number"],
        "correct_count": s["correct_count"],
        "done":          s["done"],
    }


# ── Endpoint 7: GET /baseline ───────────────────────────────────────────────
@app.get("/baseline", tags=["openenv"])
def baseline():
    """Run baseline on all tasks."""
    try:
        from inference import run_baseline
        scores = run_baseline()
        return {
            "baseline_model": "llama-3.3-70b-versatile",
            "scores": scores,
            "avg_score": round(sum(scores.values()) / len(scores), 3),
        }

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Baseline import error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline failed: {str(e)}")