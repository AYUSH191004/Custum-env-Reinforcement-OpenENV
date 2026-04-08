"""
FastAPI server exposing the CustomerSupportEnv via HTTP.
All 6 OpenEnv-required endpoints are implemented here.

Start with:
    uvicorn api.app:app --host 0.0.0.0 --port 7860 --reload
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OpenEnv import CustomerSupportEnv, Action


# ---------------------------------------------------------------------------
# Shared state: one env instance per task_id
# ---------------------------------------------------------------------------
_envs: Dict[str, CustomerSupportEnv] = {}

TASK_IDS = [
    "task_1_ticket_classification",
    "task_2_response_drafting",
    "task_3_churn_detection",
]


def get_env(task_id: str) -> CustomerSupportEnv:
    """Retrieve (or lazily create) the environment for a given task."""
    if task_id not in TASK_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid options: {TASK_IDS}",
        )

    if task_id not in _envs:
        _envs[task_id] = CustomerSupportEnv(task_id=task_id)

    return _envs[task_id]


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm all environments
    for tid in TASK_IDS:
        env = CustomerSupportEnv(task_id=tid)
        env.reset()
        _envs[tid] = env
    yield


app = FastAPI(
    title="Customer Support Escalation — OpenEnv",
    description=(
        "A real-world RL environment where AI agents learn to classify support tickets, "
        "draft contextual replies, and detect customer churn risk."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Normalize score
# ---------------------------------------------------------------------------
def normalize_score(avg_reward: float) -> float:
    """Map avg reward [-1, 1] → [0, 1]."""
    return round(max(0.0, min(1.0, (avg_reward + 1.0) / 2.0)), 3)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/", tags=["health"])
def root():
    return {
        "status": "ok",
        "env": "customer-support-escalation-env",
        "version": "1.0.0",
        "tasks": TASK_IDS,
    }


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
@app.get("/reset", tags=["openenv"])
def reset(task_id: str = Query(default="task_1_ticket_classification")):
    env = get_env(task_id)
    obs = env.reset()
    return obs.model_dump()


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------
@app.post("/step", tags=["openenv"])
def step(
    action: Action,
    task_id: str = Query(default="task_1_ticket_classification"),
):
    env = get_env(task_id)

    if env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode finished. Call /reset first.",
        )

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@app.get("/state", tags=["openenv"])
def state(task_id: str = Query(default="task_1_ticket_classification")):
    env = get_env(task_id)
    return env.state()


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
@app.get("/tasks", tags=["openenv"])
def list_tasks():
    return {
        "tasks": [
            {
                "id": "task_1_ticket_classification",
                "difficulty": "easy",
                "description": "Classify ticket type and assign team",
                "max_steps": 10,
            },
            {
                "id": "task_2_response_drafting",
                "difficulty": "medium",
                "description": "Draft contextual reply",
                "max_steps": 10,
            },
            {
                "id": "task_3_churn_detection",
                "difficulty": "hard",
                "description": "Detect churn risk",
                "max_steps": 8,
            },
        ]
    }


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------
@app.get("/grader", tags=["openenv"])
def grader(task_id: str = Query(default="task_1_ticket_classification")):
    env = get_env(task_id)
    s = env.state()

    avg_reward = s["avg_reward"]
    normalized = normalize_score(avg_reward)

    return {
        "task_id": task_id,
        "score": normalized,
        "accuracy": s["accuracy"],
        "total_reward": s["total_reward"],
        "avg_reward": avg_reward,
        "steps_taken": s["step_number"],
        "correct_count": s["correct_count"],
        "done": s["done"],
    }


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
@app.get("/baseline", tags=["openenv"])
def baseline():
    try:
        from baseline.Inference import run_baseline

        scores = run_baseline()

        return {
            "baseline_model": "llama-3.3-70b-versatile",
            "scores": scores,
            "avg_score": round(sum(scores.values()) / len(scores), 3),
        }

    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))