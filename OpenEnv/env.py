# (FULL FILE — Only Required Changes Applied)

"""
CustomerSupportEnv — OpenEnv-compliant environment.
Implements: reset(), step(), state()

Improvements:
  [1] task_4_escalation_decision added
  [2] difficulty field exposed in Observation + state()
  [3] customer_reaction added to step() info dict
"""

import json
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

from .models import Observation, Action, Reward
from .graders import (
    grade_task_1, grade_task_2, grade_task_3,
    grade_task_4, apply_difficulty_bonus,
)

TASK_IDS = [
    "task_1_ticket_classification",
    "task_2_response_drafting",
    "task_3_churn_detection",
    "task_4_escalation_decision",
]

MAX_STEPS = {
    "task_1_ticket_classification": 10,
    "task_2_response_drafting":     10,
    "task_3_churn_detection":        8,
    "task_4_escalation_decision":    8,
}

DATA_PATH = Path(__file__).parent / "data" / "tickets.json"


# ---------------------------------------------------------------------------
# [Improvement 3] Customer reaction heuristic
# ---------------------------------------------------------------------------
def _compute_customer_reaction(action: Action, reward: Reward) -> str:
    """
    Simulate customer reaction after Task 2 (response drafting).
    Based on tone + reward quality. Used in step() info dict.
    """
    empathetic_tones = {"apologetic", "friendly"}
    tone = action.reply_tone or ""

    if reward.value >= 0.7 and tone in empathetic_tones:
        return "satisfied"
    elif reward.value >= 0.4:
        return "neutral"
    else:
        return "frustrated"


class CustomerSupportEnv:
    """
    Real-world customer support triage RL environment.

    Tasks:
      1 (easy)   — classify type, priority, assign team
      2 (medium) — draft an appropriate reply
      3 (hard)   — detect churn risk and choose retention action
      4 (medium) — decide escalation path
    """

    def __init__(self, task_id: str = "task_1_ticket_classification"):
        if task_id not in TASK_IDS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from {TASK_IDS}")

        self.task_id   = task_id
        self.max_steps = MAX_STEPS[task_id]

        all_tickets   = json.loads(DATA_PATH.read_text())
        self.tickets  = [t for t in all_tickets if t["task_id"] == task_id]
        if not self.tickets:
            raise RuntimeError(f"No tickets found for task_id '{task_id}'")

        self.step_number:    int            = 0
        self.done:           bool           = True
        self.total_reward:   float          = 0.0
        self.correct_count:  int            = 0
        self.history:        List           = []
        self.current_ticket: Optional[dict] = None
        self._seen_ids:      set            = set()

        # ✅ NEW — deterministic pointer
        self._ticket_pointer = 0

    # ------------------------------------------------------------------
    # Public OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Start a new episode. Returns the first Observation."""
        self.step_number   = 0
        self.done          = False
        self.total_reward  = 0.0
        self.correct_count = 0
        self.history       = []
        self._seen_ids     = set()

        # ✅ reset pointer for deterministic order
        self._ticket_pointer = 0

        self.current_ticket = self._sample_ticket()
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Process one action. Returns (observation, reward, done, info)."""
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        difficulty = self.current_ticket.get("difficulty", "medium")
        reward     = self._grade(action)

        # Apply difficulty bonus
        reward = apply_difficulty_bonus(reward, difficulty)

        # ✅ NEW — clamp reward for stability
        reward.value = max(0.0, min(1.0, reward.value))

        self.total_reward  += reward.value
        self.correct_count += int(reward.is_correct)

        self.history.append({
            "step":       self.step_number,
            "ticket_id":  self.current_ticket["ticket_id"],
            "difficulty": difficulty,
            "action":     action.model_dump(exclude_none=True),
            "reward":     reward.model_dump(),
        })

        self.step_number += 1

        if self.step_number >= self.max_steps or not self._tickets_remaining():
            self.done = True

        if not self.done:
            self.current_ticket = self._sample_ticket()

        obs  = self._make_observation()

        customer_reaction = (
            _compute_customer_reaction(action, reward)
            if self.task_id == "task_2_response_drafting"
            else None
        )

        info: Dict[str, Any] = {
            "total_reward":      round(self.total_reward, 3),
            "steps":             self.step_number,
            "correct_count":     self.correct_count,
            "done":              self.done,
            "difficulty":        difficulty,
        }

        if customer_reaction is not None:
            info["customer_reaction"] = customer_reaction

        return obs, reward, self.done, info

    def state(self) -> Dict[str, Any]:
        difficulty = (
            self.current_ticket.get("difficulty", "medium")
            if self.current_ticket else "unknown"
        )

        return {
            "task_id":            self.task_id,
            "step_number":        self.step_number,
            "max_steps":          self.max_steps,
            "done":               self.done,
            "total_reward":       round(self.total_reward, 3),
            "correct_count":      self.correct_count,
            "accuracy":           round(self.correct_count / max(self.step_number, 1), 3),
            "avg_reward":         round(self.total_reward  / max(self.step_number, 1), 3),
            "current_ticket_id":  (self.current_ticket["ticket_id"] if self.current_ticket else None),
            "current_difficulty": difficulty,
            "tickets_seen":       list(self._seen_ids),
            "history":            self.history,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ✅ Deterministic sampling instead of random.choice
    def _sample_ticket(self) -> dict:
        if self._ticket_pointer >= len(self.tickets):
            self._ticket_pointer = 0

        ticket = self.tickets[self._ticket_pointer]
        self._ticket_pointer += 1

        self._seen_ids.add(ticket["ticket_id"])
        return ticket

    def _tickets_remaining(self) -> bool:
        return len(self._seen_ids) < len(self.tickets)

    def _make_observation(self) -> Observation:
        t = self.current_ticket

        agent_context = {
            k: v for k, v in t.get("context", {}).items()
            if k not in ("expected_retention_actions",)
        }

        return Observation(
            ticket_id           = t["ticket_id"],
            subject             = t["subject"],
            body                = t["body"],
            customer_name       = t["customer_name"],
            plan                = t["plan"],
            account_age_days    = t["account_age_days"],
            mrr                 = t["mrr"],
            open_tickets_count  = t["open_tickets_count"],
            last_login_days_ago = t["last_login_days_ago"],
            previous_sentiment  = t["previous_sentiment"],
            difficulty          = t.get("difficulty", "medium"),
            task_id             = self.task_id,
            step_number         = self.step_number,
            context             = agent_context,
        )

    def _grade(self, action: Action) -> Reward:
        t   = self.current_ticket
        gt  = t["ground_truth"]
        ctx = t.get("context", {})
        diff = t.get("difficulty", "medium")

        if self.task_id == "task_1_ticket_classification":
            return grade_task_1(action, gt)
        elif self.task_id == "task_2_response_drafting":
            return grade_task_2(action, ctx)
        elif self.task_id == "task_3_churn_detection":
            return grade_task_3(action, gt)
        elif self.task_id == "task_4_escalation_decision":
            return grade_task_4(action, gt, diff)

        raise ValueError(f"No grader for task_id '{self.task_id}'")

    def __repr__(self) -> str:
        return (
            f"CustomerSupportEnv("
            f"task={self.task_id}, "
            f"step={self.step_number}/{self.max_steps}, "
            f"reward={self.total_reward:.3f}, "
            f"done={self.done})"
        )