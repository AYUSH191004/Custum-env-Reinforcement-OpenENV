"""
CustomerSupportEnv — the core OpenEnv environment class.
Implements the full OpenEnv interface: reset(), step(), state().
"""

import json
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

from .models import Observation, Action, Reward
from .graders import grade_task_1, grade_task_2, grade_task_3

TASK_IDS = [
    "task_1_ticket_classification",
    "task_2_response_drafting",
    "task_3_churn_detection",
]

MAX_STEPS = {
    "task_1_ticket_classification": 4,
    "task_2_response_drafting":     4,
    "task_3_churn_detection":       4,
}

DATA_PATH = Path(__file__).parent / "data" / "tickets.json"


class CustomerSupportEnv:
    """
    A real-world customer support triage environment.

    The agent reads support tickets and must:
      Task 1 (easy)   — classify type, priority, assign team
      Task 2 (medium) — draft an appropriate reply
      Task 3 (hard)   — detect churn risk and choose retention action
    """

    def __init__(self, task_id: str = "task_1_ticket_classification"):
        if task_id not in TASK_IDS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from {TASK_IDS}")

        self.task_id  = task_id
        self.max_steps = MAX_STEPS[task_id]

        # Load and filter dataset for this task
        all_tickets = json.loads(DATA_PATH.read_text())
        self.tickets = [t for t in all_tickets if t["task_id"] == task_id]
        if not self.tickets:
            raise RuntimeError(f"No tickets found for task_id '{task_id}'")

        # Episode state — initialised properly by reset()
        self.step_number:    int   = 0
        self.done:           bool  = True   # must call reset() first
        self.total_reward:   float = 0.0
        self.correct_count:  int   = 0
        self.history:        List  = []
        self.current_ticket: Optional[dict] = None
        self._seen_ids:      set   = set()

    # ------------------------------------------------------------------
    # Public OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Start a new episode. Returns the first observation."""
        self.step_number   = 0
        self.done          = False
        self.total_reward  = 0.0
        self.correct_count = 0
        self.history       = []
        self._seen_ids     = set()
        self.current_ticket = self._sample_ticket()
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process one action. Returns (observation, reward, done, info).
        Raises RuntimeError if called after episode is done.
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Grade the action against the current ticket's ground truth
        reward = self._grade(action)

        # Track history
        self.total_reward  += reward.value
        self.correct_count += int(reward.is_correct)
        self.history.append({
            "step":      self.step_number,
            "ticket_id": self.current_ticket["ticket_id"],
            "action":    action.model_dump(exclude_none=True),
            "reward":    reward.model_dump(),
        })
        self.step_number += 1

        # Check terminal conditions
        if self.step_number >= self.max_steps or not self._tickets_remaining():
            self.done = True

        # Advance to next ticket (or stay on last if done)
        if not self.done:
            self.current_ticket = self._sample_ticket()

        obs  = self._make_observation()
        info = {
            "total_reward":  round(self.total_reward, 3),
            "steps":         self.step_number,
            "correct_count": self.correct_count,
            "done":          self.done,
        }
        return obs, reward, self.done, info

    def state(self) -> Dict[str, Any]:
        """Return a full snapshot of the environment state."""
        return {
            "task_id":         self.task_id,
            "step_number":     self.step_number,
            "max_steps":       self.max_steps,
            "done":            self.done,
            "total_reward":    round(self.total_reward, 3),
            "correct_count":   self.correct_count,
            "accuracy":        round(self.correct_count / max(self.step_number, 1), 3),
            "avg_reward":      round(self.total_reward / max(self.step_number, 1), 3),
            "current_ticket_id": (
                self.current_ticket["ticket_id"] if self.current_ticket else None
            ),
            "tickets_seen":    list(self._seen_ids),
            "history":         self.history,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_ticket(self) -> dict:
        """Sample a ticket not yet seen in this episode."""
        available = [t for t in self.tickets if t["ticket_id"] not in self._seen_ids]
        if not available:
            # Cycle — reset seen set but keep history
            self._seen_ids = set()
            available = self.tickets
        ticket = random.choice(available)
        self._seen_ids.add(ticket["ticket_id"])
        return ticket

    def _tickets_remaining(self) -> bool:
        seen = len(self._seen_ids)
        return seen < len(self.tickets)

    def _make_observation(self) -> Observation:
        t = self.current_ticket

        # Strip ground_truth from context before sending to agent
        agent_context = {
            k: v for k, v in t.get("context", {}).items()
            if k not in ("expected_retention_actions",)
            # churn_signals are shown as hints; ground truth numbers are hidden
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
            task_id             = self.task_id,
            step_number         = self.step_number,
            context             = agent_context,
        )

    def _grade(self, action: Action) -> Reward:
        t  = self.current_ticket
        gt = t["ground_truth"]
        ctx = t.get("context", {})

        if self.task_id == "task_1_ticket_classification":
            return grade_task_1(action, gt)

        elif self.task_id == "task_2_response_drafting":
            return grade_task_2(action, ctx)

        elif self.task_id == "task_3_churn_detection":
            return grade_task_3(action, gt)

        raise ValueError(f"No grader for task_id '{self.task_id}'")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CustomerSupportEnv("
            f"task={self.task_id}, "
            f"step={self.step_number}/{self.max_steps}, "
            f"reward={self.total_reward:.3f}, "
            f"done={self.done})"
        )
