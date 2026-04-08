from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List, Literal


class Observation(BaseModel):
    """What the agent sees at each step."""
    ticket_id: str
    subject: str
    body: str
    customer_name: str
    plan: str                          # free | starter | pro | enterprise
    account_age_days: int
    mrr: float                         # monthly recurring revenue in USD
    open_tickets_count: int
    last_login_days_ago: int
    previous_sentiment: str            # positive | neutral | negative
    difficulty: str                    # easy | medium | hard  [Improvement 2]
    task_id: str
    step_number: int
    context: Dict[str, Any] = {}


class Action(BaseModel):
    """What the agent submits per step. All fields Optional — task determines which are used."""

    # Task 1: Classification
    ticket_type: Optional[Literal[
        "bug", "billing", "feature_request", "churn_signal", "general_inquiry"
    ]] = None
    priority: Optional[Literal["critical", "high", "medium", "low"]] = None
    assigned_team: Optional[Literal[
        "engineering", "billing", "customer_success", "sales", "support"
    ]] = None

    # Task 2: Response drafting
    reply_body: Optional[str] = None
    reply_tone: Optional[Literal["formal", "friendly", "apologetic", "urgent"]] = None

    # Task 3: Churn detection
    churn_risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    retention_action: Optional[Literal[
        "offer_discount", "schedule_call", "flag_account_manager",
        "send_feature_highlight", "no_action"
    ]] = None

    # Task 4: Escalation decision  [Improvement 1]
    escalation_decision: Optional[Literal[
        "auto_resolve", "escalate_to_human", "request_more_info"
    ]] = None

    @field_validator("churn_risk_score", mode="before")
    @classmethod
    def round_churn_score(cls, v):
        if v is not None:
            return round(float(v), 3)
        return v


class Reward(BaseModel):
    """Feedback signal returned after every step."""
    value: float = Field(ge=-1.0, le=1.0)
    breakdown: Dict[str, float] = {}
    feedback: str = ""
    is_correct: bool = False

    @field_validator("value", mode="before")
    @classmethod
    def clamp_and_round(cls, v):
        return round(max(-1.0, min(1.0, float(v))), 3)
