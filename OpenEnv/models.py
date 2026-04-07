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
    open_tickets_count: int            # how many tickets this customer has open
    last_login_days_ago: int           # days since last product login
    previous_sentiment: str            # positive | neutral | negative
    task_id: str
    step_number: int
    context: Dict[str, Any] = {}      # task-specific grading metadata (hidden hints)


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

    @field_validator("churn_risk_score", mode="before")
    @classmethod
    def round_churn_score(cls, v):
        if v is not None:
            return round(float(v), 3)
        return v


class Reward(BaseModel):
    """Feedback signal returned after every step."""
    value: float = Field(ge=-1.0, le=1.0)
    breakdown: Dict[str, float] = {}   # component-level scores
    feedback: str = ""                 # human-readable explanation
    is_correct: bool = False           # did the agent get the core decision right?

    @field_validator("value", mode="before")
    @classmethod
    def clamp_and_round(cls, v):
        return round(max(-1.0, min(1.0, float(v))), 3)
