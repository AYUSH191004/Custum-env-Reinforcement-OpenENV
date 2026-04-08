"""
Deterministic graders for all 3 tasks.
Same input always produces the same score — no randomness.
Each grader returns a Reward with value in [-1.0, 1.0] and a breakdown dict.
"""

from .models import Action, Reward


# ---------------------------------------------------------------------------
# Task 1 — Ticket Classification (Easy)
# ---------------------------------------------------------------------------

TEAM_MAP = {
    "bug":              ["engineering", "support"],
    "billing":          ["billing"],
    "feature_request":  ["customer_success", "sales"],
    "churn_signal":     ["customer_success", "sales"],
    "general_inquiry":  ["support", "customer_success"],
}

RELATED_TYPES = {
    "churn_signal":     ["general_inquiry"],
    "general_inquiry":  ["churn_signal", "feature_request"],
    "feature_request":  ["general_inquiry"],
    "bug":              [],
    "billing":          [],
}

PRIORITY_DISTANCE = {
    ("critical", "high"):   0.5,
    ("high", "critical"):   0.5,
    ("high", "medium"):     0.5,
    ("medium", "high"):     0.5,
    ("medium", "low"):      0.5,
    ("low", "medium"):      0.5,
    ("critical", "medium"): 0.0,
    ("critical", "low"):    0.0,
    ("low", "critical"):    0.0,
}


def grade_task_1(action: Action, ground_truth: dict) -> Reward:

    bd = {}
    gt_type     = ground_truth["ticket_type"]
    gt_priority = ground_truth["priority"]
    gt_teams    = TEAM_MAP.get(gt_type, ["support"])

    if action.ticket_type is None:
        bd["type"] = -0.2
    elif action.ticket_type == gt_type:
        bd["type"] = 0.40
    elif action.ticket_type in RELATED_TYPES.get(gt_type, []):
        bd["type"] = 0.15
    else:
        bd["type"] = 0.0

    if action.priority is None:
        bd["priority"] = -0.15
    elif action.priority == gt_priority:
        bd["priority"] = 0.35
    else:
        bd["priority"] = PRIORITY_DISTANCE.get(
            (action.priority, gt_priority), 0.0
        ) * 0.35

    if action.assigned_team is None:
        bd["team"] = -0.10
    elif action.assigned_team in gt_teams:
        bd["team"] = 0.25
    else:
        bd["team"] = 0.0

    total = sum(bd.values())
    total = round(max(-1.0, min(1.0, total)), 3)

    is_correct = (
        action.ticket_type == gt_type
        and action.priority == gt_priority
        and action.assigned_team in gt_teams
    )

    feedback = "Perfect." if is_correct else "Classification partially correct."

    return Reward(value=total, breakdown=bd, feedback=feedback, is_correct=is_correct)


# ---------------------------------------------------------------------------
# Task 2 — Response Drafting (Medium)
# ---------------------------------------------------------------------------

def grade_task_2(action: Action, context: dict) -> Reward:

    bd = {}
    required_keywords = context.get("required_keywords", [])
    expected_tone     = context.get("expected_tone", "formal")
    forbidden_phrases = context.get("forbidden_phrases", [])

    body = action.reply_body or ""
    body_lower = body.lower()

    if not body or len(body.strip()) < 10:
        return Reward(
            value=-0.2,
            breakdown={"exists": -0.2},
            feedback="No reply body provided.",
            is_correct=False,
        )

    bd["exists"] = 0.10

    word_count = len(body.split())

    if 30 <= word_count <= 250:
        bd["length"] = 0.15
    elif 15 <= word_count < 30:
        bd["length"] = 0.08
    else:
        bd["length"] = 0.0

    # FIX 1 — reduced free keyword score
    if required_keywords:
        hits = sum(1 for kw in required_keywords if kw.lower() in body_lower)
        bd["keywords"] = round((hits / len(required_keywords)) * 0.40, 3)
    else:
        bd["keywords"] = 0.10

    if action.reply_tone == expected_tone:
        bd["tone"] = 0.25
    else:
        bd["tone"] = 0.12

    violations = sum(
        1 for ph in forbidden_phrases if ph.lower() in body_lower
    )

    bd["professionalism"] = max(0.0, 0.10 - violations * 0.05)

    total = sum(bd.values())
    total = round(max(-1.0, min(1.0, total)), 3)

    is_correct = total >= 0.75

    return Reward(
        value=total,
        breakdown=bd,
        feedback="Response evaluated.",
        is_correct=is_correct,
    )


# ---------------------------------------------------------------------------
# Task 3 — Churn Detection
# ---------------------------------------------------------------------------

RETENTION_MAP = {
    "high":   ["schedule_call", "offer_discount"],
    "medium": ["send_feature_highlight", "schedule_call"],
    "low":    ["no_action"],
}


def grade_task_3(action: Action, ground_truth: dict) -> Reward:

    bd = {}

    gt_score = ground_truth["churn_risk_score"]
    gt_tier  = ground_truth["risk_tier"]

    if action.churn_risk_score is None:
        bd["score"] = -0.2
    else:
        distance = abs(action.churn_risk_score - gt_score)

        if distance <= 0.1:
            bd["score"] = 0.45
        elif distance <= 0.2:
            bd["score"] = 0.30
        else:
            bd["score"] = 0.10

    acceptable = RETENTION_MAP.get(gt_tier, [])

    if action.retention_action in acceptable:
        bd["action"] = 0.45
    else:
        bd["action"] = 0.10

    bd["reasoning"] = 0.10

    total = sum(bd.values())
    total = round(max(-1.0, min(1.0, total)), 3)

    is_correct = total >= 0.75

    return Reward(
        value=total,
        breakdown=bd,
        feedback="Churn evaluated.",
        is_correct=is_correct,
    )


# ---------------------------------------------------------------------------
# Task 4 — Escalation Decision
# ---------------------------------------------------------------------------

DIFFICULTY_BONUS = {
    "easy": 0.0,
    "medium": 0.05,
    "hard": 0.10,
}


def grade_task_4(action: Action, ground_truth: dict, difficulty="medium") -> Reward:

    bd = {}

    expected = ground_truth["escalation_decision"]

    diff_bonus = DIFFICULTY_BONUS.get(difficulty, 0.0)

    # FIX 2 — prevent perfect score inflation
    if action.escalation_decision == expected:
        bd["decision"] = round(0.85 + diff_bonus, 3)
        is_correct = True
    else:
        bd["decision"] = 0.3
        is_correct = False

    total = sum(bd.values())

    return Reward(
        value=round(max(-1.0, min(1.0, total)), 3),
        breakdown=bd,
        feedback="Escalation evaluated.",
        is_correct=is_correct,
    )


# ---------------------------------------------------------------------------
# Difficulty Bonus
# ---------------------------------------------------------------------------

def apply_difficulty_bonus(reward: Reward, difficulty: str) -> Reward:

    if not reward.is_correct:
        return reward

    bonus = DIFFICULTY_BONUS.get(difficulty, 0.0)

    return Reward(
        value=min(1.0, reward.value + bonus),
        breakdown=reward.breakdown,
        feedback=reward.feedback,
        is_correct=reward.is_correct,
    )