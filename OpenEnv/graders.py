"""
Deterministic graders for all 3 tasks.
Same input always produces the same score — no randomness.
Each grader returns a Reward with value in [-1.0, 1.0] and a breakdown dict.
"""

from .models import Action, Reward


# ---------------------------------------------------------------------------
# Task 1 — Ticket Classification (Easy)
# Agent must identify: ticket_type, priority, assigned_team
# ---------------------------------------------------------------------------

# Which teams are acceptable for each ticket type
TEAM_MAP = {
    "bug":              ["engineering", "support"],
    "billing":          ["billing"],
    "feature_request":  ["customer_success", "sales"],
    "churn_signal":     ["customer_success", "sales"],
    "general_inquiry":  ["support", "customer_success"],
}

# Related types get partial credit (confusion is understandable)
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
    """
    Score: type (40%) + priority (35%) + team (25%)
    Penalties: -0.2 for None fields
    """
    bd = {}
    gt_type     = ground_truth["ticket_type"]
    gt_priority = ground_truth["priority"]
    gt_teams    = TEAM_MAP.get(gt_type, ["support"])

    # --- Ticket type (40 pts) ---
    if action.ticket_type is None:
        bd["type"] = -0.2
    elif action.ticket_type == gt_type:
        bd["type"] = 0.40
    elif action.ticket_type in RELATED_TYPES.get(gt_type, []):
        bd["type"] = 0.15    # partial — close category
    else:
        bd["type"] = 0.0

    # --- Priority (35 pts) ---
    if action.priority is None:
        bd["priority"] = -0.15
    elif action.priority == gt_priority:
        bd["priority"] = 0.35
    else:
        bd["priority"] = PRIORITY_DISTANCE.get(
            (action.priority, gt_priority), 0.0
        ) * 0.35

    # --- Assigned team (25 pts) ---
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

    feedback_parts = []
    if action.ticket_type != gt_type:
        feedback_parts.append(f"type: got '{action.ticket_type}', expected '{gt_type}'")
    if action.priority != gt_priority:
        feedback_parts.append(f"priority: got '{action.priority}', expected '{gt_priority}'")
    if action.assigned_team not in gt_teams:
        feedback_parts.append(f"team: got '{action.assigned_team}', accepted {gt_teams}")
    feedback = "Perfect." if is_correct else "Errors — " + "; ".join(feedback_parts)

    return Reward(value=total, breakdown=bd, feedback=feedback, is_correct=is_correct)


# ---------------------------------------------------------------------------
# Task 2 — Response Drafting (Medium)
# Agent must write reply_body with correct reply_tone
# ---------------------------------------------------------------------------

def grade_task_2(action: Action, context: dict) -> Reward:
    """
    Score: reply_exists (10%) + length (15%) + keyword_coverage (40%)
           + tone_match (25%) + professionalism (10%)
    """
    bd = {}
    required_keywords = context.get("required_keywords", [])
    expected_tone     = context.get("expected_tone", "formal")
    forbidden_phrases = context.get("forbidden_phrases", [])
    body = action.reply_body or ""
    body_lower = body.lower()

    # --- Reply exists (10 pts) ---
    if not body or len(body.strip()) < 10:
        bd["exists"] = -0.20
        # If no body — other scores are all 0
        bd["length"]        = 0.0
        bd["keywords"]      = 0.0
        bd["tone"]          = 0.0
        bd["professionalism"] = 0.0
        total = sum(bd.values())
        return Reward(
            value=round(max(-1.0, min(1.0, total)), 3),
            breakdown=bd,
            feedback="No reply body provided.",
            is_correct=False,
        )

    bd["exists"] = 0.10

    # --- Length appropriateness (15 pts) ---
    word_count = len(body.split())
    if 30 <= word_count <= 250:
        bd["length"] = 0.15
    elif 15 <= word_count < 30 or 250 < word_count <= 400:
        bd["length"] = 0.08    # too short or slightly long — partial
    else:
        bd["length"] = 0.0

    # --- Keyword coverage (40 pts) ---
    if required_keywords:
        hits = sum(1 for kw in required_keywords if kw.lower() in body_lower)
        bd["keywords"] = round((hits / len(required_keywords)) * 0.40, 3)
    else:
        bd["keywords"] = 0.20  # no required keywords = free points

    # --- Tone match (25 pts) ---
    if action.reply_tone is None:
        bd["tone"] = -0.10
    elif action.reply_tone == expected_tone:
        bd["tone"] = 0.25
    else:
        # Adjacent tones get half credit
        adjacent = {
            "formal":     ["friendly"],
            "friendly":   ["formal", "apologetic"],
            "apologetic":  ["friendly", "formal"],
            "urgent":     ["formal"],
        }
        if action.reply_tone in adjacent.get(expected_tone, []):
            bd["tone"] = 0.12
        else:
            bd["tone"] = 0.0

    # --- Professionalism (10 pts): no forbidden phrases ---
    if forbidden_phrases:
        violations = sum(1 for ph in forbidden_phrases if ph.lower() in body_lower)
        bd["professionalism"] = 0.10 if violations == 0 else max(0.0, 0.10 - violations * 0.05)
    else:
        bd["professionalism"] = 0.10

    total = sum(bd.values())
    total = round(max(-1.0, min(1.0, total)), 3)

    is_correct = (
        bd["keywords"] >= 0.28        # caught ≥70% of required keywords
        and action.reply_tone == expected_tone
        and word_count >= 30
    )

    kw_pct = round(bd["keywords"] / 0.40 * 100) if 0.40 > 0 else 0
    feedback = (
        f"Keyword coverage: {kw_pct}%. "
        f"Tone: {'correct' if action.reply_tone == expected_tone else f'got {action.reply_tone!r}, expected {expected_tone!r}'}. "
        f"Words: {word_count}."
    )

    return Reward(value=total, breakdown=bd, feedback=feedback, is_correct=is_correct)


# ---------------------------------------------------------------------------
# Task 3 — Churn Detection (Hard)
# Agent must: score churn_risk_score + choose correct retention_action
# ---------------------------------------------------------------------------

# Which actions are acceptable given risk tier
RETENTION_MAP = {
    "high":   ["schedule_call", "offer_discount", "flag_account_manager"],
    "medium": ["send_feature_highlight", "schedule_call"],
    "low":    ["no_action", "send_feature_highlight"],
}


def grade_task_3(action: Action, ground_truth: dict) -> Reward:
    """
    Score: churn_score_accuracy (45%) + retention_action (45%) + reasoning (10%)
    """
    bd = {}
    gt_score   = ground_truth["churn_risk_score"]    # float 0.0–1.0
    gt_tier    = ground_truth["risk_tier"]            # high|medium|low
    acceptable = RETENTION_MAP.get(gt_tier, ["no_action"])

    # --- Churn score accuracy (45 pts) ---
    if action.churn_risk_score is None:
        bd["score_accuracy"] = -0.20
    else:
        distance = abs(action.churn_risk_score - gt_score)
        if distance <= 0.10:
            bd["score_accuracy"] = 0.45          # near-perfect
        elif distance <= 0.20:
            bd["score_accuracy"] = 0.35          # very good
        elif distance <= 0.35:
            bd["score_accuracy"] = 0.20          # acceptable
        elif distance <= 0.50:
            bd["score_accuracy"] = 0.08          # poor but tried
        else:
            bd["score_accuracy"] = 0.0           # far off

    # --- Retention action (45 pts) ---
    if action.retention_action is None:
        bd["retention_action"] = -0.20
    elif action.retention_action in acceptable:
        bd["retention_action"] = 0.45           # correct tier action
    else:
        # Wrong tier — check if it's at least a real action
        all_valid = [a for actions in RETENTION_MAP.values() for a in actions]
        if action.retention_action in all_valid:
            bd["retention_action"] = 0.10       # wrong strategy but not random
        else:
            bd["retention_action"] = 0.0

    # --- Implicit reasoning bonus (10 pts) ---
    # If score tier matches gt_tier, agent understood the signal
    if action.churn_risk_score is not None:
        inferred_tier = (
            "high"   if action.churn_risk_score >= 0.65 else
            "medium" if action.churn_risk_score >= 0.35 else
            "low"
        )
        bd["risk_tier_logic"] = 0.10 if inferred_tier == gt_tier else 0.0
    else:
        bd["risk_tier_logic"] = 0.0

    total = sum(bd.values())
    total = round(max(-1.0, min(1.0, total)), 3)

    is_correct = (
        action.retention_action in acceptable
        and action.churn_risk_score is not None
        and abs(action.churn_risk_score - gt_score) <= 0.20
    )

    feedback = (
        f"GT score={gt_score} (tier={gt_tier}). "
        f"Agent score={action.churn_risk_score}. "
        f"Action={action.retention_action!r}, accepted={acceptable}."
    )

    return Reward(value=total, breakdown=bd, feedback=feedback, is_correct=is_correct)
