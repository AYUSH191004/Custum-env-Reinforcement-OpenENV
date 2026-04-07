"""
Baseline inference script.
Runs Groq LLM as an agent against all 3 tasks and returns reproducible scores.

Usage:
    python -m baseline.run_baseline
    # or triggered via GET /baseline
"""

import os
import sys
import json
import logging 

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from openai import OpenAI
from environment import CustomerSupportEnv, Action


# ---------------------------------------------------------------------------
# Groq Client
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama3-8b-8192"
temperature = 0.0


TASK_IDS = [
    "task_1_ticket_classification",
    "task_2_response_drafting",
    "task_3_churn_detection",
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert SaaS customer support AI.

Your goal:
Maximize reward by making accurate decisions.

Rules:
- Return ONLY valid JSON
- Use EXACT allowed values
- No explanations
- No markdown

Guidelines:
- Downgrade/cancel → churn_signal
- Billing complaints → billing
- Bugs/errors → bug
- Feature requests → feature_request

Be accurate and conservative.
Follow the rules precisely.
"""


# ---------------------------------------------------------------------------
# Task formatters
# ---------------------------------------------------------------------------

def format_task_1(obs: dict) -> str:
    return f"""TASK: Ticket Classification

Subject: {obs['subject']}
Customer: {obs['customer_name']} (Plan: {obs['plan']}, MRR: ${obs['mrr']})
Body: {obs['body']}

Guidelines:
- Cancellation or downgrade intent → churn_signal
- Payment or invoice issues → billing
- Errors or system failures → bug
- Feature requests → feature_request
- General questions → general_inquiry

Example 1:

Customer: "We're thinking of cancelling"
Output:
{{
  "ticket_type": "churn_signal",
  "priority": "critical",
  "assigned_team": "customer_success"
}}

Example 2:

Customer: "I was charged twice"
Output:
{{
  "ticket_type": "billing",
  "priority": "high",
  "assigned_team": "billing"
}}

Return JSON:
{{
  "ticket_type": "bug" | "billing" | "feature_request" | "churn_signal" | "general_inquiry",
  "priority": "critical" | "high" | "medium" | "low",
  "assigned_team": "engineering" | "billing" | "customer_success" | "sales" | "support"
}}
"""


def format_task_2(obs: dict) -> str:
    return f"""TASK: Response Drafting

Subject: {obs['subject']}
Customer: {obs['customer_name']}
Body: {obs['body']}

Guidelines:
- Negative sentiment → apologetic
- Urgent issue → urgent
- General support → friendly
- Business request → formal

Example:

Customer: frustrated with product crashing

Output:
{{
  "reply_body": "We're sorry for the inconvenience. Our team is investigating the issue and will update you shortly.",
  "reply_tone": "apologetic"
}}

Tone Examples:
- billing complaint → apologetic
- feature question → friendly
- account request → formal
- service outage → urgent

Return JSON:
{{
  "reply_body": "<text>",
  "reply_tone": "formal" | "friendly" | "apologetic" | "urgent"
}}
"""


def format_task_3(obs: dict) -> str:
    return f"""TASK: Churn Detection

Subject: {obs['subject']}
Customer: {obs['customer_name']}
Body: {obs['body']}

Guidelines:
- Downgrade or cancellation intent → high churn risk
- Negative sentiment → higher risk
- High MRR customers → prioritize retention
- Positive engagement → low churn risk

Example 1:

Customer: "We're considering downgrading"

Output:
{{
  "churn_risk_score": 0.85,
  "retention_action": "schedule_call"
}}

Example 2:

Customer: "Just exploring features"

Output:
{{
  "churn_risk_score": 0.2,
  "retention_action": "no_action"
}}

Return JSON:
{{
  "churn_risk_score": <float 0.0-1.0>,
  "retention_action":
    "offer_discount" |
    "schedule_call" |
    "flag_account_manager" |
    "send_feature_highlight" |
    "no_action"
}}
"""


FORMATTERS = {
    "task_1_ticket_classification": format_task_1,
    "task_2_response_drafting": format_task_2,
    "task_3_churn_detection": format_task_3,
}


# ---------------------------------------------------------------------------
# Tone Normalization (Safe improvement)
# ---------------------------------------------------------------------------

def normalize_tone(tone):

    if not tone:
        return tone

    tone = tone.lower().strip()

    if tone in ["professional", "business", "formal tone"]:
        return "formal"

    if tone in ["sorry", "apology", "apologize", "apologetic tone"]:
        return "apologetic"

    if tone in ["friendly tone", "casual", "helpful"]:
        return "friendly"

    if tone in ["critical", "important", "immediate"]:
        return "urgent"

    return tone


def safe_action(task_id, parsed):

    try:

        if not isinstance(parsed, dict):
            return Action()

        # Task 2
        if task_id == "task_2_response_drafting":
            if "reply_body" in parsed and "reply_tone" in parsed:
                return Action(**parsed)
            return Action()

        # Task 1
        if task_id == "task_1_ticket_classification":
            if "ticket_type" in parsed:
                return Action(**parsed)
            return Action()

        # Task 3
        if task_id == "task_3_churn_detection":
            if "churn_risk_score" in parsed:
                return Action(**parsed)
            return Action()

        return Action()

    except Exception:
        return Action()
# ---------------------------------------------------------------------------
# Run Single Task
# ---------------------------------------------------------------------------

def run_task(task_id: str, max_steps: int = 5, verbose: bool = False):

    env = CustomerSupportEnv(task_id=task_id)
    obs = env.reset()

    total = 0.0
    steps = 0

    formatter = FORMATTERS[task_id]

    logger.info(f"Starting task: {task_id}")

    while not env.done and steps < max_steps:

        prompt = formatter(obs.model_dump())

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=600,
            )

            raw = response.choices[0].message.content.strip()

            logger.info(f"RAW OUTPUT: {raw}")

            # Remove markdown
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                raw = raw.replace("json", "").strip()

            try:

                parsed = json.loads(raw)

                # Normalize tone
                if task_id == "task_2_response_drafting":
                    if "reply_tone" in parsed:
                        parsed["reply_tone"] = normalize_tone(parsed["reply_tone"])

            except Exception as e:

                logger.info(f"JSON parse failed: {raw}")
                parsed = {}

            # SAFE ACTION (UPDATED)
            action = safe_action(task_id, parsed)

            logger.info(f"Parsed Action: {action}")

        except Exception as e:

            logger.info(f"LLM Error: {e}")
            action = Action()

        obs, reward, done, info = env.step(action)

        logger.info(f"Reward: {reward.value}")
        logger.info(f"Done: {done}")

        total += reward.value
        steps += 1

    avg = round(total / max(steps, 1), 3)

    logger.info(f"Task {task_id} finished with avg reward: {avg}")

    return avg


# ---------------------------------------------------------------------------
# Run All Tasks
# ---------------------------------------------------------------------------

def run_baseline(verbose: bool = False):

    scores = {}

    for task_id in TASK_IDS:

        if verbose:
            print("\nRunning:", task_id)

        score = run_task(task_id, verbose=verbose)
        scores[task_id] = score

    return scores


# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--task", "-t", default=None)

    args = parser.parse_args()

    if args.task:
        score = run_task(args.task, verbose=args.verbose)
        print({args.task: score})
    else:
        scores = run_baseline(verbose=args.verbose)
        print(scores)