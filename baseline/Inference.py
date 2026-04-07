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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from openai import OpenAI
from OpenEnv import CustomerSupportEnv, Action


# ---------------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

assert API_BASE_URL is not None, "API_BASE_URL missing"
assert MODEL is not None, "MODEL_NAME missing"
assert HF_TOKEN is not None, "HF_TOKEN missing"


# ---------------------------------------------------------------------------
# OpenAI Client
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

temperature = 0.0


TASK_IDS = [
    "task_1_ticket_classification",
    "task_2_response_drafting",
    "task_3_churn_detection",
]


# ---------------------------------------------------------------------------
# System Prompt
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
# Task Formatters
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
# Run Single Task
# ---------------------------------------------------------------------------

def run_task(task_id: str, max_steps: int = 4, verbose: bool = False):

    logger.info(f"[STEP] Starting task: {task_id}")

    env = CustomerSupportEnv(task_id=task_id)
    obs = env.reset()

    total = 0.0
    steps = 0

    formatter = FORMATTERS[task_id]

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

            # Remove markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                raw = raw.replace("json", "").strip()

            try:
                parsed = json.loads(raw)

                # SAFE Tone Normalization (Task 2 only)
                if task_id == "task_2_response_drafting":
                    if isinstance(parsed, dict) and "reply_tone" in parsed:

                        tone = parsed["reply_tone"].lower().strip()

                        if tone in ["professional", "business"]:
                            parsed["reply_tone"] = "formal"

                        elif tone in ["sorry", "apology", "apologize"]:
                            parsed["reply_tone"] = "apologetic"

                        elif tone in ["casual", "helpful"]:
                            parsed["reply_tone"] = "friendly"

                        elif tone in ["critical", "important"]:
                            parsed["reply_tone"] = "urgent"

            except Exception:
                logger.info(f"JSON parse failed: {raw}")
                parsed = {}

            action = Action(**parsed)

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

    print("[START] Running OpenEnv Baseline")

    scores = {}

    for task_id in TASK_IDS:

        print(f"[STEP] Running {task_id}")

        if verbose:
            print("\nRunning:", task_id)

        score = run_task(task_id, verbose=verbose)
        scores[task_id] = score

    print("[END] Baseline completed")

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
        print("[START] Running Single Task")
        score = run_task(args.task, verbose=args.verbose)
        print({args.task: score})
        print("[END] Single Task Completed")
    else:
        scores = run_baseline(verbose=args.verbose)
        print(scores)