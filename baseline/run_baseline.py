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
# Groq Client (Free API)
# ---------------------------------------------------------------------------
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"  # fast + free
temperature = 0.0  # deterministic output


TASK_IDS = [
    "task_1_ticket_classification",
    "task_2_response_drafting",
    "task_3_churn_detection",
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """SYSTEM_PROMPT = You are a strict JSON generator.

Rules:
- Return ONLY valid JSON
- No markdown, no explanation
- Use EXACT allowed values
- Do NOT invent new values
- If unsure, choose closest valid option

Invalid output will break the system. Follow the rules precisely."""""


# ---------------------------------------------------------------------------
# Task formatters
# ---------------------------------------------------------------------------

def format_task_1(obs: dict) -> str:
    return f"""TASK: Ticket Classification

Subject: {obs['subject']}
Customer: {obs['customer_name']} (Plan: {obs['plan']}, MRR: ${obs['mrr']})
Body: {obs['body']}

Return JSON:
{{
  "ticket_type": "bug" | "billing" | "feature_request" | "churn_signal" | "general_inquiry",
  "priority": "critical" | "high" | "medium" | "low",
  "assigned_team": "engineering" | "billing" | "customer_success" | "sales" | "support"
}}"""


def format_task_2(obs: dict) -> str:
    return f"""TASK: Response Drafting

Subject: {obs['subject']}
Customer: {obs['customer_name']}
Body: {obs['body']}

Return JSON:
{{
  "reply_body": "<text>",
  "reply_tone": "formal" | "friendly" | "apologetic" | "urgent"
}}"""


def format_task_3(obs: dict) -> str:
    return f"""TASK: Churn Detection

Subject: {obs['subject']}
Customer: {obs['customer_name']}
Body: {obs['body']}

Return JSON:
{{
  "churn_risk_score": <float 0.0-1.0>,
  "retention_action":
    "offer_discount" |
    "schedule_call" |
    "flag_account_manager" |
    "send_feature_highlight" |
    "no_action"
}}"""


FORMATTERS = {
    "task_1_ticket_classification": format_task_1,
    "task_2_response_drafting": format_task_2,
    "task_3_churn_detection": format_task_3,
}


# ---------------------------------------------------------------------------
# Run Single Task
# ---------------------------------------------------------------------------

def run_task(task_id: str, max_steps: int = 8, verbose: bool = False) -> float:
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

            # Remove markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                raw = raw.replace("json", "").strip()

            try:
                parsed = json.loads(raw)
            except Exception as e:
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

    scores = {}

    for task_id in TASK_IDS:
        if verbose:
            print("\nRunning:", task_id)

        score = run_task(task_id, verbose=verbose)
        scores[task_id] = score

    print("Action:", task_id)
    print("Reward:", scores[task_id])

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