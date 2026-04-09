"""
Baseline inference script.
Runs Groq LLM as an agent against all 4 tasks and returns reproducible scores.
"""


import os
import sys
import json
import logging
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from openai import OpenAI
from OpenEnv.env import CustomerSupportEnv, Action

# ---------------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL = os.getenv("MODEL_NAME", "llama3-8b-8192")
HF_TOKEN = os.getenv("HF_TOKEN")

assert API_BASE_URL is not None
assert MODEL is not None
assert HF_TOKEN is not None


client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

TASK_IDS = [
    "task_1_ticket_classification",
    "task_2_response_drafting",
    "task_3_churn_detection",
    "task_4_escalation_decision",
]
# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT =  """
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
# Tone Normalization
# ---------------------------------------------------------------------------

def normalize_tone(tone):

    if not tone:
        return None

    tone = str(tone).lower().strip()

    mapping = {
        "professional": "formal",
        "business": "formal",
        "sorry": "apologetic",
        "apology": "apologetic",
        "empathetic": "apologetic",
        "casual": "friendly",
        "helpful": "friendly",
        "critical": "urgent",
        "important": "urgent",
    }

    if tone in ["formal","friendly","apologetic","urgent"]:
        return tone

    return mapping.get(tone, "friendly")


# ---------------------------------------------------------------------------
# Robust JSON Extraction
# ---------------------------------------------------------------------------

def extract_json(raw):

    if not raw:
        return {}

    raw = raw.strip()

    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if "{" in part and "}" in part:
                raw = part
                break

    try:
        return json.loads(raw)
    except:
        pass

    try:
        start = raw.find("{")
        end = raw.rfind("}")

        if start != -1 and end != -1:
            return json.loads(raw[start:end+1])
    except:
        pass

    return {}


# ---------------------------------------------------------------------------
# Task-Specific Sanitization
# ---------------------------------------------------------------------------

def sanitize_action(parsed, task_id):

    if not isinstance(parsed, dict):
        return {}

    if task_id == "task_1_ticket_classification":
        return {
            "ticket_type": parsed.get("ticket_type"),
            "priority": parsed.get("priority"),
            "assigned_team": parsed.get("assigned_team"),
        }

    if task_id == "task_2_response_drafting":
        return {
            "reply_body": parsed.get("reply_body"),
            "reply_tone": normalize_tone(parsed.get("reply_tone")),
        }

    if task_id == "task_3_churn_detection":

        score = parsed.get("churn_risk_score")

        try:
            score = float(score) if score is not None else None
        except:
            score = None

        return {
            "churn_risk_score": score,
            "retention_action": parsed.get("retention_action"),
        }

    if task_id == "task_4_escalation_decision":
        return {
            "escalation_decision": parsed.get("escalation_decision"),
        }

    return parsed


# ---------------------------------------------------------------------------
# Task Formatters (unchanged)
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

def format_task_4(obs: dict) -> str:
    return f"""TASK: Escalation Decision
Subject: {obs['subject']}
Customer: {obs['customer_name']}
Body: {obs['body']}
Decision Rules:
- Critical bug / outage → escalate_to_human
- Ambiguous issue → request_more_info
- Simple request → auto_resolve
Return JSON:
{{
  "escalation_decision":
    "auto_resolve" |
    "escalate_to_human" |
    "request_more_info"
}}
"""


FORMATTERS = {
    "task_1_ticket_classification": format_task_1,
    "task_2_response_drafting": format_task_2,
    "task_3_churn_detection": format_task_3,
    "task_4_escalation_decision": format_task_4,
}


# ---------------------------------------------------------------------------
# Run Task
# ---------------------------------------------------------------------------

def run_task(task_id, verbose=False):

    env = CustomerSupportEnv(task_id=task_id)
    obs = env.reset()

    total = 0
    steps = 0
    max_steps = env.max_steps

    formatter = FORMATTERS[task_id]

    print(f"[START] task={task_id}", flush=True)

    while not env.done and steps < max_steps:

        prompt = formatter(obs.model_dump())

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role":"system","content":"Return JSON only"},
                    {"role":"user","content":prompt},
                ],
                temperature=0,
                max_tokens=600
            )

            raw = response.choices[0].message.content

            logger.info(f"RAW: {raw}")

            parsed = extract_json(raw)
            parsed = sanitize_action(parsed, task_id)

            try:
                action = Action(**parsed)
            except:
                action = Action()

        except Exception as e:
            logger.info(f"LLM Error: {e}")
            action = Action()

        obs, reward, done, info = env.step(action)

        total += reward.value
        steps += 1

    print(
            f"[STEP] step={steps} reward={round(reward.value,3)}",
            flush=True
        )
    
    score = round(total / max(steps,1),3)

    print(
        f"[END] task={task_id} score={score} steps={steps}",
        flush=True
    )

    return score

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def run_baseline():

    scores = {}

    for task in TASK_IDS:
        scores[task] = run_task(task)

    return scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(run_baseline())



