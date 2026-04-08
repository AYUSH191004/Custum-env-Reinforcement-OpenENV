"""
Test suite for CustomerSupportEnv v1.1.0
Covers: all 4 tasks, difficulty field, customer_reaction, graders.

Run: pytest tests/ -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import CustomerSupportEnv, Observation, Action, Reward
from environment.graders import (
    grade_task_1, grade_task_2, grade_task_3,
    grade_task_4, apply_difficulty_bonus,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def env1():
    e = CustomerSupportEnv("task_1_ticket_classification"); e.reset(); return e

@pytest.fixture
def env2():
    e = CustomerSupportEnv("task_2_response_drafting"); e.reset(); return e

@pytest.fixture
def env3():
    e = CustomerSupportEnv("task_3_churn_detection"); e.reset(); return e

@pytest.fixture
def env4():
    e = CustomerSupportEnv("task_4_escalation_decision"); e.reset(); return e


# ── Core interface ────────────────────────────────────────────────────────────

class TestEnvInterface:
    def test_reset_returns_observation(self, env1):
        obs = env1.reset()
        assert isinstance(obs, Observation)
        assert obs.task_id == "task_1_ticket_classification"
        assert obs.step_number == 0

    def test_step_returns_correct_tuple(self, env1):
        obs, reward, done, info = env1.step(
            Action(ticket_type="bug", priority="high", assigned_team="engineering")
        )
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert "total_reward" in info and "steps" in info

    def test_state_has_required_fields(self, env1):
        s = env1.state()
        for f in ["task_id", "step_number", "done", "total_reward",
                  "correct_count", "avg_reward", "history", "current_difficulty"]:
            assert f in s, f"Missing: {f}"

    def test_step_after_done_raises(self, env1):
        env1.done = True
        with pytest.raises(RuntimeError):
            env1.step(Action())

    def test_invalid_task_id_raises(self):
        with pytest.raises(ValueError):
            CustomerSupportEnv("nonexistent_task")

    def test_all_four_tasks_construct(self):
        for tid in [
            "task_1_ticket_classification", "task_2_response_drafting",
            "task_3_churn_detection", "task_4_escalation_decision",
        ]:
            e = CustomerSupportEnv(tid)
            obs = e.reset()
            assert obs.task_id == tid

    def test_history_grows_per_step(self, env1):
        env1.step(Action(ticket_type="bug", priority="low", assigned_team="support"))
        env1.step(Action(ticket_type="billing", priority="medium", assigned_team="billing"))
        assert len(env1.history) == 2

    def test_reward_accumulates(self, env1):
        cumulative = 0.0
        for _ in range(3):
            if env1.done: break
            _, r, _, _ = env1.step(Action(ticket_type="bug", priority="high", assigned_team="engineering"))
            cumulative += r.value
        assert abs(env1.total_reward - cumulative) < 1e-6


# ── Improvement 2: Difficulty field ──────────────────────────────────────────

class TestDifficulty:
    def test_observation_has_difficulty(self, env1):
        obs = env1.reset()
        assert obs.difficulty in ("easy", "medium", "hard")

    def test_state_has_current_difficulty(self, env1):
        s = env1.state()
        assert "current_difficulty" in s
        assert s["current_difficulty"] in ("easy", "medium", "hard")

    def test_info_has_difficulty(self, env1):
        _, _, _, info = env1.step(
            Action(ticket_type="bug", priority="high", assigned_team="engineering")
        )
        assert "difficulty" in info
        assert info["difficulty"] in ("easy", "medium", "hard")

    def test_history_records_difficulty(self, env1):
        env1.step(Action(ticket_type="bug", priority="low", assigned_team="support"))
        assert "difficulty" in env1.history[0]

    def test_difficulty_bonus_applied_on_correct(self):
        reward = Reward(value=0.8, breakdown={"decision": 0.8}, is_correct=True)
        boosted = apply_difficulty_bonus(reward, "hard")
        assert boosted.value > reward.value
        assert "difficulty_bonus" in boosted.breakdown

    def test_difficulty_bonus_not_applied_on_wrong(self):
        reward = Reward(value=0.0, breakdown={"decision": 0.0}, is_correct=False)
        same = apply_difficulty_bonus(reward, "hard")
        assert same.value == reward.value

    def test_difficulty_bonus_easy_is_zero(self):
        reward = Reward(value=0.8, breakdown={}, is_correct=True)
        same = apply_difficulty_bonus(reward, "easy")
        assert same.value == reward.value


# ── Improvement 3: Customer reaction ─────────────────────────────────────────

class TestCustomerReaction:
    def test_reaction_present_in_task2_info(self, env2):
        _, _, _, info = env2.step(Action(
            reply_body="We sincerely apologize for the issue. Our team is looking into it now.",
            reply_tone="apologetic",
        ))
        assert "customer_reaction" in info
        assert info["customer_reaction"] in ("satisfied", "neutral", "frustrated")

    def test_reaction_absent_in_task1(self, env1):
        _, _, _, info = env1.step(
            Action(ticket_type="bug", priority="high", assigned_team="engineering")
        )
        assert "customer_reaction" not in info

    def test_reaction_absent_in_task4(self, env4):
        _, _, _, info = env4.step(Action(escalation_decision="auto_resolve"))
        assert "customer_reaction" not in info

    def test_good_reply_satisfied(self, env2):
        _, reward, _, info = env2.step(Action(
            reply_body="We sincerely apologize for the inconvenience. Our engineering team is investigating the export crash and we will update you within 2 hours.",
            reply_tone="apologetic",
        ))
        if reward.value >= 0.7:
            assert info["customer_reaction"] == "satisfied"

    def test_empty_reply_frustrated(self, env2):
        _, reward, _, info = env2.step(Action(reply_body=None, reply_tone=None))
        assert info["customer_reaction"] == "frustrated"


# ── Grader: Task 1 ────────────────────────────────────────────────────────────

class TestGraderTask1:
    GT = {"ticket_type": "bug", "priority": "high", "assigned_team": "engineering"}

    def test_perfect(self):
        r = grade_task_1(Action(ticket_type="bug", priority="high", assigned_team="engineering"), self.GT)
        assert r.value == pytest.approx(1.0) and r.is_correct

    def test_wrong_type(self):
        r = grade_task_1(Action(ticket_type="billing", priority="high", assigned_team="engineering"), self.GT)
        assert r.value < 1.0 and not r.is_correct

    def test_related_partial(self):
        gt = {"ticket_type": "churn_signal", "priority": "critical", "assigned_team": "customer_success"}
        r = grade_task_1(Action(ticket_type="general_inquiry", priority="critical", assigned_team="customer_success"), gt)
        assert 0 < r.value < 1.0

    def test_none_penalized(self):
        assert grade_task_1(Action(), self.GT).value < 0

    def test_alt_team_accepted(self):
        r = grade_task_1(Action(ticket_type="bug", priority="high", assigned_team="support"), self.GT)
        assert r.is_correct

    def test_breakdown_keys(self):
        r = grade_task_1(Action(ticket_type="bug", priority="high", assigned_team="engineering"), self.GT)
        assert all(k in r.breakdown for k in ["type", "priority", "team"])

    def test_reward_in_range(self):
        for tt in ["bug", "billing", "feature_request", "churn_signal", "general_inquiry"]:
            r = grade_task_1(Action(ticket_type=tt, priority="high", assigned_team="engineering"), self.GT)
            assert -1.0 <= r.value <= 1.0


# ── Grader: Task 2 ────────────────────────────────────────────────────────────

class TestGraderTask2:
    CTX = {
        "expected_tone": "apologetic",
        "required_keywords": ["apologize", "export", "fix"],
        "forbidden_phrases": ["not our fault"],
    }

    def test_good_reply(self):
        r = grade_task_2(Action(
            reply_body="We sincerely apologize for the export issue. Our team is working on a fix.",
            reply_tone="apologetic"), self.CTX)
        assert r.value >= 0.6

    def test_empty_penalized(self):
        assert grade_task_2(Action(reply_body=None, reply_tone="formal"), self.CTX).value < 0

    def test_wrong_tone_partial(self):
        r = grade_task_2(Action(
            reply_body="We apologize for the export bug. Fix coming soon.",
            reply_tone="formal"), self.CTX)
        assert 0 < r.value < 1.0

    def test_forbidden_phrase_lowers_score(self):
        clean = grade_task_2(Action(reply_body="We apologize for the export fix.", reply_tone="apologetic"), self.CTX)
        dirty = grade_task_2(Action(reply_body="This is not our fault. We apologize.", reply_tone="apologetic"), self.CTX)
        assert clean.value > dirty.value

    def test_keyword_coverage_proportional(self):
        full = grade_task_2(Action(
            reply_body="We apologize for the export problem. A fix is coming.", reply_tone="apologetic"), self.CTX)
        partial = grade_task_2(Action(
            reply_body="We apologize for the inconvenience.", reply_tone="apologetic"), self.CTX)
        assert full.breakdown["keywords"] > partial.breakdown["keywords"]


# ── Grader: Task 3 ────────────────────────────────────────────────────────────

class TestGraderTask3:
    GT_HIGH = {"churn_risk_score": 0.82, "risk_tier": "high"}
    GT_LOW  = {"churn_risk_score": 0.08, "risk_tier": "low"}

    def test_correct_high(self):
        r = grade_task_3(Action(churn_risk_score=0.85, retention_action="schedule_call"), self.GT_HIGH)
        assert r.value >= 0.7 and r.is_correct

    def test_correct_low(self):
        r = grade_task_3(Action(churn_risk_score=0.10, retention_action="no_action"), self.GT_LOW)
        assert r.value >= 0.7 and r.is_correct

    def test_wrong_tier_action(self):
        r = grade_task_3(Action(churn_risk_score=0.80, retention_action="no_action"), self.GT_HIGH)
        assert r.breakdown["retention_action"] < 0.3

    def test_none_penalized(self):
        r = grade_task_3(Action(retention_action="schedule_call"), self.GT_HIGH)
        assert r.breakdown["score_accuracy"] < 0

    def test_reward_in_range(self):
        for score, action in [(None, None), (0.5, "schedule_call"), (0.0, "no_action")]:
            r = grade_task_3(Action(churn_risk_score=score, retention_action=action), self.GT_HIGH)
            assert -1.0 <= r.value <= 1.0


# ── Grader: Task 4 (Improvement 1) ───────────────────────────────────────────

class TestGraderTask4:
    GT_ESC  = {"escalation_decision": "escalate_to_human",  "priority": "critical", "ticket_type": "bug"}
    GT_AUTO = {"escalation_decision": "auto_resolve",        "priority": "low",      "ticket_type": "general_inquiry"}
    GT_INFO = {"escalation_decision": "request_more_info",   "priority": "medium",   "ticket_type": "billing"}

    def test_correct_escalate(self):
        r = grade_task_4(Action(escalation_decision="escalate_to_human"), self.GT_ESC)
        assert r.is_correct and r.value >= 0.9

    def test_correct_auto_resolve(self):
        r = grade_task_4(Action(escalation_decision="auto_resolve"), self.GT_AUTO)
        assert r.is_correct and r.value >= 0.9

    def test_correct_request_info(self):
        r = grade_task_4(Action(escalation_decision="request_more_info"), self.GT_INFO)
        assert r.is_correct and r.value >= 0.9

    def test_wrong_gets_partial(self):
        r = grade_task_4(Action(escalation_decision="request_more_info"), self.GT_ESC)
        assert not r.is_correct and 0 < r.value < 1.0

    def test_auto_vs_escalate_zero(self):
        r = grade_task_4(Action(escalation_decision="auto_resolve"), self.GT_ESC)
        assert r.breakdown["decision"] == 0.0

    def test_none_penalized(self):
        r = grade_task_4(Action(), self.GT_ESC)
        assert r.value < 0

    def test_hard_difficulty_bonus(self):
        r_easy = grade_task_4(Action(escalation_decision="escalate_to_human"), self.GT_ESC, difficulty="easy")
        r_hard = grade_task_4(Action(escalation_decision="escalate_to_human"), self.GT_ESC, difficulty="hard")
        assert r_hard.breakdown.get("decision", 0) > r_easy.breakdown.get("decision", 0)

    def test_reward_always_in_range(self):
        for dec in ["auto_resolve", "escalate_to_human", "request_more_info", None]:
            r = grade_task_4(Action(escalation_decision=dec), self.GT_ESC)
            assert -1.0 <= r.value <= 1.0

    def test_feedback_present(self):
        r = grade_task_4(Action(escalation_decision="auto_resolve"), self.GT_ESC)
        assert len(r.feedback) > 0

    def test_task4_full_episode(self, env4):
        steps = 0
        while not env4.done and steps < 10:
            _, r, _, info = env4.step(Action(escalation_decision="escalate_to_human"))
            assert -1.0 <= r.value <= 1.0
            assert "difficulty" in info
            steps += 1
        assert env4.step_number > 0
