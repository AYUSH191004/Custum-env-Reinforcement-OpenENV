"""
Test suite for the CustomerSupportEnv.
Run with:  python -m pytest tests/ -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import CustomerSupportEnv, Observation, Action, Reward
from environment.graders import grade_task_1, grade_task_2, grade_task_3


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def env1():
    e = CustomerSupportEnv("task_1_ticket_classification")
    e.reset()
    return e

@pytest.fixture
def env2():
    e = CustomerSupportEnv("task_2_response_drafting")
    e.reset()
    return e

@pytest.fixture
def env3():
    e = CustomerSupportEnv("task_3_churn_detection")
    e.reset()
    return e


# ===========================================================================
# Core env interface
# ===========================================================================

class TestEnvInterface:
    def test_reset_returns_observation(self, env1):
        obs = env1.reset()
        assert isinstance(obs, Observation)
        assert obs.task_id == "task_1_ticket_classification"
        assert obs.step_number == 0
        assert obs.ticket_id.startswith("T")

    def test_step_returns_correct_tuple(self, env1):
        action = Action(ticket_type="bug", priority="high", assigned_team="engineering")
        obs, reward, done, info = env1.step(action)
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "total_reward" in info
        assert "steps" in info

    def test_state_has_required_fields(self, env1):
        s = env1.state()
        for field in ["task_id", "step_number", "done", "total_reward",
                      "correct_count", "avg_reward", "history"]:
            assert field in s, f"Missing field: {field}"

    def test_step_after_done_raises(self, env1):
        # Force episode to end
        env1.step_number = env1.max_steps - 1
        env1.done = True
        with pytest.raises(RuntimeError, match="Episode is done"):
            env1.step(Action())

    def test_step_increments_step_number(self, env1):
        before = env1.step_number
        env1.step(Action(ticket_type="bug", priority="low", assigned_team="support"))
        assert env1.step_number == before + 1

    def test_history_grows_per_step(self, env1):
        env1.step(Action(ticket_type="bug", priority="low", assigned_team="support"))
        env1.step(Action(ticket_type="billing", priority="medium", assigned_team="billing"))
        assert len(env1.history) == 2

    def test_invalid_task_id_raises(self):
        with pytest.raises(ValueError):
            CustomerSupportEnv("invalid_task")

    def test_all_three_tasks_construct(self):
        for tid in ["task_1_ticket_classification",
                    "task_2_response_drafting",
                    "task_3_churn_detection"]:
            env = CustomerSupportEnv(tid)
            obs = env.reset()
            assert obs.task_id == tid


# ===========================================================================
# Observation model
# ===========================================================================

class TestObservation:
    def test_observation_fields_present(self, env1):
        obs = env1.reset()
        assert obs.subject
        assert obs.body
        assert obs.customer_name
        assert obs.plan in ("free", "starter", "pro", "enterprise")
        assert obs.account_age_days >= 0
        assert obs.mrr >= 0.0
        assert obs.previous_sentiment in ("positive", "neutral", "negative")

    def test_ground_truth_not_in_observation(self, env3):
        """Agent must NOT see the ground truth churn score."""
        obs = env3.reset()
        assert "churn_risk_score" not in obs.context
        assert "ground_truth" not in obs.model_dump()


# ===========================================================================
# Grader: Task 1
# ===========================================================================

class TestGraderTask1:
    GT = {"ticket_type": "bug", "priority": "high", "assigned_team": "engineering"}

    def test_perfect_score(self):
        action = Action(ticket_type="bug", priority="high", assigned_team="engineering")
        r = grade_task_1(action, self.GT)
        assert r.value == pytest.approx(1.0)
        assert r.is_correct is True

    def test_wrong_type_zero_priority_correct(self):
        action = Action(ticket_type="billing", priority="high", assigned_team="engineering")
        r = grade_task_1(action, self.GT)
        assert r.value < 1.0
        assert r.is_correct is False

    def test_related_type_gets_partial(self):
        """churn_signal → general_inquiry should give partial credit."""
        gt = {"ticket_type": "churn_signal", "priority": "critical", "assigned_team": "customer_success"}
        action = Action(ticket_type="general_inquiry", priority="critical", assigned_team="customer_success")
        r = grade_task_1(action, gt)
        assert 0.0 < r.value < 1.0

    def test_none_fields_penalized(self):
        action = Action()   # all None
        r = grade_task_1(action, self.GT)
        assert r.value < 0.0

    def test_reward_in_range(self):
        for ticket_type in ["bug", "billing", "feature_request", "churn_signal", "general_inquiry"]:
            action = Action(ticket_type=ticket_type, priority="high", assigned_team="engineering")
            r = grade_task_1(action, self.GT)
            assert -1.0 <= r.value <= 1.0

    def test_breakdown_keys_present(self):
        action = Action(ticket_type="bug", priority="high", assigned_team="engineering")
        r = grade_task_1(action, self.GT)
        assert "type" in r.breakdown
        assert "priority" in r.breakdown
        assert "team" in r.breakdown

    def test_acceptable_alternative_team(self):
        """'support' is also acceptable for bug tickets."""
        action = Action(ticket_type="bug", priority="high", assigned_team="support")
        r = grade_task_1(action, self.GT)
        assert r.is_correct is True


# ===========================================================================
# Grader: Task 2
# ===========================================================================

class TestGraderTask2:
    CTX = {
        "expected_tone": "apologetic",
        "required_keywords": ["apologize", "export", "fix"],
        "forbidden_phrases": ["not our fault"],
    }

    def test_good_reply_high_score(self):
        action = Action(
            reply_body="We sincerely apologize for the inconvenience with the export feature. Our team is investigating the fix and will update you shortly.",
            reply_tone="apologetic",
        )
        r = grade_task_2(action, self.CTX)
        assert r.value >= 0.6

    def test_empty_reply_penalized(self):
        action = Action(reply_body=None, reply_tone="formal")
        r = grade_task_2(action, self.CTX)
        assert r.value < 0.0

    def test_wrong_tone_partial_credit(self):
        action = Action(
            reply_body="We apologize for the export bug. Our team is fixing it urgently.",
            reply_tone="formal",   # expected: apologetic — adjacent tone
        )
        r = grade_task_2(action, self.CTX)
        assert 0.0 < r.value < 1.0

    def test_forbidden_phrase_penalty(self):
        action_clean = Action(
            reply_body="We apologize for the export issue and will fix it.",
            reply_tone="apologetic",
        )
        action_dirty = Action(
            reply_body="We apologize, but this is not our fault. The export bug will be fixed.",
            reply_tone="apologetic",
        )
        r_clean = grade_task_2(action_clean, self.CTX)
        r_dirty = grade_task_2(action_dirty, self.CTX)
        assert r_clean.value > r_dirty.value

    def test_too_short_partial_credit(self):
        action = Action(reply_body="Sorry.", reply_tone="apologetic")
        r = grade_task_2(action, self.CTX)
        assert r.value < 0.5

    def test_keyword_coverage_proportional(self):
        # All 3 keywords
        action_full = Action(
            reply_body="We apologize for the export problem. Our team is working on a fix.",
            reply_tone="apologetic",
        )
        # Only 1 keyword
        action_partial = Action(
            reply_body="We apologize for the inconvenience you experienced.",
            reply_tone="apologetic",
        )
        r_full    = grade_task_2(action_full, self.CTX)
        r_partial = grade_task_2(action_partial, self.CTX)
        assert r_full.breakdown["keywords"] > r_partial.breakdown["keywords"]


# ===========================================================================
# Grader: Task 3
# ===========================================================================

class TestGraderTask3:
    GT_HIGH = {"churn_risk_score": 0.82, "risk_tier": "high"}
    GT_LOW  = {"churn_risk_score": 0.08, "risk_tier": "low"}
    GT_MED  = {"churn_risk_score": 0.50, "risk_tier": "medium"}

    def test_correct_high_risk(self):
        action = Action(churn_risk_score=0.85, retention_action="schedule_call")
        r = grade_task_3(action, self.GT_HIGH)
        assert r.value >= 0.7
        assert r.is_correct is True

    def test_correct_low_risk(self):
        action = Action(churn_risk_score=0.10, retention_action="no_action")
        r = grade_task_3(action, self.GT_LOW)
        assert r.value >= 0.7
        assert r.is_correct is True

    def test_wrong_tier_action_penalized(self):
        """Sending 'no_action' for a high-risk customer should score poorly."""
        action = Action(churn_risk_score=0.80, retention_action="no_action")
        r = grade_task_3(action, self.GT_HIGH)
        assert r.breakdown["retention_action"] < 0.3

    def test_none_score_penalized(self):
        action = Action(retention_action="schedule_call")
        r = grade_task_3(action, self.GT_HIGH)
        assert r.breakdown["score_accuracy"] < 0.0

    def test_close_score_partial_credit(self):
        action = Action(churn_risk_score=0.65, retention_action="schedule_call")
        r = grade_task_3(action, self.GT_HIGH)   # gt=0.82, distance=0.17 → partial
        assert r.breakdown["score_accuracy"] >= 0.2

    def test_all_valid_actions_acceptable_for_high(self):
        valid = ["schedule_call", "offer_discount", "flag_account_manager"]
        for act in valid:
            action = Action(churn_risk_score=0.80, retention_action=act)
            r = grade_task_3(action, self.GT_HIGH)
            assert r.breakdown["retention_action"] == pytest.approx(0.45)

    def test_reward_always_in_range(self):
        test_cases = [
            (None, None, self.GT_HIGH),
            (0.5, "schedule_call", self.GT_HIGH),
            (0.0, "no_action", self.GT_HIGH),
            (1.0, "offer_discount", self.GT_LOW),
        ]
        for score, action, gt in test_cases:
            a = Action(churn_risk_score=score, retention_action=action)
            r = grade_task_3(a, gt)
            assert -1.0 <= r.value <= 1.0

    def test_risk_tier_logic_bonus(self):
        """Agent that correctly infers the risk tier gets bonus points."""
        action = Action(churn_risk_score=0.78, retention_action="schedule_call")
        r = grade_task_3(action, self.GT_HIGH)   # 0.78 ≥ 0.65 → high tier ✓
        assert r.breakdown["risk_tier_logic"] == pytest.approx(0.10)


# ===========================================================================
# Full episode integration test
# ===========================================================================

class TestFullEpisode:
    def test_task_1_episode_completes(self):
        env = CustomerSupportEnv("task_1_ticket_classification")
        env.reset()
        steps = 0
        while not env.done and steps < 20:
            obs, reward, done, info = env.step(
                Action(ticket_type="bug", priority="medium", assigned_team="engineering")
            )
            steps += 1
        s = env.state()
        assert s["step_number"] > 0
        assert s["total_reward"] is not None

    def test_reward_accumulates_correctly(self):
        env = CustomerSupportEnv("task_1_ticket_classification")
        env.reset()
        cumulative = 0.0
        for _ in range(3):
            if env.done:
                break
            _, reward, _, _ = env.step(
                Action(ticket_type="bug", priority="high", assigned_team="engineering")
            )
            cumulative += reward.value
        assert abs(env.total_reward - cumulative) < 1e-6

    def test_task_2_episode(self):
        env = CustomerSupportEnv("task_2_response_drafting")
        env.reset()
        _, reward, _, _ = env.step(Action(
            reply_body="We sincerely apologize and our team is investigating the issue urgently.",
            reply_tone="apologetic",
        ))
        assert -1.0 <= reward.value <= 1.0
        assert reward.feedback

    def test_task_3_episode(self):
        env = CustomerSupportEnv("task_3_churn_detection")
        env.reset()
        _, reward, _, _ = env.step(Action(
            churn_risk_score=0.75,
            retention_action="schedule_call",
        ))
        assert -1.0 <= reward.value <= 1.0
