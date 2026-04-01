"""
Test script for the Trust and Safety Decision Engine.
Runs basic sanity checks to verify installation and functionality.
"""
import sys
from typing import List, Tuple


def test_imports() -> Tuple[bool, str]:
    """Test that all required modules can be imported."""
    try:
        from models import Observation, Action, StepResult, EpisodeConfig
        from dataset import Dataset
        from environment import TrustAndSafetyEnv
        from reward import RewardCalculator, DeterministicGrader
        return True, "✓ All imports successful"
    except ImportError as e:
        return False, f"✗ Import failed: {e}"


def test_dataset() -> Tuple[bool, str]:
    """Test that dataset loads correctly."""
    try:
        from dataset import Dataset
        
        # Test getting tasks
        easy_tasks = Dataset.get_tasks_by_difficulty("easy")
        medium_tasks = Dataset.get_tasks_by_difficulty("medium")
        hard_tasks = Dataset.get_tasks_by_difficulty("hard")
        
        assert len(easy_tasks) == 4, f"Expected 4 easy tasks, got {len(easy_tasks)}"
        assert len(medium_tasks) == 4, f"Expected 4 medium tasks, got {len(medium_tasks)}"
        assert len(hard_tasks) == 4, f"Expected 4 hard tasks, got {len(hard_tasks)}"
        
        # Test getting specific task
        task = Dataset.get_task("easy")
        assert task.task_id.startswith("easy_"), f"Expected easy task, got {task.task_id}"
        
        return True, "✓ Dataset tests passed"
    except Exception as e:
        return False, f"✗ Dataset test failed: {e}"


def test_models() -> Tuple[bool, str]:
    """Test that Pydantic models work correctly."""
    try:
        from models import Observation, Action, StepResult, StepType
        
        # Create observation
        obs = Observation(
            content="Test content",
            user_history=["post1", "post2"],
            previous_flags=["flag1"],
            step_type=StepType.ANALYZE,
            last_action_error=False,
            episode_id="test_episode",
            step_number=1,
            task_difficulty="easy",
        )
        
        # Create action
        action = Action(action="ANALYZE: toxicity=high, intent=malicious")
        
        # Create step result
        result = StepResult(
            observation=obs,
            reward=0.3,
            done=False,
            info={},
        )
        
        return True, "✓ Model tests passed"
    except Exception as e:
        return False, f"✗ Model test failed: {e}"


def test_environment() -> Tuple[bool, str]:
    """Test basic environment functionality."""
    try:
        from environment import TrustAndSafetyEnv
        from models import Action, EpisodeConfig
        
        env = TrustAndSafetyEnv()
        
        # Test reset
        reset_result = env.reset(EpisodeConfig(difficulty="easy"))
        assert reset_result.observation is not None
        assert reset_result.info["episode_id"] is not None
        
        # Test step
        action = Action(action="ANALYZE: toxicity=high, intent=malicious")
        step_result = env.step(action)
        assert step_result.reward is not None
        assert isinstance(step_result.done, bool)
        
        # Test state
        state = env.state()
        assert state["episode_id"] is not None
        assert len(state["actions_taken"]) == 1
        
        return True, "✓ Environment tests passed"
    except Exception as e:
        return False, f"✗ Environment test failed: {e}"


def test_reward_function() -> Tuple[bool, str]:
    """Test reward calculation."""
    try:
        from reward import RewardCalculator, DeterministicGrader
        
        # Test correct analysis
        reward = RewardCalculator.calculate_step_reward(
            step_type="analyze",
            action="ANALYZE: toxicity=high, intent=malicious",
            ground_truth_toxicity="high",
            ground_truth_intent="malicious",
            ground_truth_decision="DELETE",
            user_history_length=2,
            previous_flags_length=1,
            is_valid_action=True,
        )
        assert reward > 0, f"Expected positive reward for correct analysis, got {reward}"
        
        # Test incorrect analysis
        reward = RewardCalculator.calculate_step_reward(
            step_type="analyze",
            action="ANALYZE: toxicity=low, intent=benign",
            ground_truth_toxicity="high",
            ground_truth_intent="malicious",
            ground_truth_decision="DELETE",
            user_history_length=2,
            previous_flags_length=1,
            is_valid_action=True,
        )
        assert reward < 0, f"Expected negative reward for incorrect analysis, got {reward}"
        
        # Test grader
        actions = ["ANALYZE: toxicity=high, intent=malicious", "CHECK_HISTORY", "DECIDE: DELETE"]
        step_types = ["analyze", "context", "decision"]
        grade = DeterministicGrader.grade_episode(
            actions=actions,
            step_types=step_types,
            ground_truth_toxicity="high",
            ground_truth_intent="malicious",
            ground_truth_decision="DELETE",
            user_history_length=2,
            previous_flags_length=1,
        )
        assert 0.0 <= grade <= 1.0, f"Expected grade in [0, 1], got {grade}"
        
        return True, "✓ Reward tests passed"
    except Exception as e:
        return False, f"✗ Reward test failed: {e}"


def test_full_episode() -> Tuple[bool, str]:
    """Test a full episode from reset to done."""
    try:
        from environment import TrustAndSafetyEnv
        from models import Action, EpisodeConfig
        
        env = TrustAndSafetyEnv()
        reset_result = env.reset(EpisodeConfig(difficulty="easy"))
        
        actions = [
            "ANALYZE: toxicity=high, intent=malicious",
            "CHECK_HISTORY",
            "DECIDE: DELETE",
        ]
        
        total_reward = 0
        for action_str in actions:
            result = env.step(Action(action=action_str))
            total_reward += result.reward
            
            if result.done:
                assert "episode_analysis" in result.info
                assert "grade" in result.info
                break
        
        return True, "✓ Full episode test passed"
    except Exception as e:
        return False, f"✗ Full episode test failed: {e}"


def test_action_validation() -> Tuple[bool, str]:
    """Test action validation."""
    try:
        from environment import TrustAndSafetyEnv
        
        env = TrustAndSafetyEnv()
        env.reset()
        
        # Test valid actions
        valid_actions = [
            "ANALYZE: toxicity=high, intent=malicious",
            "ANALYZE: toxicity=low",
            "CHECK_HISTORY",
            "DECIDE: DELETE",
            "DECIDE: ALLOW",
            "DECIDE: REDUCE_VISIBILITY",
            "ESCALATE",
        ]
        
        for action in valid_actions:
            is_valid = env._validate_action(action)
            assert is_valid, f"Expected action to be valid: {action}"
        
        # Test invalid actions
        invalid_actions = [
            "INVALID_ACTION",
            "ANALYZE: invalid_format",
            "DECIDE: INVALID",
            "",
        ]
        
        for action in invalid_actions:
            is_valid = env._validate_action(action)
            assert not is_valid, f"Expected action to be invalid: {action}"
        
        return True, "✓ Action validation tests passed"
    except Exception as e:
        return False, f"✗ Action validation test failed: {e}"


def main():
    """Run all tests."""
    print("=" * 80)
    print("Trust and Safety Decision Engine - Test Suite")
    print("=" * 80)
    print()
    
    tests: List[Tuple[callable, str]] = [
        (test_imports, "Imports"),
        (test_models, "Pydantic Models"),
        (test_dataset, "Dataset"),
        (test_environment, "Environment"),
        (test_reward_function, "Reward Function"),
        (test_action_validation, "Action Validation"),
        (test_full_episode, "Full Episode"),
    ]
    
    results = []
    for test_func, test_name in tests:
        print(f"Running {test_name}...", end=" ")
        try:
            passed, message = test_func()
            results.append((test_name, passed, message))
            print(message)
        except Exception as e:
            results.append((test_name, False, f"✗ Unexpected error: {e}"))
            print(f"✗ Unexpected error: {e}")
    
    print()
    print("=" * 80)
    print("Test Results")
    print("=" * 80)
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)
    
    for test_name, passed, message in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {test_name}: {message}")
    
    print()
    print(f"Passed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
