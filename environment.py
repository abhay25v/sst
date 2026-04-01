"""
Trust and Safety Decision Engine - OpenEnv-compatible RL Environment
"""
import uuid
from typing import Optional, Dict, Any
from models import Observation, Action, StepResult, ResetResult, StepType, EpisodeConfig
from dataset import Dataset, Task
from reward import RewardCalculator, DeterministicGrader, EpisodeAnalyzer


class TrustAndSafetyEnv:
    """
    OpenEnv-compatible reinforcement learning environment for content moderation.
    
    The environment simulates a real-world content moderation workflow with three stages:
    1. ANALYZE: Agent classifies toxicity and intent
    2. CONTEXT: Agent optionally checks user history and flag history
    3. DECISION: Agent makes final moderation decision
    """
    
    def __init__(self):
        """Initialize the environment."""
        self.episode_id: Optional[str] = None
        self.step_number: int = 0
        self.current_task: Optional[Task] = None
        self.current_step_type: StepType = StepType.ANALYZE
        self.actions_taken: list = []
        self.rewards_received: list = []
        self.step_types: list = []
        self.last_action_error: bool = False
        self.episode_config: Optional[EpisodeConfig] = None
    
    def reset(self, config: Optional[EpisodeConfig] = None) -> ResetResult:
        """
        Reset the environment and start a new episode.
        
        Args:
            config: Optional episode configuration (difficulty, task_id, seed)
        
        Returns:
            ResetResult with initial observation and metadata
        """
        self.episode_id = str(uuid.uuid4())
        self.step_number = 0
        self.current_step_type = StepType.ANALYZE
        self.actions_taken = []
        self.rewards_received = []
        self.step_types = []
        self.last_action_error = False
        self.episode_config = config or EpisodeConfig()
        
        # Load task from dataset
        self.current_task = Dataset.get_task(
            self.episode_config.difficulty,
            self.episode_config.task_id
        )
        
        # Create initial observation
        observation = Observation(
            content=self.current_task.content,
            user_history=self.current_task.user_history,
            previous_flags=self.current_task.previous_flags,
            step_type=self.current_step_type,
            last_action_error=False,
            episode_id=self.episode_id,
            step_number=self.step_number,
            task_difficulty=self.current_task.difficulty,
        )
        
        info = {
            "episode_id": self.episode_id,
            "difficulty": self.current_task.difficulty,
            "task_id": self.current_task.task_id,
        }
        
        return ResetResult(observation=observation, info=info)
    
    def step(self, action: Action) -> StepResult:
        """
        Take a step in the environment.
        
        Args:
            action: Agent's action
        
        Returns:
            StepResult with observation, reward, done flag, and info
        """
        if self.episode_id is None:
            raise RuntimeError("Must call reset() before step()")
        
        self.step_number += 1
        action_str = action.action
        self.actions_taken.append(action_str)
        self.step_types.append(self.current_step_type)
        
        # Validate action format
        is_valid = self._validate_action(action_str)
        self.last_action_error = not is_valid
        
        # Calculate reward
        reward = RewardCalculator.calculate_step_reward(
            step_type=self.current_step_type,
            action=action_str,
            ground_truth_toxicity=self.current_task.ground_truth_toxicity,
            ground_truth_intent=self.current_task.ground_truth_intent,
            ground_truth_decision=self.current_task.ground_truth_decision,
            user_history_length=len(self.current_task.user_history),
            previous_flags_length=len(self.current_task.previous_flags),
            is_valid_action=is_valid,
        )
        
        self.rewards_received.append(reward)
        
        # Determine next step or finish
        done = False
        next_step_type = self._get_next_step_type(action_str)
        
        if next_step_type is None:
            # Episode is complete
            done = True
        else:
            self.current_step_type = next_step_type
        
        # Create observation for next step or final observation
        observation = Observation(
            content=self.current_task.content,
            user_history=self.current_task.user_history if not done else [],
            previous_flags=self.current_task.previous_flags if not done else [],
            step_type=next_step_type or StepType.DECISION,
            last_action_error=self.last_action_error,
            episode_id=self.episode_id,
            step_number=self.step_number,
            task_difficulty=self.current_task.difficulty,
        )
        
        info = {
            "action_valid": is_valid,
            "episode_id": self.episode_id,
            "step": self.step_number,
            "step_type": str(self.current_step_type),
        }
        
        if done:
            info["episode_analysis"] = EpisodeAnalyzer.analyze(
                actions=self.actions_taken,
                step_types=self.step_types,
                rewards=self.rewards_received,
                ground_truth_decision=self.current_task.ground_truth_decision,
            )
            info["grade"] = DeterministicGrader.grade_episode(
                actions=self.actions_taken,
                step_types=self.step_types,
                ground_truth_toxicity=self.current_task.ground_truth_toxicity,
                ground_truth_intent=self.current_task.ground_truth_intent,
                ground_truth_decision=self.current_task.ground_truth_decision,
                user_history_length=len(self.current_task.user_history),
                previous_flags_length=len(self.current_task.previous_flags),
            )
        
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )
    
    def state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.
        
        Returns:
            Dictionary with current environment state
        """
        return {
            "episode_id": self.episode_id,
            "step_number": self.step_number,
            "current_step_type": str(self.current_step_type),
            "actions_taken": self.actions_taken,
            "rewards_received": self.rewards_received,
            "last_action_error": self.last_action_error,
            "task_id": self.current_task.task_id if self.current_task else None,
            "task_difficulty": self.current_task.difficulty if self.current_task else None,
        }
    
    def _validate_action(self, action: str) -> bool:
        """Validate action format."""
        action = action.strip()
        
        # ANALYZE: toxicity=X, intent=Y
        if action.startswith("ANALYZE:"):
            content = action.replace("ANALYZE:", "").strip()
            if not content:
                return False
            parts = content.split(",")
            keys = set()
            for part in parts:
                if "=" not in part:
                    return False
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key or not value:
                    return False
                keys.add(key)
            return len(keys) >= 1  # At least toxicity or intent
        
        # CHECK_HISTORY
        if action == "CHECK_HISTORY":
            return True
        
        # DECIDE: X or ESCALATE
        if action.startswith("DECIDE:"):
            decision = action.replace("DECIDE:", "").strip()
            return decision in ["ALLOW", "DELETE", "REDUCE_VISIBILITY", "ESCALATE"]
        
        if action == "ESCALATE":
            return True
        
        return False
    
    def _get_next_step_type(self, action: str) -> Optional[StepType]:
        """Determine the next step type based on current action."""
        action = action.strip()
        
        if self.current_step_type == StepType.ANALYZE:
            # After analysis, go to context
            if action.startswith("ANALYZE:"):
                return StepType.CONTEXT
            return None
        
        elif self.current_step_type == StepType.CONTEXT:
            # After checking context, go to decision
            # Or agent can skip and go to decision
            if action == "CHECK_HISTORY":
                return StepType.DECISION
            elif action.startswith("ANALYZE:"):
                # Agent can re-analyze
                return StepType.CONTEXT
            else:
                return None
        
        elif self.current_step_type == StepType.DECISION:
            # After decision, episode ends
            if action.startswith("DECIDE:") or action == "ESCALATE":
                return None
            return None
        
        return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            "name": "Trust and Safety Decision Engine",
            "version": "1.0.0",
            "action_space": {
                "type": "string",
                "examples": [
                    "ANALYZE: toxicity=high, intent=malicious",
                    "CHECK_HISTORY",
                    "DECIDE: DELETE",
                    "ESCALATE",
                ]
            },
            "observation_space": {
                "type": "Observation",
                "fields": ["content", "user_history", "previous_flags", "step_type", "last_action_error"],
            },
            "reward_range": [-1.0, 0.7],
            "episode_steps": 3,  # Up to 3 steps max
            "difficulties": ["easy", "medium", "hard"],
        }
