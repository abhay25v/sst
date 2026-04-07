"""
Reward function and grading logic for the Trust and Safety Decision Engine.
"""
from typing import Dict, List, Tuple, Any
import re


class RewardCalculator:
    """
    Computes rewards based on agent actions and ground truth.
    
    Reward values:
    - Correct analysis: +0.3
    - Correct context usage: +0.3
    - Correct final decision: +0.7
    - Wrong delete: -0.7
    - Missed harmful: -1.0
    - Unnecessary escalate: -0.2
    """
    
    # Reward constants
    CORRECT_ANALYSIS = 0.3
    CORRECT_CONTEXT = 0.3
    CORRECT_DECISION = 0.7
    WRONG_DELETE = -0.7
    MISSED_HARMFUL = -1.0
    UNNECESSARY_ESCALATE = -0.2
    INVALID_ACTION = -0.1
    
    @staticmethod
    def calculate_step_reward(
        step_type: str,
        action: str,
        ground_truth_toxicity: str,
        ground_truth_intent: str,
        ground_truth_decision: str,
        user_history_length: int,
        previous_flags_length: int,
        is_valid_action: bool
    ) -> float:
        """
        Calculate reward for a single step.
        
        Args:
            step_type: Current step type (analyze, context, decision)
            action: Agent's action
            ground_truth_toxicity: Ground truth toxicity level
            ground_truth_intent: Ground truth intent
            ground_truth_decision: Ground truth final decision
            user_history_length: Length of user history
            previous_flags_length: Length of previous flags
            is_valid_action: Whether action format is valid
        
        Returns:
            Float reward value
        """
        if not is_valid_action:
            return RewardCalculator.INVALID_ACTION
        
        reward = 0.0
        
        if step_type == "analyze":
            reward = RewardCalculator._evaluate_analyze(
                action, ground_truth_toxicity, ground_truth_intent
            )
        elif step_type == "context":
            reward = RewardCalculator._evaluate_context(
                action, user_history_length, previous_flags_length
            )
        elif step_type == "decision":
            reward = RewardCalculator._evaluate_decision(
                action, ground_truth_decision, ground_truth_toxicity
            )
        
        return reward
    
    @staticmethod
    def _evaluate_analyze(action: str, ground_truth_toxicity: str, ground_truth_intent: str) -> float:
        """Evaluate ANALYZE step."""
        try:
            # Parse: "ANALYZE: toxicity=high, intent=malicious"
            if not action.startswith("ANALYZE:"):
                return RewardCalculator.INVALID_ACTION
            
            content = action.replace("ANALYZE:", "").strip()
            parts = [p.strip() for p in content.split(",")]
            
            parsed = {}
            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    parsed[key.strip()] = value.strip().lower()
            
            reward = 0.0
            
            # Check toxicity prediction
            if "toxicity" in parsed:
                predicted_toxicity = parsed["toxicity"]
                if predicted_toxicity == ground_truth_toxicity.lower():
                    reward += RewardCalculator.CORRECT_ANALYSIS * 0.5
                else:
                    reward -= 0.15
            
            # Check intent prediction
            if "intent" in parsed:
                predicted_intent = parsed["intent"]
                if predicted_intent == ground_truth_intent.lower():
                    reward += RewardCalculator.CORRECT_ANALYSIS * 0.5
                else:
                    reward -= 0.15
            
            return reward
        except Exception:
            return RewardCalculator.INVALID_ACTION
    
    @staticmethod
    def _evaluate_context(
        action: str,
        user_history_length: int,
        previous_flags_length: int
    ) -> float:
        """Evaluate CONTEXT step (CHECK_HISTORY)."""
        if action.strip() == "CHECK_HISTORY":
            # Reward for checking context when potentially needed
            context_value = (user_history_length + previous_flags_length) / 10.0
            return min(RewardCalculator.CORRECT_CONTEXT, context_value)
        
        return RewardCalculator.INVALID_ACTION
    
    @staticmethod
    def _evaluate_decision(
        action: str,
        ground_truth_decision: str,
        ground_truth_toxicity: str
    ) -> float:
        """Evaluate DECISION step."""
        try:
            if action.startswith("DECIDE:"):
                decision = action.replace("DECIDE:", "").strip()
                
                if decision == ground_truth_decision:
                    return RewardCalculator.CORRECT_DECISION
                
                # Penalize inappropriate deletions
                if decision == "DELETE" and ground_truth_decision != "DELETE":
                    if ground_truth_toxicity == "low":
                        return RewardCalculator.WRONG_DELETE
                
                # Penalize missed harmful content
                if decision != "DELETE" and ground_truth_decision == "DELETE":
                    return RewardCalculator.MISSED_HARMFUL
                
                # Penalize unnecessary escalation
                if decision == "ESCALATE" and ground_truth_decision not in ["ESCALATE", "DELETE"]:
                    return RewardCalculator.UNNECESSARY_ESCALATE
                
                # Partial credit for reasonable decisions
                if (decision in ["REDUCE_VISIBILITY", "ESCALATE"] and 
                    ground_truth_decision in ["REDUCE_VISIBILITY", "ESCALATE"]):
                    return 0.5 * RewardCalculator.CORRECT_DECISION
                
                return -0.3
            
            elif action == "ESCALATE":
                if ground_truth_decision == "ESCALATE":
                    return RewardCalculator.CORRECT_DECISION
                if ground_truth_decision == "DELETE":
                    return 0.5 * RewardCalculator.CORRECT_DECISION
                return RewardCalculator.UNNECESSARY_ESCALATE
            
            return RewardCalculator.INVALID_ACTION
        except Exception:
            return RewardCalculator.INVALID_ACTION


class DeterministicGrader:
    """
    Deterministic grader for evaluating full episodes.
    Returns scores from 0.0 to 1.0.
    """
    
    @staticmethod
    def grade_episode(
        actions: List[str],
        step_types: List[str],
        ground_truth_toxicity: str,
        ground_truth_intent: str,
        ground_truth_decision: str,
        user_history_length: int,
        previous_flags_length: int
    ) -> float:
        """
        Grade a complete episode.
        
        Returns:
            Score strictly between 0 and 1 (not 0.0 or 1.0)
        """
        eps = 0.01  # 1% margin from boundaries
        
        if not actions or not step_types:
            return eps  # Return epsilon instead of exact 0.0
        
        analyze_score = 0.0
        context_score = 0.0
        decision_score = 0.0
        
        total_score = 0.0
        action_count = 0
        
        for action, step_type in zip(actions, step_types):
            if step_type == "analyze":
                analyze_score = max(analyze_score, 
                    DeterministicGrader._score_analyze_action(action, ground_truth_toxicity, ground_truth_intent))
            elif step_type == "context":
                context_score = max(context_score,
                    DeterministicGrader._score_context_action(action, user_history_length, previous_flags_length))
            elif step_type == "decision":
                decision_score = DeterministicGrader._score_decision_action(action, ground_truth_decision)
            
            action_count += 1
        
        # Weighted final score
        final_score = (analyze_score * 0.25 + context_score * 0.25 + decision_score * 0.5)
        
        # Normalize to (0, 1) - strictly between, not including endpoints
        # Validator requires scores strictly between 0 and 1, not 0.0 or 1.0
        # Use larger epsilon to ensure no exact boundaries
        eps = 0.01  # 1% margin from boundaries
        score = (final_score + 1.0) / 2.0  # Maps [-1, 1] to [0, 1]
        return max(eps, min(1.0 - eps, score))
    
    @staticmethod
    def _score_analyze_action(action: str, ground_truth_toxicity: str, ground_truth_intent: str) -> float:
        """Score ANALYZE action on -1.0 to 1.0 scale."""
        try:
            if not action.startswith("ANALYZE:"):
                return -1.0
            
            content = action.replace("ANALYZE:", "").strip()
            parts = [p.strip() for p in content.split(",")]
            
            parsed = {}
            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    parsed[key.strip()] = value.strip().lower()
            
            score = 0.0
            checks = 0
            
            if "toxicity" in parsed:
                checks += 1
                if parsed["toxicity"] == ground_truth_toxicity.lower():
                    score += 1.0
                else:
                    score -= 0.5
            
            if "intent" in parsed:
                checks += 1
                if parsed["intent"] == ground_truth_intent.lower():
                    score += 1.0
                else:
                    score -= 0.5
            
            if checks == 0:
                return -1.0
            
            return score / checks
        except Exception:
            return -1.0
    
    @staticmethod
    def _score_context_action(action: str, user_history_length: int, previous_flags_length: int) -> float:
        """Score CONTEXT action on -1.0 to 1.0 scale."""
        if action.strip() == "CHECK_HISTORY":
            context_needed = (user_history_length + previous_flags_length) > 0
            if context_needed:
                return 1.0
            else:
                return 0.5  # Neutral if no context to check
        return -1.0
    
    @staticmethod
    def _score_decision_action(action: str, ground_truth_decision: str) -> float:
        """Score DECISION action on -1.0 to 1.0 scale."""
        try:
            decision = None
            
            if action.startswith("DECIDE:"):
                decision = action.replace("DECIDE:", "").strip()
            elif action == "ESCALATE":
                decision = "ESCALATE"
            
            if not decision:
                return -1.0
            
            if decision == ground_truth_decision:
                return 1.0
            
            # Partial credit for reasonable alternatives
            similarity = DeterministicGrader._decision_similarity(decision, ground_truth_decision)
            return -1.0 + (2.0 * similarity)
        except Exception:
            return -1.0
    
    @staticmethod
    def _decision_similarity(predicted: str, ground_truth: str) -> float:
        """Calculate similarity between predictions on 0.0-1.0."""
        decision_hierarchy = {
            "ALLOW": 1,
            "REDUCE_VISIBILITY": 2,
            "ESCALATE": 3,
            "DELETE": 4,
        }
        
        if predicted == ground_truth:
            return 1.0
        
        if predicted not in decision_hierarchy or ground_truth not in decision_hierarchy:
            return 0.0
        
        distance = abs(decision_hierarchy[predicted] - decision_hierarchy[ground_truth])
        return max(0.0, 1.0 - (distance * 0.25))


class EpisodeAnalyzer:
    """Analyze episode performance."""
    
    @staticmethod
    def analyze(
        actions: List[str],
        step_types: List[str],
        rewards: List[float],
        ground_truth_decision: str
    ) -> Dict[str, Any]:
        """
        Analyze episode performance.
        
        Returns:
            Dictionary with analysis metrics
        """
        total_reward = sum(rewards)
        num_steps = len(actions)
        final_decision = None
        
        # Extract final decision
        for action in reversed(actions):
            if action.startswith("DECIDE:") or action == "ESCALATE":
                if action.startswith("DECIDE:"):
                    final_decision = action.replace("DECIDE:", "").strip()
                else:
                    final_decision = "ESCALATE"
                break
        
        correct = final_decision == ground_truth_decision
        
        return {
            "total_reward": total_reward,
            "num_steps": num_steps,
            "average_reward": total_reward / num_steps if num_steps > 0 else 0.0,
            "final_decision": final_decision,
            "ground_truth_decision": ground_truth_decision,
            "correct": correct,
            "accuracy": 1.0 if correct else 0.0,
        }
