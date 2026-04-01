"""
Pydantic models for the Trust and Safety Decision Engine environment.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class StepType(str, Enum):
    """Enum for the different step types in an episode."""
    ANALYZE = "analyze"
    CONTEXT = "context"
    DECISION = "decision"


class Observation(BaseModel):
    """
    Represents the observation returned by the environment at each step.
    
    Attributes:
        content: The content to be moderated (string).
        user_history: List of previous content from the user.
        previous_flags: List of previous moderation actions on user's content.
        step_type: Current stage of the episode (analyze, context, or decision).
        last_action_error: Whether the previous action was invalid.
        episode_id: Unique identifier for the current episode.
        step_number: Current step within the episode.
        task_difficulty: Difficulty level of the current task (easy, medium, hard).
    """
    content: str = Field(..., description="Content to be moderated")
    user_history: List[str] = Field(default_factory=list, description="User's previous content")
    previous_flags: List[str] = Field(default_factory=list, description="Previous moderation actions")
    step_type: StepType = Field(..., description="Current stage of the episode")
    last_action_error: bool = Field(default=False, description="Whether previous action was invalid")
    episode_id: str = Field(..., description="Unique episode identifier")
    step_number: int = Field(..., description="Current step within the episode")
    task_difficulty: str = Field(..., description="Difficulty level: easy, medium, hard")


class Action(BaseModel):
    """
    Represents an action the agent can take.
    
    Examples:
        - "ANALYZE: toxicity=high, intent=harassment"
        - "CHECK_HISTORY"
        - "DECIDE: DELETE"
        - "ESCALATE"
        - "DECIDE: ALLOW"
        - "DECIDE: REDUCE_VISIBILITY"
    """
    action: str = Field(..., description="Action string in the specified format")

    class Config:
        json_schema_extra = {
            "examples": [
                {"action": "ANALYZE: toxicity=high, intent=harassment"},
                {"action": "CHECK_HISTORY"},
                {"action": "DECIDE: DELETE"},
                {"action": "ESCALATE"},
            ]
        }


class StepResult(BaseModel):
    """
    Represents the result of a step() call.
    
    Attributes:
        observation: The observation returned by the environment.
        reward: Scalar reward value.
        done: Whether the episode has ended.
        info: Additional metadata about the step.
    """
    observation: Observation = Field(..., description="Next observation")
    reward: float = Field(..., description="Reward for this step")
    done: bool = Field(..., description="Whether the episode is complete")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ResetResult(BaseModel):
    """
    Represents the result of a reset() call.
    
    Attributes:
        observation: Initial observation for the new episode.
        info: Additional metadata about the episode.
    """
    observation: Observation = Field(..., description="Initial observation")
    info: Dict[str, Any] = Field(default_factory=dict, description="Episode metadata")


class EpisodeConfig(BaseModel):
    """Configuration for an episode."""
    difficulty: str = Field(default="medium", description="Difficulty level: easy, medium, hard")
    task_id: Optional[str] = Field(default=None, description="Optional specific task ID")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
