"""
Sample dataset for the Trust and Safety Decision Engine.
Includes tasks with varying difficulty levels and deterministic scenarios.
"""
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Task:
    """Represents a moderation task."""
    task_id: str
    difficulty: str
    content: str
    user_history: List[str]
    previous_flags: List[str]
    ground_truth_toxicity: str  # low, medium, high
    ground_truth_intent: str    # benign, suspicious, malicious
    ground_truth_decision: str  # ALLOW, REDUCE_VISIBILITY, DELETE, ESCALATE
    reasoning: str


class Dataset:
    """Embedded dataset for Trust and Safety environment."""
    
    EASY_TASKS: List[Task] = [
        Task(
            task_id="easy_001",
            difficulty="easy",
            content="I hope you die in a fire, you disgusting piece of trash",
            user_history=["Hello everyone", "Nice day today", "How are you all?"],
            previous_flags=[],
            ground_truth_toxicity="high",
            ground_truth_intent="malicious",
            ground_truth_decision="DELETE",
            reasoning="Clear death threat with aggressive language",
        ),
        Task(
            task_id="easy_002",
            difficulty="easy",
            content="You're all a bunch of idiots",
            user_history=["Just sharing my opinion"],
            previous_flags=["verbal_abuse"],
            ground_truth_toxicity="high",
            ground_truth_intent="abusive",
            ground_truth_decision="DELETE",
            reasoning="Explicit insults targeting group, repeat offender",
        ),
        Task(
            task_id="easy_003",
            difficulty="easy",
            content="I'm so happy today! Feeling blessed!",
            user_history=["Positive vibes only", "Love this community"],
            previous_flags=[],
            ground_truth_toxicity="low",
            ground_truth_intent="benign",
            ground_truth_decision="ALLOW",
            reasoning="Clearly positive sentiment, no policy violation",
        ),
        Task(
            task_id="easy_004",
            difficulty="easy",
            content="k*ll all people from [ethnic group]",
            user_history=[],
            previous_flags=[],
            ground_truth_toxicity="high",
            ground_truth_intent="malicious",
            ground_truth_decision="DELETE",
            reasoning="Genocidal speech, coded language, severe harm",
        ),
    ]

    MEDIUM_TASKS: List[Task] = [
        Task(
            task_id="medium_001",
            difficulty="medium",
            content="That study is complete BS, the methodology is flawed",
            user_history=["Academic researcher", "Published 5 papers", "Peer review comments often harsh"],
            previous_flags=[],
            ground_truth_toxicity="low",
            ground_truth_intent="benign",
            ground_truth_decision="ALLOW",
            reasoning="Professional criticism with strong language but context shows academic discussion",
        ),
        Task(
            task_id="medium_002",
            difficulty="medium",
            content="These people are ruining our country",
            user_history=["Posted same message 10 times yesterday", "Frequently uses inflammatory language"],
            previous_flags=["hate_speech_warning", "repeated_content"],
            ground_truth_toxicity="high",
            ground_truth_intent="malicious",
            ground_truth_decision="REDUCE_VISIBILITY",
            reasoning="Vague group targeting; user history shows pattern; reduce but don't delete for free speech",
        ),
        Task(
            task_id="medium_003",
            difficulty="medium",
            content="Let's take this conversation offline and discuss in private",
            user_history=["Newbie user", "Posted 3 times total"],
            previous_flags=[],
            ground_truth_toxicity="low",
            ground_truth_intent="benign",
            ground_truth_decision="ALLOW",
            reasoning="Suggests private discussion, no toxicity; context shows inexperienced user",
        ),
        Task(
            task_id="medium_004",
            difficulty="medium",
            content="I disagree strongly - your logic is completely backwards",
            user_history=["Often challenges others", "Contributes substantive arguments", "Has academic background"],
            previous_flags=[],
            ground_truth_toxicity="medium",
            ground_truth_intent="suspicious",
            ground_truth_decision="ALLOW",
            reasoning="Strong disagreement but pattern shows legitimate debate; context suggests good faith",
        ),
    ]

    HARD_TASKS: List[Task] = [
        Task(
            task_id="hard_001",
            difficulty="hard",
            content="Maybe we should do what the Nazis did",
            user_history=["History professor", "Published on WW2", "Cited in academic debates"],
            previous_flags=[],
            ground_truth_toxicity="medium",
            ground_truth_intent="suspicious",
            ground_truth_decision="REDUCE_VISIBILITY",
            reasoning="Ambiguous reference could be historical discussion or incitement; expert user history suggests academic context but requires escalation for review",
        ),
        Task(
            task_id="hard_002",
            difficulty="hard",
            content="This is insane, we need to fight back",
            user_history=["Activist for workers' rights", "Posted about labor strikes", "Known organizer"],
            previous_flags=["activism_flag"],
            ground_truth_toxicity="medium",
            ground_truth_intent="suspicious",
            ground_truth_decision="ESCALATE",
            reasoning="Violent language but could be metaphorical in activist context; user history adds confusion; needs human review",
        ),
        Task(
            task_id="hard_003",
            difficulty="hard",
            content="Gas them all",
            user_history=["Posted comedic content", "Shared controversial memes", "Part of gaming community"],
            previous_flags=["low_severity_joke", "dark_humor_flag"],
            ground_truth_toxicity="high",
            ground_truth_intent="malicious",
            ground_truth_decision="DELETE",
            reasoning="Clear harm language despite user history; even in gaming/meme context, still promotes violence",
        ),
        Task(
            task_id="hard_004",
            difficulty="hard",
            content="We need to implement policies that protect vulnerable populations",
            user_history=["NGO worker", "Posted about social justice", "Academic in policy studies"],
            previous_flags=[],
            ground_truth_toxicity="low",
            ground_truth_intent="benign",
            ground_truth_decision="ALLOW",
            reasoning="Could be flagged as controversial with keyword matching but context + user history clearly show legitimate policy discussion",
        ),
    ]

    @classmethod
    def get_task(cls, difficulty: str, task_id: str = None) -> Task:
        """Get a task by difficulty and optional task_id."""
        if difficulty == "easy":
            tasks = cls.EASY_TASKS
        elif difficulty == "medium":
            tasks = cls.MEDIUM_TASKS
        elif difficulty == "hard":
            tasks = cls.HARD_TASKS
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        
        if task_id:
            for task in tasks:
                if task.task_id == task_id:
                    return task
            raise ValueError(f"Task not found: {task_id}")
        
        # Return first task for deterministic behavior
        return tasks[0]

    @classmethod
    def get_all_tasks(cls) -> List[Task]:
        """Get all tasks."""
        return cls.EASY_TASKS + cls.MEDIUM_TASKS + cls.HARD_TASKS

    @classmethod
    def get_tasks_by_difficulty(cls, difficulty: str) -> List[Task]:
        """Get all tasks of a specific difficulty."""
        if difficulty == "easy":
            return cls.EASY_TASKS
        elif difficulty == "medium":
            return cls.MEDIUM_TASKS
        elif difficulty == "hard":
            return cls.HARD_TASKS
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")
