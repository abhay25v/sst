"""
Inference script for the Trust and Safety Decision Engine.
Uses OpenAI client to run the environment loop with an LLM-based agent.
Supports structured logging for evaluation harness.
"""
import json
import os
import sys
from typing import List, Dict, Any, Optional
from openai import OpenAI
from environment import TrustAndSafetyEnv
from models import Action, EpisodeConfig
from reward import EpisodeAnalyzer, DeterministicGrader


# ============================================================================
# Structured Logging Functions (Required by Validator)
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    """Log episode start in structured format."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None
) -> None:
    """Log step in structured format."""
    error_str = f"error={error}" if error else "error=null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={done} {error_str}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end in structured format."""
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards=[{rewards_str}]", flush=True)


class TrustAndSafetyInference:
    """Runs inference with an LLM agent in the environment."""
    
    def __init__(self, model: str = None, api_key: str = None, api_base: str = None):
        """
        Initialize inference engine.
        
        Args:
            model: OpenAI model to use (defaults to MODEL_NAME env var)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            api_base: API base URL (defaults to API_BASE_URL env var)
        """
        # Read from environment variables with defaults
        self.model = model or os.getenv("MODEL_NAME", "gpt-4")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        self.hf_token = os.getenv("HF_TOKEN")
        
        # Allow dummy key for non-inference operations
        if not self.api_key and "_FORCE_INFERENCE_" in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable not set but inference is being used")
        
        # Initialize OpenAI client with base URL (defer if no API key)
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
        else:
            self.client = None
        
        self.env = TrustAndSafetyEnv()
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        return """You are an expert content moderator for a social media platform.
Your task is to evaluate content and determine appropriate moderation actions.

The moderation process has 3 stages:
1. ANALYZE: Classify the content's toxicity level and intent
   - Toxicity: low, medium, high
   - Intent: benign, suspicious, malicious
   - Format: "ANALYZE: toxicity=<level>, intent=<intent>"

2. CONTEXT: Check user's history and previous flags
   - Action: "CHECK_HISTORY"
   - This helps you understand if there's a pattern of abuse

3. DECIDE: Make the final moderation decision
   - Format: "DECIDE: <action>" where action is one of:
     - ALLOW: Allow the content
     - REDUCE_VISIBILITY: Reduce visibility (limiting reach)
     - DELETE: Remove the content
     - ESCALATE: Send to human review
   - Alternative: "ESCALATE" directly for ambiguous cases

Decision-making principles:
- ALLOW: Content violates no policies
- REDUCE_VISIBILITY: Borderline content or repeat minor offenses
- DELETE: Clear policy violation, harmful content
- ESCALATE: Ambiguous cases needing human judgment

Examples:
- Obviously toxic death threat → ANALYZE → CHECK_HISTORY → DECIDE: DELETE
- Mild criticism from established user → ANALYZE → DECIDE: ALLOW
- Questionable reference needing context → ANALYZE → CHECK_HISTORY → ESCALATE

Always proceed through the stages. Be thoughtful about user context.
Format your actions precisely as shown. Never deviate from the action formats."""
    
    def run_episode(
        self,
        difficulty: str = "medium",
        task_id: str = None,
        max_steps: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run a single episode with the LLM agent.
        
        Args:
            difficulty: Task difficulty (easy, medium, hard)
            task_id: Optional specific task ID
            max_steps: Maximum steps per episode
            verbose: Print step-by-step output
        
        Returns:
            Episode results with rewards, actions, and grade
        """
        config = EpisodeConfig(difficulty=difficulty, task_id=task_id)
        reset_result = self.env.reset(config)
        observation = reset_result.observation
        episode_info = reset_result.info
        
        # Log episode start
        log_start(
            task=f"{difficulty}_{episode_info['task_id']}",
            env="TrustAndSafetyEnv",
            model=self.model
        )
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Episode: {episode_info['episode_id']}")
            print(f"Difficulty: {episode_info['difficulty']}")
            print(f"Task: {episode_info['task_id']}")
            print(f"{'='*80}")
            print(f"\nContent to moderate:\n{observation.content}\n")
        
        messages = [{"role": "system", "content": self.system_prompt}]
        step_count = 0
        episode_rewards = []
        episode_done = False
        episode_success = False
        
        try:
            while step_count < max_steps:
                # Build context for the agent
                context = self._build_agent_context(observation, step_count)
                messages.append({"role": "user", "content": context})
                
                # Get agent action
                error_msg = None
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=100,
                    )
                    action_text = response.choices[0].message.content.strip()
                    messages.append({"role": "assistant", "content": action_text})
                except Exception as e:
                    error_msg = str(e)
                    if verbose:
                        print(f"Error calling OpenAI: {e}")
                    action_text = "ESCALATE"
                
                # Parse and execute action
                action = Action(action=action_text)
                step_result = self.env.step(action)
                observation = step_result.observation
                reward = step_result.reward
                done = step_result.done
                
                step_count += 1
                episode_rewards.append(reward)
                
                # Log step
                log_step(
                    step=step_count,
                    action=action_text,
                    reward=reward,
                    done=done,
                    error=error_msg
                )
                
                if verbose:
                    print(f"Step {step_count}: {self.env.current_step_type.value}")
                    print(f"  Action: {action_text}")
                    print(f"  Reward: {reward:.3f}")
                    print(f"  Valid: {step_result.info.get('action_valid', False)}")
                    if not step_result.info.get('action_valid', False):
                        print(f"  Error: Invalid action format")
                
                if done:
                    episode_done = True
                    if verbose:
                        print(f"\nEpisode complete!")
                        if "episode_analysis" in step_result.info:
                            analysis = step_result.info["episode_analysis"]
                            print(f"  Final decision: {analysis['final_decision']}")
                            print(f"  Ground truth: {analysis['ground_truth_decision']}")
                            print(f"  Correct: {analysis['correct']}")
                            print(f"  Total reward: {analysis['total_reward']:.3f}")
                        if "grade" in step_result.info:
                            print(f"  Grade: {step_result.info['grade']:.3f}")
                    break
            
            # Compile episode results
            state = self.env.state()
            total_reward = sum(episode_rewards)
            score = min(max(total_reward / 0.7, 0.0), 1.0)  # Normalize to [0, 1] based on max reward
            episode_success = score >= 0.6  # Success threshold
            
            final_info = {
                "episode_id": episode_info["episode_id"],
                "difficulty": episode_info["difficulty"],
                "task_id": episode_info["task_id"],
                "steps": step_count,
                "total_reward": total_reward,
                "average_reward": sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0,
                "actions": state["actions_taken"],
                "score": score,
                "success": episode_success,
            }
            
            # Add analysis if episode completed
            if episode_done and hasattr(self.env, 'current_task'):
                task = self.env.current_task
                analysis = EpisodeAnalyzer.analyze(
                    actions=state["actions_taken"],
                    step_types=state["step_types"],
                    rewards=state["rewards_received"],
                    ground_truth_decision=task.ground_truth_decision,
                )
                final_info.update(analysis)
                
                grade = DeterministicGrader.grade_episode(
                    actions=state["actions_taken"],
                    step_types=state["step_types"],
                    ground_truth_toxicity=task.ground_truth_toxicity,
                    ground_truth_intent=task.ground_truth_intent,
                    ground_truth_decision=task.ground_truth_decision,
                    user_history_length=len(task.user_history),
                    previous_flags_length=len(task.previous_flags),
                )
                final_info["grade"] = grade
            
            if verbose:
                print(f"\nResults: {json.dumps(final_info, indent=2)}\n")
            
        finally:
            # Log episode end
            log_end(
                success=episode_success,
                steps=step_count,
                score=score if step_count > 0 else 0.0,
                rewards=episode_rewards
            )
        
        return final_info
    
    def run_benchmark(
        self,
        difficulties: List[str] = None,
        num_per_difficulty: int = 1,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run a benchmark across multiple tasks.
        
        Args:
            difficulties: List of difficulties to test
            num_per_difficulty: Number of tasks per difficulty
            verbose: Print detailed output
        
        Returns:
            Benchmark results with aggregate statistics
        """
        from dataset import Dataset
        
        if difficulties is None:
            difficulties = ["easy", "medium", "hard"]
        
        results_by_difficulty = {d: [] for d in difficulties}
        all_results = []
        
        for difficulty in difficulties:
            tasks = Dataset.get_tasks_by_difficulty(difficulty)
            num_tasks = min(num_per_difficulty, len(tasks))
            
            for i in range(num_tasks):
                task = tasks[i]
                result = self.run_episode(
                    difficulty=difficulty,
                    task_id=task.task_id,
                    verbose=verbose
                )
                results_by_difficulty[difficulty].append(result)
                all_results.append(result)
        
        # Aggregate results
        benchmark_results = {
            "total_episodes": len(all_results),
            "by_difficulty": {},
            "overall": self._compute_aggregate_stats(all_results),
        }
        
        for difficulty, results in results_by_difficulty.items():
            if results:
                benchmark_results["by_difficulty"][difficulty] = self._compute_aggregate_stats(results)
        
        return benchmark_results
    
    def _build_agent_context(self, observation: Any, step_count: int) -> str:
        """Build the context message for the agent."""
        context = f"""
Current step type: {observation.step_type.value}
Step number: {step_count + 1}/3

Content to evaluate:
{observation.content}

User history (previous posts):
{self._format_list(observation.user_history) if observation.user_history else '(empty)'}

Previous moderation flags on this user:
{self._format_list(observation.previous_flags) if observation.previous_flags else '(none)'}

Last action error: {observation.last_action_error}

Task: Please take the next action in your moderation decision process.
Remember to format your action exactly as specified.
"""
        return context
    
    @staticmethod
    def _format_list(items: List[str]) -> str:
        """Format a list for display."""
        return "\n".join(f"  - {item}" for item in items)
    
    @staticmethod
    def _compute_aggregate_stats(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute aggregate statistics from results."""
        if not results:
            return {}
        
        total_reward = sum(r.get("total_reward", 0) for r in results)
        correct_count = sum(1 for r in results if r.get("correct", False))
        grades = [r.get("grade", 0) for r in results if "grade" in r]
        
        return {
            "num_episodes": len(results),
            "avg_total_reward": total_reward / len(results),
            "accuracy": correct_count / len(results),
            "avg_grade": sum(grades) / len(grades) if grades else 0.0,
            "num_correct": correct_count,
        }


def main():
    """Main entry point for inference."""
    import argparse
    from dataset import Dataset
    
    parser = argparse.ArgumentParser(description="Run Trust and Safety inference")
    parser.add_argument("--model", default=None, help="OpenAI model to use (default: MODEL_NAME env var)")
    parser.add_argument("--difficulty", default=None, help="Task difficulty (easy, medium, hard)")
    parser.add_argument("--task-id", default=None, help="Specific task ID")
    parser.add_argument("--all-tasks", action="store_true", help="Run all 12 tasks")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark across all difficulties")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        inference = TrustAndSafetyInference(model=args.model)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    
    if args.all_tasks or args.benchmark:
        # Run all 12 tasks
        print(f"\n{'='*80}")
        print("Running all 12 tasks (easy, medium, hard)")
        print(f"{'='*80}\n", flush=True)
        
        all_results = []
        results_by_difficulty = {d: [] for d in ["easy", "medium", "hard"]}
        
        for difficulty in ["easy", "medium", "hard"]:
            tasks = Dataset.get_tasks_by_difficulty(difficulty)
            for task in tasks:
                print(f"\nRunning {difficulty.upper()} task: {task.task_id}", flush=True)
                result = inference.run_episode(
                    difficulty=difficulty,
                    task_id=task.task_id,
                    verbose=args.verbose or False
                )
                all_results.append(result)
                results_by_difficulty[difficulty].append(result)
        
        # Print summary
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}\n", flush=True)
        
        for difficulty in ["easy", "medium", "hard"]:
            results = results_by_difficulty[difficulty]
            if results:
                avg_score = sum(r.get("score", 0) for r in results) / len(results)
                avg_reward = sum(r.get("total_reward", 0) for r in results) / len(results)
                num_correct = sum(1 for r in results if r.get("correct", False))
                print(f"{difficulty.upper()} ({len(results)} tasks):")
                print(f"  Avg Score: {avg_score:.4f}")
                print(f"  Avg Reward: {avg_reward:.4f}")
                print(f"  Correct: {num_correct}/{len(results)}")
                print()
        
        overall_avg_score = sum(r.get("score", 0) for r in all_results) / len(all_results) if all_results else 0.0
        overall_avg_reward = sum(r.get("total_reward", 0) for r in all_results) / len(all_results) if all_results else 0.0
        
        print(f"OVERALL:")
        print(f"  Total Episodes: {len(all_results)}")
        print(f"  Avg Score: {overall_avg_score:.4f}")
        print(f"  Avg Reward: {overall_avg_reward:.4f}")
        print(f"{'='*80}\n", flush=True)
        
    else:
        # Run single task
        if not args.difficulty:
            args.difficulty = "medium"
        
        result = inference.run_episode(
            difficulty=args.difficulty,
            task_id=args.task_id,
            verbose=True
        )


if __name__ == "__main__":
    main()
