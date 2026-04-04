#!/usr/bin/env python3
"""
Gradio interface for Trust and Safety Decision Engine.
Optimized for Hugging Face Spaces.
"""

import gradio as gr
from environment import TrustAndSafetyEnv
from models import Action, EpisodeConfig


# Lazy initialization
env = None

def get_env() -> TrustAndSafetyEnv:
    """Get or create the global environment instance."""
    global env
    if env is None:
        env = TrustAndSafetyEnv()
    return env


# Session state
session_state = {
    'episode_id': None,
    'total_reward': 0.0,
    'steps': 0,
}


def reset_episode(difficulty: str) -> tuple:
    """Reset the environment and start a new episode."""
    try:
        config = EpisodeConfig(difficulty=difficulty)
        reset_result = get_env().reset(config)
        obs = reset_result.observation
        
        session_state['episode_id'] = obs.episode_id
        session_state['total_reward'] = 0.0
        session_state['steps'] = 0
        
        output = f"""
✅ **Episode Started**
- Episode ID: `{obs.episode_id}`
- Difficulty: **{difficulty}**
- Task: {obs.content}

📊 **Current Stage: ANALYZE**
Please classify the content's toxicity and intent.

Format: `ANALYZE: toxicity=<level>, intent=<intent>`
- toxicity: low, medium, high
- intent: benign, suspicious, malicious
"""
        
        status = f"Ready ✅ | Difficulty: {difficulty}"
        return output, status, ""
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "Error ❌", ""


def take_action(action_text: str, history: str) -> tuple:
    """Execute an action in the environment."""
    
    # Check if episode is running
    if session_state['episode_id'] is None:
        return f"{history}\n\n❌ No active episode. Click 'Reset' first.", "Error ❌", ""
    
    # Check if action is provided
    if not action_text or not action_text.strip():
        return f"{history}\n\n❌ Please enter an action.", "Error ❌", ""
    
    try:
        action = Action(action=action_text.strip())
        result = get_env().step(action)
        
        session_state['steps'] += 1
        session_state['total_reward'] += result.reward
        
        obs = result.observation
        
        # Build response
        response = f"{history}\n\n"
        response += f"🎬 **Step {session_state['steps']}**\n"
        response += f"- Action: `{action_text}`\n"
        response += f"- Reward: +{result.reward:.2f}\n"
        response += f"- Total Reward: {session_state['total_reward']:.2f}\n"
        
        if result.done:
            # Episode finished
            grade = result.info.get('grade', 0.0)
            response += f"\n✅ **Episode Complete!**\n"
            response += f"- Final Grade: **{grade:.1%}**\n"
            response += f"- Total Steps: {session_state['steps']}\n"
            response += f"- Total Reward: {session_state['total_reward']:.2f}\n"
            
            session_state['episode_id'] = None
            status = f"Complete! Grade: {grade:.1%} 🎉"
        else:
            # Next stage
            next_stage = obs.step_type.value.upper()
            response += f"\n📋 **Next Stage: {next_stage}**\n"
            
            if next_stage == "CONTEXT":
                response += "Check user history: `CHECK_HISTORY`"
            elif next_stage == "DECISION":
                response += "Make decision: `DECIDE: <action>` (ALLOW, DELETE, REDUCE_VISIBILITY, ESCALATE)"
            elif next_stage == "ANALYZE":
                response += "Analyze: `ANALYZE: toxicity=<level>, intent=<intent>`"
            
            status = f"Step {session_state['steps']} ✓ | Stage: {next_stage}"
        
        return response, status, ""
        
    except Exception as e:
        error_msg = f"{history}\n\n❌ Error: {str(e)}"
        return error_msg, "Error ❌", ""


# Build Gradio interface
with gr.Blocks(title="Trust & Safety Engine") as demo:
    
    gr.Markdown("# 🛡️ Trust & Safety Decision Engine")
    gr.Markdown("""
An OpenEnv-compatible reinforcement learning environment for content moderation.
Practice making moderation decisions in a realistic multi-stage workflow.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            difficulty = gr.Radio(
                ["easy", "medium", "hard"],
                value="easy",
                label="🎯 Difficulty"
            )
            reset_btn = gr.Button("Reset Episode", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            status = gr.Textbox(
                label="Status",
                interactive=False,
                value="Click 'Reset' to start"
            )
    
    # Main display
    output = gr.Textbox(
        label="Episode Output",
        interactive=False,
        lines=12,
        value="Episode output will appear here..."
    )
    
    # Action input
    with gr.Row():
        action_input = gr.Textbox(
            label="Your Action",
            placeholder="Enter action (e.g., ANALYZE: toxicity=high, intent=malicious)",
            lines=1
        )
        submit_btn = gr.Button("Submit Action", variant="primary")
    
    # Examples
    gr.Examples(
        examples=[
            ["ANALYZE: toxicity=high, intent=malicious"],
            ["CHECK_HISTORY"],
            ["DECIDE: DELETE"],
            ["DECIDE: ALLOW"],
            ["DECIDE: REDUCE_VISIBILITY"],
            ["ESCALATE"],
        ],
        inputs=[action_input],
        label="Action Examples"
    )
    
    gr.Markdown("""
---
## How to Use

1. **Reset**: Choose difficulty and click "Reset Episode"
2. **Analyze**: Classify toxicity (low/medium/high) and intent (benign/suspicious/malicious)
3. **Check Context**: Optionally review user history
4. **Decide**: Make final moderation decision
5. **Grade**: See your performance score

## Action Format
- `ANALYZE: toxicity=<level>, intent=<intent>`
- `CHECK_HISTORY`
- `DECIDE: ALLOW|DELETE|REDUCE_VISIBILITY|ESCALATE`
- `ESCALATE`
    """)
    
    # Event handlers
    reset_btn.click(
        fn=reset_episode,
        inputs=[difficulty],
        outputs=[output, status, action_input]
    )
    
    submit_btn.click(
        fn=take_action,
        inputs=[action_input, output],
        outputs=[output, status, action_input]
    )
    
    action_input.submit(
        fn=take_action,
        inputs=[action_input, output],
        outputs=[output, status, action_input]
    )


# Launch for HF Spaces
if __name__ == "__main__":
    demo.launch()
