#!/usr/bin/env python3
"""
Gradio web interface for Trust and Safety Engine
Optimized for Hugging Face Spaces
"""

import gradio as gr
from environment import TrustAndSafetyEnv
from models import Action, EpisodeConfig, StepType
from typing import Optional


class WebInterface:
    """Interactive Gradio interface for the Trust and Safety Engine."""
    
    def __init__(self):
        self.env = TrustAndSafetyEnv()
        self.current_episode = None
        self.episode_history = []
    
    def reset_episode(self, difficulty: str) -> tuple:
        """Start a new episode."""
        try:
            config = EpisodeConfig(difficulty=difficulty)
            reset_result = self.env.reset(config)
            self.current_episode = {
                'episode_id': reset_result.episode_id,
                'difficulty': difficulty,
                'observation': reset_result.observation,
                'total_reward': 0.0,
                'steps': 0,
                'actions': []
            }
            
            obs = reset_result.observation
            display_text = f"""
**Episode Started** 🎬
- Episode ID: {reset_result.episode_id}
- Difficulty: {difficulty}
- Task: {obs.content[:100]}...

**Current Step: ANALYZE** 📊
Please analyze the content's toxicity and intent.
"""
            
            return (
                display_text,
                "Ready for action ✅",
                ""  # Clear input
            )
        except Exception as e:
            return (
                f"❌ Error: {str(e)}",
                "Error ❌",
                ""
            )
    
    def take_action(self, action_text: str) -> tuple:
        """Execute an action in the environment."""
        if not self.current_episode:
            return (
                "❌ No episode running. Click 'Reset Episode' first.",
                "Error ❌",
                ""
            )
        
        if not action_text.strip():
            return (
                "❌ Please enter an action.",
                "Error ❌",
                ""
            )
        
        try:
            action = Action(action=action_text)
            result = self.env.step(action)
            
            self.current_episode['steps'] += 1
            self.current_episode['total_reward'] += result.reward
            self.current_episode['actions'].append(action_text)
            self.current_episode['observation'] = result.observation
            
            obs = result.observation
            step_name = obs.step_type.value.upper()
            
            display_text = f"""
**Step {self.current_episode['steps']}** 
- Reward: +{result.reward:.2f}
- Total Reward: {self.current_episode['total_reward']:.2f}
- Action: {action_text}

"""
            
            if not result.done:
                next_step = obs.step_type.value.upper()
                display_text += f"**Next Step: {next_step}** 📋\n"
                if next_step == "ANALYZE":
                    display_text += "Analyze the content's toxicity and intent.\n"
                elif next_step == "CONTEXT":
                    display_text += "Check user history and previous flags.\n"
                elif next_step == "DECIDE":
                    display_text += "Make final moderation decision.\n"
                
                status = f"Step {self.current_episode['steps']} ✓"
            else:
                # Episode done
                grade = result.info.get('grade', 0.0)
                analysis = result.info.get('episode_analysis', {})
                
                display_text += f"""
**Episode Complete!** 🏁
- Final Grade: {grade:.2%}
- Total Reward: {self.current_episode['total_reward']:.2f}
- Steps Taken: {self.current_episode['steps']}

**Analysis:**
- Toxicity Score: {analysis.get('avg_toxicity_score', 0):.2f}
- Context Score: {analysis.get('avg_context_score', 0):.2f}
- Decision Score: {analysis.get('avg_decision_score', 0):.2f}
"""
                status = f"Complete! Grade: {grade:.1%} 🎉"
                self.episode_history.append(self.current_episode)
                self.current_episode = None
            
            return (
                display_text,
                status,
                ""  # Clear input
            )
        
        except Exception as e:
            return (
                f"❌ Error: {str(e)}\n\nMake sure your action format is correct.",
                "Error ❌",
                ""
            )


def build_interface() -> gr.Blocks:
    """Build the Gradio interface."""
    
    interface = WebInterface()
    
    with gr.Blocks(
        title="Trust and Safety Engine",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
# 🛡️ Trust and Safety Engine
        
Interactive content moderation environment. Help make moderation decisions using multi-stage analysis.

**How it works:**
1. Select difficulty level
2. Click "Reset Episode" to start
3. Enter actions following the format shown
4. See rewards and episode grades
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Settings")
                difficulty = gr.Radio(
                    choices=["easy", "medium", "hard"],
                    value="easy",
                    label="Difficulty Level"
                )
                reset_btn = gr.Button("🔄 Reset Episode", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### Instructions")
                gr.Markdown("""
- **ANALYZE**: toxicity=low/medium/high, intent=benign/suspicious/malicious
- **CONTEXT**: CHECK_HISTORY
- **DECIDE**: DELETE/ALLOW/REDUCE_VISIBILITY/ESCALATE

Example: `ANALYZE: toxicity=high, intent=malicious`
                """)
        
        # Display area
        display = gr.Textbox(
            label="Episode Output",
            interactive=False,
            lines=15,
            max_lines=20
        )
        
        status = gr.Textbox(
            label="Status",
            interactive=False,
            lines=1
        )
        
        # Action input
        gr.Markdown("### Your Action")
        with gr.Row():
            action_input = gr.Textbox(
                placeholder="Enter action (e.g., ANALYZE: toxicity=high, intent=malicious)",
                label="Action",
                lines=2
            )
            action_btn = gr.Button("➡️ Take Action", variant="primary", size="lg")
        
        # Connect events
        reset_btn.click(
            fn=interface.reset_episode,
            inputs=[difficulty],
            outputs=[display, status, action_input]
        )
        
        action_btn.click(
            fn=interface.take_action,
            inputs=[action_input],
            outputs=[display, status, action_input]
        )
        
        # Allow Enter key to submit action
        action_input.submit(
            fn=interface.take_action,
            inputs=[action_input],
            outputs=[display, status, action_input]
        )
        
        # Footer
        gr.Markdown("""
---
**Trust and Safety Engine** | OpenEnv-compatible RL environment
- [Documentation](https://huggingface.co/spaces/docs)
- [Source Code](https://github.com)
        """)
    
    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
