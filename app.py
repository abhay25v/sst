#!/usr/bin/env python3
"""Minimal Gradio app for HF Spaces"""

import sys
print("🟢 Starting app.py...", flush=True)

try:
    import gradio as gr
    print("✓ Gradio imported", flush=True)
except Exception as e:
    print(f"✗ Gradio error: {e}", flush=True)
    sys.exit(1)

try:
    from environment import TrustAndSafetyEnv
    from models import Action, EpisodeConfig
    print("✓ Environment imported", flush=True)
except Exception as e:
    print(f"✗ Import error: {e}", flush=True)
    sys.exit(1)

# Lazy initialization - don't load env until needed
env = None

def get_env():
    global env
    if env is None:
        env = TrustAndSafetyEnv()
    return env

state = {'episode_id': None, 'reward': 0.0, 'steps': 0}

def reset(difficulty):
    try:
        env = get_env()
        result = env.reset(EpisodeConfig(difficulty=difficulty))
        obs = result.observation
        state['episode_id'] = obs.episode_id
        state['reward'] = 0.0
        state['steps'] = 0
        
        return f"✅ Ready!\nEpisode: {obs.episode_id}\nContent: {obs.content[:80]}...", "Ready ✅", ""
    except Exception as e:
        return f"❌ {str(e)[:200]}", "Error", ""

def action(txt):
    if not state['episode_id']:
        return "❌ Reset first", "Error", ""
    try:
        env = get_env()
        result = env.step(Action(action=txt))
        state['reward'] += result.reward
        state['steps'] += 1
        
        msg = f"Step {state['steps']}\nReward: +{result.reward:.2f}\nTotal: {state['reward']:.2f}"
        if result.done:
            grade = result.info.get('grade', 0.0)
            msg += f"\n✅ Done! Grade: {grade:.1%}"
            state['episode_id'] = None
        
        return msg, f"Step {state['steps']}", ""
    except Exception as e:
        return f"❌ {str(e)[:200]}", "Error", ""

print("🟢 Building interface...", flush=True)

with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ Trust & Safety Engine")
    
    diff = gr.Radio(["easy", "medium", "hard"], value="easy", label="Level")
    reset_btn = gr.Button("Reset", variant="primary")
    
    out = gr.Textbox(lines=8, interactive=False)
    stat = gr.Textbox(interactive=False)
    
    act = gr.Textbox(label="Action", lines=1, placeholder="ANALYZE: toxicity=high, intent=malicious")
    btn = gr.Button("Go", variant="primary")
    
    reset_btn.click(reset, [diff], [out, stat, act])
    btn.click(action, [act], [out, stat, act])
    act.submit(action, [act], [out, stat, act])

print("🟢 Launching demo...", flush=True)
demo.launch()
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
            obs = reset_result.observation
            self.current_episode = {
                'episode_id': obs.episode_id,
                'difficulty': difficulty,
                'observation': obs,
                'total_reward': 0.0,
                'steps': 0,
                'actions': []
            }
            
            display_text = f"""
**Episode Started** 🎬
- Episode ID: {obs.episode_id}
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
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
