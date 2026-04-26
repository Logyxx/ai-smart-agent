"""
AI Smart Agent — Gradio interface.

The agent uses Llama 3 (via Groq API) to decide which tools to call,
runs the tools, then synthesises a final answer.

Tools available:
  - datetime  — current date/time
  - calculator — evaluate maths expressions
  - wikipedia — Wikipedia summaries
  - web_search — DuckDuckGo search

Requires: GROQ_API_KEY environment variable (free at console.groq.com)
"""

import os
import gradio as gr

from agent.agent import SmartAgent

_agent: SmartAgent | None = None


def _get_agent() -> SmartAgent:
    global _agent
    if _agent is None:
        _agent = SmartAgent()
    return _agent


def run_agent(question: str, history: list) -> tuple:
    """Run the agent and stream thought log + final answer to the UI."""
    if not question.strip():
        return "", "", history

    if not os.getenv("GROQ_API_KEY"):
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "⚠️ GROQ_API_KEY is not set. Add it as a Space secret."})
        return "", "", history

    thought_log = ""
    final_answer = ""

    try:
        agent = _get_agent()
        for thought_log, final_answer in agent.run(question):
            pass  # collect final state
    except Exception as e:
        thought_log = f"❌ Agent error: {e}"
        final_answer = ""

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": final_answer or "_No answer generated._"})
    return "", thought_log, history


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title="AI Smart Agent") as demo:
    gr.Markdown(
        """
        # 🤖 AI Smart Agent
        Ask any question — the agent decides which tools to use, runs them, and synthesises an answer.

        > Powered by **Llama 3** via Groq API. Tools: web search, Wikipedia, calculator, datetime.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=420, type="messages")
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask me anything...",
                    label="Question",
                    scale=4,
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    ["What is today's date?"],
                    ["What is 15% of 340?"],
                    ["Who invented the World Wide Web?"],
                    ["What are the latest developments in quantum computing?"],
                    ["What is the capital of Australia and what is it known for?"],
                ],
                inputs=question_box,
                label="Example questions",
            )

        with gr.Column(scale=1):
            thought_box = gr.Markdown(
                value="_Agent thoughts will appear here..._",
                label="Agent Thought Process",
                elem_id="thought-box",
            )

    ask_btn.click(
        fn=run_agent,
        inputs=[question_box, chatbot],
        outputs=[question_box, thought_box, chatbot],
    )
    question_box.submit(
        fn=run_agent,
        inputs=[question_box, chatbot],
        outputs=[question_box, thought_box, chatbot],
    )

    gr.Markdown(
        "---\nBuilt by [Laksh Menroy](https://github.com/lakshmenroy) · "
        "[Logyxx](https://github.com/Logyxx) portfolio"
    )

demo.launch()

