---
title: AI Smart Agent
emoji: 🤖
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "4.0.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# 🤖 AI Smart Agent

An **AI agent** that reasons step-by-step — selecting the right tools, executing them, and synthesising a grounded answer. Ask anything: current events, maths, facts, or the date.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?logo=langchain&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-FF7C00?logo=gradio&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace_Spaces-deployed-yellow?logo=huggingface&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Live Demo

> 🔗 **[Try it on Hugging Face Spaces →](https://huggingface.co/spaces/lakshmenroy/ai-smart-agent)**

## How the Agent Works

```
User question
    │
    ├─▶ LLM (Mistral-7B): "Which tools do I need?"
    │       └─▶ Returns JSON list of tool calls
    │
    ├─▶ Execute tools (in parallel concept):
    │       ├─ datetime   → current date/time
    │       ├─ calculator → evaluate maths expression
    │       ├─ wikipedia  → fetch Wikipedia summary
    │       └─ web_search → DuckDuckGo results
    │
    └─▶ LLM (Mistral-7B): synthesise final answer from tool results
            └─▶ Answer grounded in real tool output
```

The **thought process panel** in the UI shows exactly which tools were called and what they returned — full transparency into the agent's reasoning.

## Features

- **4 tools** — datetime, calculator, Wikipedia, DuckDuckGo web search
- **LLM-driven tool selection** — Mistral-7B decides which tools to use based on the question
- **Transparent reasoning** — thought process panel shows every tool call and result
- **Completely free** — Mistral-7B via HuggingFace Inference API, all tools have no API keys
- **Gradio chat UI** — conversation history preserved across questions

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.11 | Core language |
| LangChain | LLM orchestration (LCEL chains) |
| Mistral-7B-Instruct | LLM — tool selection + answer synthesis (free via HF API) |
| duckduckgo-search | Free web search, no API key |
| wikipedia | Free Wikipedia summaries, no API key |
| Gradio | Chat interface with thought process panel |
| HuggingFace Spaces | Deployment |

## Getting Started

```bash
git clone https://github.com/ByteMe-UK/ai-smart-agent.git
cd ai-smart-agent

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export HF_TOKEN=hf_...   # your HuggingFace token (free)
python app.py
```

Open `http://localhost:7860` and ask a question.

## Example Questions

| Question | Tools Used |
|---------|-----------|
| "What is today's date?" | `datetime` |
| "What is 15% of 340?" | `calculator` |
| "Who invented the World Wide Web?" | `wikipedia` |
| "What is the latest news about AI?" | `web_search` |
| "What is the capital of Japan and what time is it?" | `wikipedia` + `datetime` |

## Project Structure

```
ai-smart-agent/
├── app.py                  ← Gradio UI — question input, thought panel, chat
├── agent/
│   ├── tools.py            ← Tool implementations (datetime, calc, wiki, search)
│   └── agent.py            ← SmartAgent class — tool selection + synthesis loop
├── .github/
│   └── workflows/
│       └── sync_to_hf.yml  ← Auto-sync to HuggingFace on push
├── requirements.txt
└── README.md
```

## Deployment (HuggingFace Spaces)

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space) → SDK: **Gradio**
2. Add `HFTOKEN` to GitHub repo secrets (for the sync Action)
3. Add `HF_TOKEN` as a **Space secret** → Settings → Variables and secrets
4. Push to `main` — GitHub Action syncs automatically

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Part of the [ByteMe-UK](https://github.com/ByteMe-UK) portfolio collection.**
