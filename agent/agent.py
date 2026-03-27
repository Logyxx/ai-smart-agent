"""
AI Smart Agent — ReAct-style reasoning loop.

Flow:
  1. Use Mistral-7B to decide which tools to call (tool selection step)
  2. Execute each tool and collect results
  3. Use Mistral-7B to synthesise a final answer from the tool results

All LLM calls go through HuggingFace Inference API (free with HF account).
"""

import os
import json
import re
from typing import Generator

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agent.tools import ALL_TOOLS, TOOL_MAP, Tool

# ── LLM setup ─────────────────────────────────────────────────────────────────

def _get_llm() -> HuggingFaceEndpoint:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN is not set.")
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=hf_token,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
    )


# ── Prompts ───────────────────────────────────────────────────────────────────

TOOL_SELECTION_TEMPLATE = """\
<s>[INST] You are an AI assistant that decides which tools to use to answer a question.

Available tools:
{tool_list}

Question: {question}

Respond with a JSON array of tool calls. Each item has "tool" (tool name) and "input" (what to pass to the tool).
Only include tools that are actually needed.
Respond with ONLY the JSON array, nothing else.

Example: [{{"tool": "wikipedia", "input": "Python programming language"}}]
[/INST]"""

SYNTHESIS_TEMPLATE = """\
<s>[INST] You are a helpful AI assistant. Answer the user's question using the tool results below.
Be concise and direct. If the tools didn't return useful information, say so.

Question: {question}

Tool results:
{tool_results}

Answer: [/INST]"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class SmartAgent:
    def __init__(self):
        self.llm = _get_llm()
        self._parser = StrOutputParser()

    def _select_tools(self, question: str) -> list[dict]:
        """Ask Mistral which tools to call for this question."""
        tool_list = "\n".join(
            f"- {t.name}: {t.description}" for t in ALL_TOOLS
        )
        prompt = PromptTemplate.from_template(TOOL_SELECTION_TEMPLATE)
        chain = prompt | self.llm | self._parser
        raw = chain.invoke({"question": question, "tool_list": tool_list})

        # Extract JSON array from the response
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return []

    def _run_tools(self, tool_calls: list[dict]) -> list[dict]:
        """Execute each tool call and collect results."""
        results = []
        for call in tool_calls:
            tool_name = call.get("tool", "")
            tool_input = call.get("input", "")
            tool = TOOL_MAP.get(tool_name)
            if tool is None:
                results.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "output": f"Unknown tool: {tool_name}",
                })
            else:
                output = tool.run(tool_input)
                results.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "output": output,
                })
        return results

    def _synthesise(self, question: str, tool_results: list[dict]) -> str:
        """Generate a final answer from the tool results."""
        formatted = "\n\n".join(
            f"[{r['tool']}] Input: {r['input']}\nResult: {r['output']}"
            for r in tool_results
        )
        prompt = PromptTemplate.from_template(SYNTHESIS_TEMPLATE)
        chain = prompt | self.llm | self._parser
        return chain.invoke({"question": question, "tool_results": formatted})

    def run(self, question: str) -> Generator[tuple[str, str], None, None]:
        """
        Run the agent. Yields (thought_log, answer) tuples as steps complete.

        Yields intermediate status updates so the UI can stream progress.
        """
        thought_log = ""

        # Step 1: Tool selection
        thought_log += "**Thinking...** Deciding which tools to use.\n\n"
        yield thought_log, ""

        tool_calls = self._select_tools(question)

        if not tool_calls:
            thought_log += "_No tools needed — answering directly._\n\n"
            yield thought_log, ""
            # Fall back to direct answer
            prompt = PromptTemplate.from_template(
                "<s>[INST] {question} [/INST]"
            )
            chain = prompt | self.llm | self._parser
            answer = chain.invoke({"question": question})
            yield thought_log, answer
            return

        # Step 2: Execute tools
        for call in tool_calls:
            thought_log += f"**Using tool:** `{call['tool']}`\n"
            thought_log += f"**Input:** {call['input']}\n\n"
            yield thought_log, ""

        tool_results = self._run_tools(tool_calls)

        for r in tool_results:
            thought_log += f"**Result from `{r['tool']}`:**\n{r['output']}\n\n"
            yield thought_log, ""

        # Step 3: Synthesise answer
        thought_log += "**Synthesising answer...**\n\n"
        yield thought_log, ""

        answer = self._synthesise(question, tool_results)
        yield thought_log, answer
