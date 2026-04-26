"""
AI Smart Agent — ReAct-style reasoning loop.

Flow:
  1. Use Llama 3 (via Groq) to decide which tools to call (tool selection step)
  2. Execute each tool and collect results
  3. Use Llama 3 (via Groq) to synthesise a final answer from the tool results

All LLM calls go through Groq API (free tier).
"""

import os
import json
import re
from typing import Generator

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agent.tools import ALL_TOOLS, TOOL_MAP, Tool


def _get_llm() -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is not set.")
    return ChatGroq(
        model="llama3-8b-8192",
        temperature=0.1,
        max_tokens=512,
    )


TOOL_SELECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant that decides which tools to use to answer a question.

Available tools:
{tool_list}

Respond with a JSON array of tool calls. Each item has "tool" (tool name) and "input" (what to pass to the tool).
Only include tools that are actually needed.
Respond with ONLY the JSON array, nothing else.

Example: [{{"tool": "wikipedia", "input": "Python programming language"}}]"""),
    ("human", "{question}"),
])

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Answer the user's question using the tool results below.
Be concise and direct. If the tools didn't return useful information, say so.

Tool results:
{tool_results}"""),
    ("human", "{question}"),
])

DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])


class SmartAgent:
    def __init__(self):
        self.llm = _get_llm()
        self._parser = StrOutputParser()

    def _select_tools(self, question: str) -> list[dict]:
        tool_list = "\n".join(
            f"- {t.name}: {t.description}" for t in ALL_TOOLS
        )
        chain = TOOL_SELECTION_PROMPT | self.llm | self._parser
        raw = chain.invoke({"question": question, "tool_list": tool_list})

        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return []

    def _run_tools(self, tool_calls: list[dict]) -> list[dict]:
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
        formatted = "\n\n".join(
            f"[{r['tool']}] Input: {r['input']}\nResult: {r['output']}"
            for r in tool_results
        )
        chain = SYNTHESIS_PROMPT | self.llm | self._parser
        return chain.invoke({"question": question, "tool_results": formatted})

    def run(self, question: str) -> Generator[tuple[str, str], None, None]:
        """
        Run the agent. Yields (thought_log, answer) tuples as steps complete.
        """
        thought_log = ""

        thought_log += "**Thinking...** Deciding which tools to use.\n\n"
        yield thought_log, ""

        tool_calls = self._select_tools(question)

        if not tool_calls:
            thought_log += "_No tools needed — answering directly._\n\n"
            yield thought_log, ""
            chain = DIRECT_ANSWER_PROMPT | self.llm | self._parser
            answer = chain.invoke({"question": question})
            yield thought_log, answer
            return

        for call in tool_calls:
            thought_log += f"**Using tool:** `{call['tool']}`\n"
            thought_log += f"**Input:** {call['input']}\n\n"
            yield thought_log, ""

        tool_results = self._run_tools(tool_calls)

        for r in tool_results:
            thought_log += f"**Result from `{r['tool']}`:**\n{r['output']}\n\n"
            yield thought_log, ""

        thought_log += "**Synthesising answer...**\n\n"
        yield thought_log, ""

        answer = self._synthesise(question, tool_results)
        yield thought_log, answer
