"""
Tool implementations for the AI agent.

Available tools (all free, no API keys):
  - datetime_tool   — current date and time
  - calculator      — evaluate a maths expression
  - wikipedia       — fetch a Wikipedia summary
  - web_search      — DuckDuckGo web search
"""

import re
import datetime
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]

    def run(self, input_text: str) -> str:
        try:
            return self.func(input_text)
        except Exception as e:
            return f"Tool error: {e}"


# ── Tool: current datetime ────────────────────────────────────────────────────

def _datetime(_: str) -> str:
    now = datetime.datetime.now()
    return now.strftime("Today is %A, %d %B %Y. The time is %H:%M.")


datetime_tool = Tool(
    name="datetime",
    description="Returns the current date and time. Use when asked about today's date or current time.",
    func=_datetime,
)


# ── Tool: calculator ──────────────────────────────────────────────────────────

_SAFE_MATH = re.compile(r"^[\d\s\+\-\*\/\.\(\)\^%]+$")


def _calculator(expression: str) -> str:
    # Strip any non-math characters for safety
    expr = expression.strip()
    expr = expr.replace("^", "**")

    # Only allow digits, operators, dots, parentheses
    if not _SAFE_MATH.match(expr.replace("**", "^^")):
        return "Invalid expression. Only numeric calculations are supported."

    result = eval(expr, {"__builtins__": {}})  # noqa: S307
    return f"{expression.strip()} = {result}"


calculator = Tool(
    name="calculator",
    description="Evaluates a mathematical expression. Use for arithmetic, percentages, powers. Input must be a valid maths expression like '25 * 4 + 10'.",
    func=_calculator,
)


# ── Tool: Wikipedia ───────────────────────────────────────────────────────────

def _wikipedia(query: str) -> str:
    import wikipedia  # type: ignore

    wikipedia.set_lang("en")
    try:
        summary = wikipedia.summary(query, sentences=3, auto_suggest=True)
        return summary
    except wikipedia.DisambiguationError as e:
        # Try the first option
        try:
            summary = wikipedia.summary(e.options[0], sentences=3)
            return summary
        except Exception:
            return f"Ambiguous query. Options include: {', '.join(e.options[:5])}"
    except wikipedia.PageError:
        return f"No Wikipedia article found for '{query}'."


wikipedia_tool = Tool(
    name="wikipedia",
    description="Fetches a short summary from Wikipedia. Use for factual questions about people, places, events, or concepts.",
    func=_wikipedia,
)


# ── Tool: web search (DuckDuckGo) ─────────────────────────────────────────────

def _web_search(query: str) -> str:
    from duckduckgo_search import DDGS  # type: ignore

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))

    if not results:
        return "No results found."

    lines = []
    for r in results:
        lines.append(f"**{r['title']}**\n{r['body']}\nSource: {r['href']}")
    return "\n\n".join(lines)


web_search = Tool(
    name="web_search",
    description="Searches the web using DuckDuckGo. Use for recent news, current events, or anything not on Wikipedia.",
    func=_web_search,
)


# ── Tool registry ─────────────────────────────────────────────────────────────

ALL_TOOLS: list[Tool] = [datetime_tool, calculator, wikipedia_tool, web_search]
TOOL_MAP: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}
