"""
Multi-Agent Supervisor System (LangGraph + Claude)
----------------------------------------------------
A supervisor agent orchestrates two specialized worker agents:
  1. Research Agent  -> answers questions using live web search
  2. Analyst Agent    -> performs numeric/statistical calculations

The supervisor decides which worker to delegate to (or answer directly),
and workers hand control back to the supervisor when finished.

Author: (your name here)
Built on the LangGraph multi-agent supervisor pattern, adapted to:
  - use Anthropic's Claude instead of OpenAI
  - use a free DuckDuckGo search tool instead of a paid search API
  - add a small analyst toolset (mean/std/growth-rate) instead of toy math ops
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command
from langchain_core.tools import InjectedToolCallId
from typing import Annotated

load_dotenv()

# Groq hosts several models behind an OpenAI-compatible API.
# "openai/gpt-oss-120b" is OpenAI's open-weight model, served on Groq's
# fast inference hardware. Swap via GROQ_MODEL in .env if you prefer
# e.g. "llama-3.3-70b-versatile" or "openai/gpt-oss-20b".
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ---------------------------------------------------------------------------
# 1. Tools for each worker agent
# ---------------------------------------------------------------------------

web_search = DuckDuckGoSearchRun()


@tool
def mean(numbers: list[float]) -> float:
    """Compute the arithmetic mean of a list of numbers."""
    return sum(numbers) / len(numbers)


@tool
def std_dev(numbers: list[float]) -> float:
    """Compute the (population) standard deviation of a list of numbers."""
    m = sum(numbers) / len(numbers)
    variance = sum((x - m) ** 2 for x in numbers) / len(numbers)
    return variance ** 0.5


@tool
def growth_rate(start_value: float, end_value: float) -> float:
    """Compute the percentage growth rate between a start and end value."""
    if start_value == 0:
        raise ValueError("start_value cannot be zero")
    return ((end_value - start_value) / start_value) * 100


# ---------------------------------------------------------------------------
# 2. Worker agents
# ---------------------------------------------------------------------------

llm = ChatGroq(model=MODEL_NAME, temperature=0)

research_agent = create_react_agent(
    llm,
    tools=[web_search],
    prompt=(
        "You are a research agent. Use the web_search tool to find current, "
        "factual information. Be concise and cite what you found. "
        "Do not attempt calculations — hand those back to the supervisor."
    ),
    name="research_agent",
)

analyst_agent = create_react_agent(
    llm,
    tools=[mean, std_dev, growth_rate],
    prompt=(
        "You are a data analyst agent. Use your tools to perform numeric "
        "calculations (mean, standard deviation, growth rate). "
        "Do not search the web — hand that back to the supervisor."
    ),
    name="analyst_agent",
)


# ---------------------------------------------------------------------------
# 3. Handoff tools — how the supervisor delegates to a worker
# ---------------------------------------------------------------------------

def make_handoff_tool(agent_name: str, description: str):
    @tool(f"delegate_to_{agent_name}", description=description)
    def handoff(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Delegated to {agent_name}.",
            "name": f"delegate_to_{agent_name}",
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,
            update={"messages": state["messages"] + [tool_message]},
            graph=Command.PARENT,
        )

    return handoff


delegate_to_research = make_handoff_tool(
    "research_agent", "Delegate a question that needs live web research."
)
delegate_to_analyst = make_handoff_tool(
    "analyst_agent", "Delegate a question that needs numeric calculation."
)

supervisor_agent = create_react_agent(
    llm,
    tools=[delegate_to_research, delegate_to_analyst],
    prompt=(
        "You are a supervisor managing two workers: research_agent and "
        "analyst_agent. For questions needing current/factual web info, "
        "delegate to research_agent. For questions needing math/statistics, "
        "delegate to analyst_agent. For simple questions, answer directly."
    ),
    name="supervisor",
)


# ---------------------------------------------------------------------------
# 4. Build the graph
# ---------------------------------------------------------------------------

builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor_agent)
builder.add_node("research_agent", research_agent)
builder.add_node("analyst_agent", analyst_agent)

builder.add_edge(START, "supervisor")
# Workers report back to the supervisor once finished
builder.add_edge("research_agent", "supervisor")
builder.add_edge("analyst_agent", "supervisor")

graph = builder.compile()


# ---------------------------------------------------------------------------
# 5. CLI entry point
# ---------------------------------------------------------------------------

def run_query(user_input: str):
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
    final_message = result["messages"][-1]
    print("\n--- Answer ---")
    print(final_message.content)
    print("--------------\n")


if __name__ == "__main__":
    print("Multi-Agent Supervisor (Claude + LangGraph)")
    print("Type a question, or 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        run_query(user_input)
