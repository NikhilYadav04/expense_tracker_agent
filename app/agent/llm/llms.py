# ==============================
# Imports
# ==============================

from typing import Literal
import asyncio
import threading

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.agent.config.config import GEMINI_API_KEY, GROQ_API_KEY


# ==============================
# Async Background Event Loop
# ==============================
# Dedicated async loop to run MCP calls
# without blocking the main thread

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    """Submit coroutine to background event loop."""
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    """Run async coroutine synchronously and return result."""
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Fire-and-forget async task."""
    return _submit_async(coro)


# ==============================
# Pydantic Schemas (Structured Output)
# ==============================


class RouteDecision(BaseModel):
    """
    Router LLM output
    """

    route: Literal["rag", "web", "answer", "end"]
    reply: str | None = Field(None, description="Only filled when route == 'end'")


class RagJudge(BaseModel):
    """
    Judge whether RAG context is sufficient
    """

    sufficient: bool = Field(
        ..., description="True if retrieved info is enough to answer"
    )


# ==============================
# MCP Client Configuration
# ==============================

client = MultiServerMCPClient(
    {
        "Expense Server": {
            "transport": "stdio",
            "command": "N:\\Dev\\Langgraph-Project\\Expense-Whatsapp\\venv\\Scripts\\python.exe",
            "args": ["-m", "app.mcp.expense_server"],
            "env": {"PYTHONPATH": "N:\\Dev\\Langgraph-Project\\Expense-Whatsapp"},
        },
        "Analytics Server": {
            "transport": "stdio",
            "command": "N:\\Dev\\Langgraph-Project\\Expense-Whatsapp\\venv\\Scripts\\python.exe",
            "args": ["-m", "app.mcp.analytics_server"],
            "env": {"PYTHONPATH": "N:\\Dev\\Langgraph-Project\\Expense-Whatsapp"},
        },
    }
)


# ==============================
# MCP Tool Loader
# ==============================


import traceback


def load_mcp_tools() -> list[BaseTool]:
    """
    Fetch tools exposed by MCP servers
    and adapt them into LangChain tools.
    """
    try:
        tools = run_async(client.get_tools())

        print("\n=== MCP TOOLS LOADED ===")
        for t in tools:
            print(f"- {t.name}")
        print("=======================\n")

        return tools

    except Exception as e:
        print("\n=== MCP TOOL LOAD FAILED ===")
        print(str(e))
        traceback.print_exc()
        print("===========================\n")
        return []


mcp_tools = load_mcp_tools()
tools = [*mcp_tools]


# ==============================
# LLM Configuration
# ==============================

# Router LLM (decides RAG / web / direct answer)
router_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY,
).with_structured_output(RouteDecision)


# RAG Judge LLM (checks context sufficiency)
judge_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY,
).with_structured_output(RagJudge)


# Final Answer LLM (tool-enabled)
answer_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=GROQ_API_KEY,
).bind_tools(tools=tools)
