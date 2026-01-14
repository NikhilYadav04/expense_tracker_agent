# agent.py
import asyncio
import threading
from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.agent.state import AgentState
from app.agent.nodes import (
    router_node,
    rag_node,
    web_search,
    answer_node,
    tool_node,
)

# ============================================================
# 1. SQLITE CHECKPOINTER - GLOBAL INITIALIZATION
# ============================================================

# Initialize checkpointer synchronously at module level
# This avoids the background thread complexity
checkpointer = None


def init_checkpointer():
    """Initialize checkpointer in the main event loop"""
    global checkpointer
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    class PoolWrapper:
        """Wrapper to make aiosqlite.Connection compatible with LangGraph"""

        def __init__(self, db_path):
            self.db_path = db_path
            self._conn = None

        async def __aenter__(self):
            self._conn = await aiosqlite.connect(self.db_path)
            return self._conn

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self._conn:
                await self._conn.close()

        def is_alive(self):
            """LangGraph checks this method"""
            return self._conn is not None

        def __getattr__(self, name):
            """Delegate all other methods to the connection"""
            return getattr(self._conn, name)

    pool = PoolWrapper("N:/Dev/Langgraph-Project/Expense-Whatsapp/data/agent_state.db")
    return AsyncSqliteSaver(pool)


# ============================================================
# 2. ROUTING HELPERS
# ============================================================


def from_router(st: AgentState):
    return st["route"]


def after_rag(st: AgentState):
    return st["route"]


def after_web(st: AgentState):
    return "answer"


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ============================================================
# 3. BUILD AGENT
# ============================================================


async def build_agent():
    global checkpointer

    # Initialize checkpointer if not already done
    if checkpointer is None:
        checkpointer = init_checkpointer()
        # Open the connection pool
        await checkpointer.conn.__aenter__()
        # Setup tables
        await checkpointer.setup()

    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag_lookup", rag_node)
    graph.add_node("web_search", web_search)
    graph.add_node("answer", answer_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        from_router,
        {
            "rag": "rag_lookup",
            "web": "web_search",
            "answer": "answer",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "rag_lookup",
        after_rag,
        {"web": "web_search", "answer": "answer"},
    )

    graph.add_conditional_edges(
        "web_search",
        after_web,
        {"answer": "answer"},
    )

    graph.add_conditional_edges(
        "answer",
        should_continue,
        {"tools": "tools", END: END},
    )

    graph.add_edge("tools", "answer")

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )


# ============================================================
# 4. MAIN
# ============================================================


async def main():
    agent = await build_agent()

    config = {
        "configurable": {
            "thread_id": "user-9a56df08",
        }
    }

    initial_input = {
        "messages": [
            HumanMessage(
                content="What was the last thing i did?"
            )
        ]
    }

    # Run until pause or end
    async for _ in agent.astream(initial_input, config, stream_mode="values"):
        pass

    snapshot = await agent.aget_state(config)

    if snapshot.next:
        last_ai_msg = snapshot.values["messages"][-1]
        tool_call = last_ai_msg.tool_calls[0]

        print("\n⚠️ PAUSED")
        print("Tool:", tool_call["name"])

        approval = input("Approve? (yes/no): ").strip().lower()

        if approval == "yes":
            async for _ in agent.astream(None, config, stream_mode="values"):
                pass
        else:
            await agent.aupdate_state(
                config,
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=tool_call["id"],
                            content="User denied execution. Ask for next steps.",
                        )
                    ]
                },
                as_node="tools",
            )
            async for _ in agent.astream(None, config, stream_mode="values"):
                pass

    final_state = await agent.aget_state(config)
    print("\nFinal Assistant Response:")
    print(final_state.values["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
