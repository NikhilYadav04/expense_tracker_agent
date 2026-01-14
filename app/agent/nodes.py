from datetime import datetime
import os
from langchain.tools import tool
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig
from app.agent.state import AgentState
from app.agent.tools import rag_search_tool, web_search_tool
from app.agent.llm.llms import (
    RagJudge,
    RouteDecision,
    router_llm,
    judge_llm,
    tools,
    answer_llm,
)


trimmer = trim_messages(
    max_tokens=4000,  # The size of the "window"
    strategy="last",  # Keep the most RECENT messages
    token_counter=len,  # How to measure (characters/words/tokens)
    include_system=True,  # ALWAYS keep your Expense AI rules
    start_on="human",  # Start the history with a user question
)


# Router node to route to web | rag | llm | answer based on query
def router_node(state: AgentState) -> AgentState:
    print("Entering router node")

    # extract query
    query = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    web_search_enabled = state.get("web_search_enabled", True)

    print(f"Router received web search info : {web_search_enabled}")

    system_prompt = """
You are a routing controller for an expense tracking AI agent.

Your task is to choose the correct route for the user’s message.

ROUTES
- rag: Expense or bill-related knowledge, explanations, or document-based questions.
- web: Queries requiring very recent or external internet information.
- answer: Direct responses, clarification questions, OR requests that may require MCP expense tools (add, update, edit, delete, analytics).
- end: Greetings or small talk with no task intent.

RULES
- Do not answer the user.
- Do not mention or use tools.
- Do not explain your decision.

Output ONLY one route: rag | web | answer | end
"""

    messages = [("system", system_prompt), ("user", query)]

    result: RouteDecision = router_llm.invoke(messages)

    initial_router_decision = result.route
    router_override_reason = None

    # Override the router decision to go for web search
    if not web_search_enabled and result.route == "web":
        result.route = "rag"
        router_override_reason = "Web search disabled by user"
        print(f"Router decision overrriden : changed from 'web' to 'rag' ")
    print(f"Router final decision: {result.route}, Reply (if 'end'): {result.reply}")

    out = {
        "messages": state["messages"],
        "route": result.route,
        "web_search_enabled": web_search_enabled,
    }

    if router_override_reason:  # Add override info for tracing
        out["initial_router_decision"] = initial_router_decision
        out["router_override_reason"] = router_override_reason

    if result.route == "end":
        out["messages"] = state["messages"] + [
            AIMessage(content=result.reply or "Hello!")
        ]

    print("--- Exiting router_node ---")
    return out


# Web search node for web search
def web_search(state: AgentState) -> AgentState:
    print("Entering RAG node")

    # extract query
    query = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    web_search_enabled = state.get("web_search_enabled", True)

    print(f"Router received web search info : {web_search_enabled}")

    if not web_search_enabled:
        print("Web search node entered but search is disabled")

        return {**state, "web": "Web search was disabled by user", "route": "answer"}

    print(f"Web search query : {query}")

    snippets = web_search_tool.invoke(query)

    if snippets.startswith("WEB_ERROR::"):
        print(f"Web Error: {snippets}. Proceeding to answer with limited info.")
        return {**state, "web": "", "route": "answer"}

    print(f"Web snippets retrieved: {snippets[:200]}...")
    print("--- Exiting web_node ---")
    return {**state, "web": snippets, "route": "answer"}


# For RAG Fetch
def rag_node(state: AgentState) -> AgentState:
    print("Entering RAG node")

    # extract query
    query = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    web_search_enabled = state.get("web_search_enabled", True)

    print(f"Router received web search info : {web_search_enabled}")

    print(f"RAG Query : {query}")

    chunks = rag_search_tool.invoke(query)

    # logic to handle chunk

    if chunks.startswith("RAG_ERROR::"):
        print(f"RAG ERROR : {chunks},checking web search enabled status")

        # if rag fails, and web search is enabled
        next_route = "web" if web_search_enabled else "answer"

        return {**state, "rag": "", "route": next_route}
    if chunks:
        print(f"retrieved RAG chunks : {chunks[:500]}")
    else:
        print("No RAG chunks")

    judge_messages = [
        (
            "system",
            (
                "You are a judge evaluating if the **retrieved information** is **sufficient and relevant** "
                "to fully and accurately answer the user's question. "
                "Consider if the retrieved text directly addresses the question's core and provides enough detail."
                "If the information is incomplete, vague, outdated, or doesn't directly answer the question, it's NOT sufficient."
                "If it provides a clear, direct, and comprehensive answer, it IS sufficient."
                "If no relevant information was retrieved at all (e.g., 'No results found'), it is definitely NOT sufficient."
                '\n\nRespond ONLY with a JSON object: {"sufficient": true/false}'
                "\n\nExample 1: Question: 'What is the capital of France?' Retrieved: 'Paris is the capital of France.' -> {\"sufficient\": true}"
                "\nExample 2: Question: 'What are the symptoms of diabetes?' Retrieved: 'Diabetes is a chronic condition.' -> {\"sufficient\": false} (Doesn't answer symptoms)"
                "\nExample 3: Question: 'How to fix error X in software Y?' Retrieved: 'No relevant information found.' -> {\"sufficient\": false}"
            ),
        ),
        (
            "user",
            f"Question: {query}\n\nRetrieved info: {chunks}\n\nIs this sufficient to answer the question?",
        ),
    ]

    verdict: RagJudge = judge_llm.invoke(judge_messages)
    print(f"RAG Judge verdict: {verdict.sufficient}")
    print("--- Exiting rag_node ---")

    #  Decide next route based on sufficiency AND web_search_enabled
    if verdict.sufficient:
        next_route = "answer"
    else:
        next_route = (
            "web" if web_search_enabled else "answer"
        )  # If not sufficient, only go to web if enabled
        print(
            f"RAG not sufficient. Web search enabled: {web_search_enabled}. Next route: {next_route}"
        )

    return {
        **state,
        "rag": chunks,
        "route": next_route,
        "web_search_enabled": web_search_enabled,
    }


# MCP Tools Node

base_tool_node = ToolNode(tools=tools, handle_tool_errors=True)


async def tool_node(state: AgentState, config: RunnableConfig):
    """ """
    print(f"--- Executing Tools for User: {config['configurable'].get('user_id')} ---")
    return await base_tool_node.ainvoke(state, config)


# Answer Node ( MCP Tools + LLM Answer)
def answer_node(state: AgentState) -> AgentState:
    print("Entering answer_node")

    # 1. Enhanced System Prompt
    answer_system_prompt = f"""
# IDENTITY
You are a professional Expense AI. Current Date: {datetime.now().strftime('%A, %B %d, %Y')}

# OPERATIONAL RULES (TOOL CALLING)
1. PARAMETERS: You MUST provide ALL required input parameters for every tool (amount, category, date, source, user_id). 
   - If the user misses a field, fill it yourself using your best judgment from the context (e.g., today's date).
   - Do not ask for missing fields unless you are completely unable to guess.
2. CONSTRAINTS: Only use valid sources: 'cash' or 'upi'. Anchor all dates to the Current Date above.
3. DECISION: Stay in TOOL MODE if more data is needed to complete the request. Switch to ANSWER MODE only when you have the final result.

#. DATE LOGIC:
   - PRIORITIZE: If the user mentions a specific date or time period (e.g., "last year", "last Tuesday", "Jan 2025"), calculate the exact date relative to the Current Date and use it.
   - FALLBACK: If the user provides NO date information, use the Current Date as the default. 
   - FORMAT: Always convert relative dates to YYYY-MM-DD for tool inputs.

# RESPONSE GUIDELINES (FINAL ANSWER)
4. DATA SYNTHESIS: You will receive data from Database Tools, Internal Docs (RAG), or Web Search. Synthesize these into a cohesive, professional response.
5. SOURCE-BASED FEEDBACK:
   - FROM RAG: Explain policies or internal info clearly.
   - FROM WEB: Summarize the latest external info found.
   - FROM TOOLS: Confirm successful actions in plain English (e.g., "Confirmed: Added $30 Uber ride").
6. ERROR HANDLING: If a tool fails, explain the error simply and ask for missing/corrected info. Do not repeat failed tool calls.
7. NO REDUNDANCY: Once the task is confirmed or answered, stop all tool calls and continue as a plain conversation.
"""

    # 2. Prepare history
    trimmed_messages = trimmer.invoke(state["messages"])
    messages_for_llm = [SystemMessage(content=answer_system_prompt)] + trimmed_messages

    try:
        # 3. Call the LLM
        response = answer_llm.invoke(messages_for_llm)

        # 4. Handle 'Silent' LLM or Groq Glitches
        if not response.content and not response.tool_calls:
            # Check if we just finished a tool call
            if isinstance(state["messages"][-1], ToolMessage):
                return {
                    "messages": [
                        AIMessage(
                            content="I've processed that for you. Everything looks good!"
                        )
                    ]
                }
            return {
                "messages": [
                    AIMessage(
                        content="I'm here to help. What would you like to do with your expenses?"
                    )
                ]
            }

        return {"messages": [response]}

    except Exception as e:
        print(f"Caught LLM Error: {e}")
        # The 'Ultimate Safety' response prevents the app from crashing on Groq 400 errors
        return {
            "messages": [
                AIMessage(
                    content="Action completed successfully, but I had trouble generating a summary. Your records are updated!"
                )
            ]
        }


# Prompts

ANSWER_SYSTEM_PROMPT = """
You are a helpful AI assistant.
Current Date: January 14, 2026.

You may:
- Answer directly if confident.
- Request a tool ONLY if absolutely necessary.

If a tool successfully returns a status of 'success', you MUST respond with a natural language confirmation. 
   Example: "Great! I've added that $30 transportation expense for Uber to your records."

IMPORTANT:
- Tools cause real side effects.
- If a tool is required, request it clearly with correct arguments.
- If unsure, ask a clarification question instead of using a tool.
"""
