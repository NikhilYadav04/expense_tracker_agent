from typing import Annotated, List, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages  # <--- Import this!


class AgentState(TypedDict):
    # The 'Annotated' wrapper with 'add_messages' is the secret sauce
    messages: Annotated[List[BaseMessage], add_messages]

    route: Literal["rag", "web", "answer", "end"]
    rag: str
    web: str
    web_search_enabled: bool

    tool_retry_count: int
