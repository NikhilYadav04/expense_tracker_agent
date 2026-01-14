import os
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from app.agent.vectorstore.vectorstore import get_retriever
from app.agent.config.config import TAVILY_API_KEY

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

tavily = TavilySearch(max_results=3, topic="general")


@tool
def rag_search_tool(query: str) -> str:
    """Top-K chunks from KB (empty string if none)"""
    try:
        retriever_instance = get_retriever()
        docs = retriever_instance.invoke(query, k=5)
        return "\n\n".join(d.page_content for d in docs) if docs else ""
    except Exception as e:
        return f"RAG_ERROR::{e}"


@tool
def web_search_tool(query: str) -> str:
    """Up-to-date web info via Tavily"""
    try:
        result = tavily.invoke({"query": query})
        if isinstance(result, dict) and "results" in result:
            formatted_results = []
            for item in result["results"]:
                title = item.get("title", "No title")
                content = item.get("content", "No content")
                url = item.get("url", "")
                formatted_results.append(
                    f"Title: {title}\nContent: {content}\nURL: {url}"
                )
            return (
                "\n\n".join(formatted_results)
                if formatted_results
                else "No results found"
            )
        else:
            return str(result)
    except Exception as e:
        return f"WEB_ERROR::{e}"
