from mcp.server.fastmcp import FastMCP
from langchain_core.runnables import RunnableConfig
from app.db.connection import get_supabase


# --------------------
# Init MCP Server
# --------------------
mcp = FastMCP("expense-mcp")

# --------------------
# Supabase Client
# --------------------
supabase = get_supabase()


# ======================================================
# ADD EXPENSE
# ======================================================
@mcp.tool()
def add_expense(
    user_id: str,
    amount: float,
    category: str,
    expense_date: str,
    source: str,
    merchant: str | None = None,
    note: str | None = None,
    config: RunnableConfig = None,
):
    """Add a new expense"""

    data = {
        "user_id": user_id,
        "amount": amount,
        "category": category,
        "expense_date": expense_date,
        "source": source,
        "merchant": merchant,
        "note": note,
    }

    result = supabase.table("expenses").insert(data).execute()

    return {"status": "success", "expense_id": result.data[0]["id"]}


# ======================================================
# GET EXPENSES (DATE RANGE)
# ======================================================
@mcp.tool()
def get_expenses(user_id: str, from_date: str, to_date: str):
    """Get expenses for a user in date range"""

    result = (
        supabase.table("expenses")
        .select("*")
        .eq("user_id", user_id)
        .gte("expense_date", from_date)
        .lte("expense_date", to_date)
        .order("expense_date", desc=True)
        .execute()
    )

    total = sum(e["amount"] for e in result.data)

    return {"total": total, "count": len(result.data), "expenses": result.data}


# ======================================================
# UPDATE EXPENSE
# ======================================================
@mcp.tool()
def update_expense(
    expense_id: str,
    user_id: str,
    amount: float | None = None,
    category: str | None = None,
    expense_date: str | None = None,
    merchant: str | None = None,
    note: str | None = None,
):
    """Update an existing expense"""

    update_data = {
        k: v
        for k, v in {
            "amount": amount,
            "category": category,
            "expense_date": expense_date,
            "merchant": merchant,
            "note": note,
        }.items()
        if v is not None
    }

    if not update_data:
        return {"status": "no_changes"}

    result = (
        supabase.table("expenses")
        .update(update_data)
        .eq("id", expense_id)
        .eq("user_id", user_id)
        .execute()
    )

    if not result.data:
        return {"status": "not_found"}

    return {"status": "success", "updated_expense": result.data[0]}


# ======================================================
# DELETE EXPENSE
# ======================================================
@mcp.tool()
def delete_expense(expense_id: str, user_id: str):
    """Delete an expense"""

    result = (
        supabase.table("expenses")
        .delete()
        .eq("id", expense_id)
        .eq("user_id", user_id)
        .execute()
    )

    if not result.data:
        return {"status": "not_found"}

    return {"status": "success", "deleted_expense_id": expense_id}


# ======================================================
# CLEAR ALL EXPENSES
# ======================================================
@mcp.tool()
def clear_all_expenses(user_id: str, confirm: bool = False):
    """
    Delete all expenses for a specific user.
    Requires confirm=True to prevent accidental deletion.
    """
    if not confirm:
        return {
            "status": "error",
            "message": "Please set confirm=True to delete all data.",
        }

    # Supabase requires a filter for delete. We target all rows for this user_id.
    result = supabase.table("expenses").delete().eq("user_id", user_id).execute()

    count = len(result.data) if result.data else 0

    return {
        "status": "success",
        "message": f"Successfully deleted {count} expenses.",
        "deleted_count": count,
    }


if __name__ == "__main__":
    # You MUST specify transport="stdio" for Claude Desktop to communicate
    mcp.run(transport="stdio")
