from mcp.server.fastmcp import FastMCP
from app.db.connection import get_supabase

mcp = FastMCP("analytics-mcp")
supabase = get_supabase()


# ======================================================
# MONTHLY SUMMARY
# ======================================================
@mcp.tool()
def monthly_summary(user_id: str, from_date: str, to_date: str):
    """Total spend for a date range"""

    res = (
        supabase.table("expenses")
        .select("amount")
        .eq("user_id", user_id)
        .gte("expense_date", from_date)
        .lte("expense_date", to_date)
        .execute()
    )

    total = sum(e["amount"] for e in res.data)

    return {"total_spent": total, "count": len(res.data)}


# ======================================================
# CATEGORY BREAKDOWN
# ======================================================
@mcp.tool()
def category_breakdown(user_id: str, from_date: str, to_date: str):
    """Spend grouped by category"""

    res = (
        supabase.table("expenses")
        .select("category, amount")
        .eq("user_id", user_id)
        .gte("expense_date", from_date)
        .lte("expense_date", to_date)
        .execute()
    )

    breakdown = {}

    for e in res.data:
        breakdown[e["category"]] = breakdown.get(e["category"], 0) + e["amount"]

    return breakdown


# ======================================================
# HIGHEST SINGLE EXPENSE
# ======================================================
@mcp.tool()
def highest_spend(user_id: str, from_date: str, to_date: str):
    """Largest single expense"""

    res = (
        supabase.table("expenses")
        .select("*")
        .eq("user_id", user_id)
        .gte("expense_date", from_date)
        .lte("expense_date", to_date)
        .order("amount", desc=True)
        .limit(1)
        .execute()
    )

    if not res.data:
        return {"exists": False}

    return {"exists": True, "expense": res.data[0]}


# ======================================================
# CHECK CATEGORY LIMIT
# ======================================================
@mcp.tool()
def check_category_limit(user_id: str, category: str, from_date: str, to_date: str):
    """Check if category monthly limit exceeded"""

    # 1️⃣ Get total spent in category
    spend_res = (
        supabase.table("expenses")
        .select("amount")
        .eq("user_id", user_id)
        .eq("category", category)
        .gte("expense_date", from_date)
        .lte("expense_date", to_date)
        .execute()
    )

    total_spent = sum(e["amount"] for e in spend_res.data)

    # 2️⃣ Get category limit
    limit_res = (
        supabase.table("categories")
        .select("monthly_limit")
        .eq("user_id", user_id)
        .eq("name", category)
        .execute()
    )

    if not limit_res.data or limit_res.data[0]["monthly_limit"] is None:
        return {"has_limit": False, "total_spent": total_spent}

    limit = limit_res.data[0]["monthly_limit"]

    return {
        "has_limit": True,
        "limit": limit,
        "total_spent": total_spent,
        "exceeded": total_spent > limit,
    }


if __name__ == "__main__":
    # You MUST specify transport="stdio" for Claude Desktop to communicate
    mcp.run(transport="stdio")
