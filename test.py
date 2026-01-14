import os
from supabase import create_client

# Replace with your actual credentials
url = "https://pkrbslarbvlpqyqmoxsl.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBrcmJzbGFyYnZscHF5cW1veHNsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2Nzg5MDYzOCwiZXhwIjoyMDgzNDY2NjM4fQ.EZY6LbKZncwWFzuL7eLfz1y_tuURYd2c3h_okHSfang"

supabase = create_client(url, key)


def test_insert():
    test_data = {
        "user_id": "d4a540cd-c17f-4c3d-9ec6-e7cff177d09d",  # Use a valid UUID format
        "amount": 10.50,
        "category": "Test",
        "expense_date": "2024-05-20",
        "source": "Manual",
        "merchant": "Test Merchant",  # This is the column causing the error
        "note": "Testing schema refresh",
    }

    try:
        # We use select() to see what the API returns immediately
        response = supabase.table("expenses").insert(test_data).execute()
        print("✅ Success! The API sees the merchant column.")
        print("Data:", response.data)
    except Exception as e:
        print("❌ Failed!")
        print(f"Error Message: {e}")


if __name__ == "__main__":
    test_insert()
