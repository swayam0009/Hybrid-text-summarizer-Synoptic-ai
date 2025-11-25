# create_test_user.py
from utils.auth import AuthManager
from pathlib import Path

auth_manager = AuthManager(str(Path("data/database.db")))

# Try creating a test user
result = auth_manager.create_user_account(
    username="testuser",
    email="test@example.com",
    password="TestPass123!"
)

print("Account creation result:", result)
