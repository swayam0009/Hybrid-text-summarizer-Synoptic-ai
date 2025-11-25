# test_login_user.py
from utils.auth import AuthManager
from pathlib import Path

auth_manager = AuthManager(str(Path("data/database.db")))

# Test login with the test user
result = auth_manager.authenticate_user("testuser", "TestPass123!")
print("Login result:", result)
