# utils/__init__.py
"""
Utility modules for the Hybrid Text Summarization System

This package contains utility modules for file handling, authentication,
and other supporting functionality for the text summarization system.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .file_handler import FileHandler
from .auth import AuthManager

__all__ = [
    'FileHandler', 
    'AuthManager'
]

# Package-level constants
SUPPORTED_FILE_TYPES = ['.txt', '.pdf', '.docx', '.doc', '.html', '.htm']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_URL_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB

# Authentication constants
SESSION_TIMEOUT_HOURS = 24
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30

# Password requirements
PASSWORD_MIN_LENGTH = 8
PASSWORD_REQUIREMENTS = {
    'min_length': PASSWORD_MIN_LENGTH,
    'requires_uppercase': True,
    'requires_lowercase': True,
    'requires_digit': True,
    'requires_special': True,
    'special_characters': '!@#$%^&*()_+-=[]{}|;:,.<>?'
}
