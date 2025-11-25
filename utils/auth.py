# utils/auth.py
import hashlib
import secrets
from typing import Dict, Any, Optional, Tuple
import sqlite3
from datetime import datetime, timedelta
import streamlit as st

class AuthManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.session_timeout = timedelta(hours=24)
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        # Generate a random salt
        salt = secrets.token_hex(16)
        # Hash password with salt
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        # Return salt + hash
        return salt + pwd_hash.hex()
    
    def verify_password(self, password: str, hash_with_salt: str) -> bool:
        """Verify password against hash"""
        try:
            # Extract salt (first 32 characters)
            salt = hash_with_salt[:32]
            stored_hash = hash_with_salt[32:]
            
            # Hash provided password with stored salt
            pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            
            return pwd_hash.hex() == stored_hash
        except Exception:
            return False
    
    def create_user_account(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Create new user account"""
        try:
            # Validate input
            if not self._validate_username(username):
                return {'success': False, 'error': 'Invalid username'}
            
            if not self._validate_email(email):
                return {'success': False, 'error': 'Invalid email'}
            
            if not self._validate_password(password):
                return {'success': False, 'error': 'Password does not meet requirements'}
            
            # Check if user already exists
            if self._user_exists(username, email):
                return {'success': False, 'error': 'User already exists'}
            
            # Hash password
            password_hash = self.hash_password(password)
            
            # Create user in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash)
                    VALUES (?, ?, ?)
                ''', (username, email, password_hash))
                
                user_id = cursor.lastrowid
                
                # Create default user profile
                cursor.execute('''
                    INSERT INTO user_profiles (user_id, domain_expertise, preferences)
                    VALUES (?, ?, ?)
                ''', (user_id, 'general', '{}'))
                
                conn.commit()
            
            return {
                'success': True,
                'user_id': user_id,
                'username': username,
                'email': email
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Account creation failed: {str(e)}'}
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user login"""
        try:
            # Check if user is locked out
            if self._is_user_locked_out(username):
                return {'success': False, 'error': 'Account temporarily locked due to too many failed attempts'}
            
            # Get user from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, username, email, password_hash
                    FROM users
                    WHERE username = ?
                ''', (username,))
                
                user = cursor.fetchone()
                
                if not user:
                    self._record_failed_login(username)
                    return {'success': False, 'error': 'Invalid username or password'}
                
                user_id, username, email, stored_hash = user
                
                # Verify password
                if not self.verify_password(password, stored_hash):
                    self._record_failed_login(username)
                    return {'success': False, 'error': 'Invalid username or password'}
                
                # Clear failed login attempts
                self._clear_failed_login_attempts(username)
                
                # Create session
                session_token = self._create_session(user_id)
                
                return {
                    'success': True,
                    'user_id': user_id,
                    'username': username,
                    'email': email,
                    'session_token': session_token
                }
                
        except Exception as e:
            return {'success': False, 'error': f'Authentication failed: {str(e)}'}
    
    def validate_session(self, session_token: str) -> Dict[str, Any]:
        """Validate user session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_id, username, created_at
                    FROM user_sessions
                    WHERE session_token = ? AND expires_at > ?
                ''', (session_token, datetime.now().isoformat()))
                
                session = cursor.fetchone()
                
                if not session:
                    return {'success': False, 'error': 'Invalid or expired session'}
                
                user_id, username, created_at = session
                
                # Update session expiration
                cursor.execute('''
                    UPDATE user_sessions
                    SET expires_at = ?
                    WHERE session_token = ?
                ''', ((datetime.now() + self.session_timeout).isoformat(), session_token))
                
                conn.commit()
                
                return {
                    'success': True,
                    'user_id': user_id,
                    'username': username,
                    'session_created': created_at
                }
                
        except Exception as e:
            return {'success': False, 'error': f'Session validation failed: {str(e)}'}
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user and invalidate session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM user_sessions
                    WHERE session_token = ?
                ''', (session_token,))
                
                conn.commit()
                
                return True
                
        except Exception:
            return False
    
    def change_password(self, user_id: int, old_password: str, new_password: str) -> Dict[str, Any]:
        """Change user password"""
        try:
            # Validate new password
            if not self._validate_password(new_password):
                return {'success': False, 'error': 'New password does not meet requirements'}
            
            # Get current password hash
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT password_hash FROM users WHERE id = ?
                ''', (user_id,))
                
                result = cursor.fetchone()
                if not result:
                    return {'success': False, 'error': 'User not found'}
                
                current_hash = result[0]
                
                # Verify old password
                if not self.verify_password(old_password, current_hash):
                    return {'success': False, 'error': 'Current password is incorrect'}
                
                # Hash new password
                new_hash = self.hash_password(new_password)
                
                # Update password
                cursor.execute('''
                    UPDATE users
                    SET password_hash = ?
                    WHERE id = ?
                ''', (new_hash, user_id))
                
                conn.commit()
                
                return {'success': True, 'message': 'Password updated successfully'}
                
        except Exception as e:
            return {'success': False, 'error': f'Password change failed: {str(e)}'}
    
    def _validate_username(self, username: str) -> bool:
        """Validate username format"""
        if not username or len(username) < 3 or len(username) > 20:
            return False
        
        # Allow alphanumeric characters and underscores
        return username.replace('_', '').isalnum()
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_password(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < 8:
            return False
        
        # Check for at least one uppercase, lowercase, digit, and special character
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def _user_exists(self, username: str, email: str) -> bool:
        """Check if user already exists"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM users
                WHERE username = ? OR email = ?
            ''', (username, email))
            
            return cursor.fetchone() is not None
    
    def _create_session(self, user_id: int) -> str:
        """Create user session"""
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + self.session_timeout
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create sessions table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            cursor.execute('''
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, session_token, expires_at.isoformat()))
            
            conn.commit()
        
        return session_token
    
    def _record_failed_login(self, username: str):
        """Record failed login attempt"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create failed_logins table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS failed_logins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                INSERT INTO failed_logins (username)
                VALUES (?)
            ''', (username,))
            
            conn.commit()
    
    def _is_user_locked_out(self, username: str) -> bool:
        """Check if user is locked out due to failed attempts"""
        cutoff_time = datetime.now() - self.lockout_duration
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM failed_logins
                WHERE username = ? AND attempt_time > ?
            ''', (username, cutoff_time.isoformat()))
            
            count = cursor.fetchone()[0]
            return count >= self.max_login_attempts
    
    def _clear_failed_login_attempts(self, username: str):
        """Clear failed login attempts for user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM failed_logins
                WHERE username = ?
            ''', (username,))
            
            conn.commit()
    
    def get_password_requirements(self) -> Dict[str, Any]:
        """Get password requirements"""
        return {
            'min_length': 8,
            'requires_uppercase': True,
            'requires_lowercase': True,
            'requires_digit': True,
            'requires_special': True,
            'special_characters': '!@#$%^&*()_+-=[]{}|;:,.<>?'
        }
