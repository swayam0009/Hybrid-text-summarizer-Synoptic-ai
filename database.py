# database.py
import sqlite3
from pathlib import Path
from datetime import datetime
import pickle
import json
from typing import Dict, List, Any, Optional

class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # User profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    domain_expertise TEXT DEFAULT 'general',
                    reading_speed INTEGER DEFAULT 200,
                    detail_preference REAL DEFAULT 0.5,
                    summary_length_preference INTEGER DEFAULT 3,
                    technical_level REAL DEFAULT 0.5,
                    preferences TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Summarization history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS summarization_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    document_title TEXT,
                    document_content TEXT,
                    summary TEXT,
                    summary_type TEXT,
                    summary_length INTEGER,
                    importance_scores TEXT,
                    personalization_factors TEXT,
                    feedback_rating INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # User feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    summary_id INTEGER NOT NULL,
                    rating INTEGER,
                    feedback_text TEXT,
                    improvement_suggestions TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (summary_id) REFERENCES summarization_history (id)
                )
            ''')
            
            conn.commit()
    
    def create_user(self, username: str, email: str, password_hash: str) -> int:
        """Create a new user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password_hash)
            )
            return cursor.lastrowid
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0], 'username': row[1], 'email': row[2],
                    'password_hash': row[3], 'created_at': row[4]
                }
            return None
    
    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """Get user profile by user ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0], 'user_id': row[1], 'domain_expertise': row[2],
                    'reading_speed': row[3], 'detail_preference': row[4],
                    'summary_length_preference': row[5], 'technical_level': row[6],
                    'preferences': json.loads(row[7]) if row[7] else {},
                    'created_at': row[8], 'updated_at': row[9]
                }
            return None
    
    def create_user_profile(self, user_id: int, profile_data: Dict) -> int:
        """Create user profile"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_profiles 
                (user_id, domain_expertise, reading_speed, detail_preference, 
                 summary_length_preference, technical_level, preferences)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, profile_data.get('domain_expertise', 'general'),
                profile_data.get('reading_speed', 200),
                profile_data.get('detail_preference', 0.5),
                profile_data.get('summary_length_preference', 3),
                profile_data.get('technical_level', 0.5),
                json.dumps(profile_data.get('preferences', {}))
            ))
            return cursor.lastrowid
    
    def save_summarization_history(self, user_id: int, summary_data: Dict) -> int:
        """Save summarization history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO summarization_history 
                (user_id, document_title, document_content, summary, summary_type,
                 summary_length, importance_scores, personalization_factors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, summary_data.get('document_title'),
                summary_data.get('document_content'),
                summary_data.get('summary'),
                summary_data.get('summary_type'),
                summary_data.get('summary_length'),
                json.dumps(summary_data.get('importance_scores', {})),
                json.dumps(summary_data.get('personalization_factors', {}))
            ))
            return cursor.lastrowid
    
    def save_user_feedback(self, user_id: int, summary_id: int, feedback_data: Dict) -> int:
        """Save user feedback"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_feedback 
                (user_id, summary_id, rating, feedback_text, improvement_suggestions)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id, summary_id, feedback_data.get('rating'),
                feedback_data.get('feedback_text'),
                feedback_data.get('improvement_suggestions')
            ))
            return cursor.lastrowid
